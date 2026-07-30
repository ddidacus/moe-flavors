<<<<<<< HEAD
"""Router-distribution eval at the cache layer for the soft-cache run.

Loads the latest adapter checkpoint of the dense-router (soft-cache) GRPO run
and, for both the tuned model and the frozen base (adapter disabled), samples
256 on-policy completions (T=1.0, as in training rollouts) from held-out
Nemotron-v2 math/code prompts. Only these on-policy generations are analyzed
-- there is no teacher-forced mode; each variant is scored on its own
generations, matching what the policy actually produces at inference/rollout
time.

Layer parity (requirement): the cache layer's router is monkeypatched dense
(K = all experts, full softmax) once, on the shared base module, via
dense_patch_router() -- so BOTH the tuned forward and the base forward
(adapter disabled) read router_logits at the exact same layer (--cache-layer,
default = the middle layer, matching training's default). The baseline is
never allowed to look at a different layer than the one the cache reward was
trained on.

Per action token (generated tokens; or the whole sequence when teacher-forced
with prompt_lens all 0) we compute four routing-distribution metrics at the
cache layer:
  variance    Var_e[p_e], the spread of the full softmax over ALL experts
              (not just the top-k) -- population variance in probability
              space, not entropy.
  kl_uniform  KL(p || Uniform(E)) = log(E) - H(p): distance of the routing
              distribution to the uniform distribution over the same E
              experts.
  skew        Fisher-Pearson standardized skewness coefficient of the E
              per-expert probabilities, m3 / m2^1.5.
  hit_ratio   soft LRU cache-hit mass: the probability mass this token's
              routing distribution places on the experts *currently* in the
              per-sequence LRU working set (as-of the previous token); the
              LRU is touched by the top --touch-topk experts each step, as in
              training. This is the routing-weight-weighted hit ratio, per
              token (not divided by sequence length T, unlike the training
              reward).

Also reports the mean contiguous run length of the top-1 expert per sequence
(run_lengths()) -- a direct measure of whether the cache reward makes the
policy hold the same expert over consecutive tokens ("temporal
consolidation") -- and a standard teacher-forced perplexity on held-out
ground-truth completions (compute_perplexity(), independent of the routing
analysis above), so cache-reward gains can be checked against language
modeling quality.

Outputs (to --out-dir):
  metrics.json        aggregate (mean/median/min/max/std) + case-study numbers
                       + expert_run_length + perplexity
  expert_probs.png    expected per-expert routing probability, base vs tuned
  skew.png            sorted cumulative expert mass (Lorenz) + per-seq mean
                       hit-ratio histograms
  perplexity.png      held-out teacher-forced perplexity, base vs tuned
  aggregate/*.png      one grouped bar chart per metric: mean/median/min/max/
                       std, base vs tuned
  case_study/*.png     one figure per metric: a stack of small histograms,
                       one per random held-out sequence, each showing that
                       sequence's metric value at 4 equally spaced tokens
                       (first, 1/4, 3/4, last), base vs tuned grouped bars
  expert_trace/*.png   top-1 expert id per generated token as a color strip,
                       one (base, tuned) row pair per random held-out
                       sequence -- contiguous same-color runs show the
                       policy holding the same expert over consecutive
                       tokens (temporal consolidation)
  expert_trace/sequences.txt   text form of the same case-study sequences:
                       the decoded text, and the same text with '|' inserted
                       at every top-1-expert change -- token-level text
                       equivalent of expert_trace.png's color segments
"""

import argparse
import json
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from src.cache_reinforce import LRUExpertCache, cache_emulation_rewards

# ground-truth pool builder for the perplexity check ONLY -- unrelated to the
# (removed) teacher-forced routing/cache analysis mode.
from finetune_moe_grpo import build_eval_sequences  # noqa: E402

METRICS = ("variance", "kl_uniform", "skew", "hit_ratio")
METRIC_LABELS = {
    "variance": "Var_e[p_e]  (routing-distribution variance)",
    "kl_uniform": "KL(p || uniform)  (nats)",
    "skew": "Fisher-Pearson skewness of p_e",
    "hit_ratio": "soft LRU hit ratio (routing-weighted)",
}


def build_eval_prompts(tokenizer, dataset_name, split, n_total, prompt_len,
                       seed, pool_per_split=1000):
    """Held-out prompts (pre-assistant turns + generation prompt) from the
    reserved eval pool, truncated to prompt_len tokens."""
    from datasets import load_dataset

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = n_total // len(splits)
    rng = random.Random(seed)
    prompts = []
    for sp in splits:
        ds = load_dataset(dataset_name, split=sp, streaming=True)
        pool = []
        for row in ds:
            pre = []
            for m in row["messages"]:
                if m["role"] == "assistant":
                    break
                if m["content"].strip():
                    pre.append(m)
            if pre:
                pool.append(pre)
            if len(pool) >= pool_per_split:
                break
        for msgs in rng.sample(pool, min(per_split, len(pool))):
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            ids = tokenizer(text, truncation=True, max_length=prompt_len,
                            add_special_tokens=False)["input_ids"]
            prompts.append(ids)
    return prompts


@torch.no_grad()
def generate_seqs(model, prompts, tok, gen_len, batch_size, device):
    """Sample gen_len tokens per prompt (temperature 1.0, as in training
    rollouts). Returns (full sequences, prompt lengths)."""
    seqs, plens = [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        maxp = max(len(p) for p in batch)
        ids = torch.full((len(batch), maxp), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for j, p in enumerate(batch):  # left padding for generation
            ids[j, maxp - len(p):] = torch.tensor(p)
            attn[j, maxp - len(p):] = 1
        out = model.generate(
            input_ids=ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=gen_len, do_sample=True, temperature=1.0,
            pad_token_id=tok.pad_token_id)
        for j, p in enumerate(batch):
            gen = [t for t in out[j, maxp:].tolist() if t != tok.pad_token_id]
            seqs.append(p + gen)
            plens.append(len(p))
        print(f"[gen] {min(i + batch_size, len(prompts))}/{len(prompts)}",
              flush=True)
    return seqs, plens


def dense_patch_router(model, cache_layer):
    """Monkeypatch the cache layer's router to dense (K = all experts, full
    softmax) on the shared base module, so both the tuned and the
    adapter-disabled base forward read the SAME layer densely -- the
    baseline never gets to look at a different (or sparse) layer."""
    def _dense_forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        routing_weights = torch.softmax(router_logits.float(), dim=-1) \
            .to(hidden_states.dtype)
        selected = torch.arange(router_logits.shape[-1],
                                device=router_logits.device) \
            .expand(router_logits.shape[0], -1)
        return router_logits, routing_weights, selected

    patched = []
    for name, mod in model.named_modules():
        if (type(mod).__name__ == "PhimoeTopKRouter"
                and f"layers.{cache_layer}.mlp" in name):
            mod.forward = types.MethodType(_dense_forward, mod)
            patched.append(name)
    assert len(patched) == 1, f"expected 1 router at layer {cache_layer}: {patched}"
    print(f"[eval] dense routing patched on {patched[0]} "
          f"(base and tuned forwards share this module -> same layer)")


def load_adapter(peft_model, ckpt_dir):
    """Manual adapter load (peft 0.19 can't load ParamWrapper adapters)."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    model_keys = set(peft_model.state_dict().keys())
    remapped = {}
    for k, v in sd.items():
        nk = k.replace(".lora_A.weight", ".lora_A.default.weight") \
              .replace(".lora_B.weight", ".lora_B.default.weight")
        if nk not in model_keys:
            head, _, tail = nk.rpartition(".")
            cand = f"{head}.modules_to_save.default.{tail}"
            if cand in model_keys:
                nk = cand
        remapped[nk] = v
    res = peft_model.load_state_dict(remapped, strict=False)
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys[:5]}"
    missing = [k for k in res.missing_keys if "lora" in k]
    assert not missing, f"missing lora keys: {missing[:5]}"
    print(f"[eval] loaded {len(remapped)} adapter tensors from {ckpt_dir}")


@torch.no_grad()
def analyze(model, seqs, prompt_lens, pad_id, layers, cache_size,
            touch_topk, batch_size, device, temporal_wrappers=None):
    """Per-action-token routing-distribution metrics, computed at every
    layer in `layers` from a single shared forward pass per batch (prompt
    tokens warm each layer's own LRU but are not scored; prompt_lens of all
    zeros = teacher-forced whole-sequence scoring). Each layer gets its own
    independent LRU-cache simulation (cache_size, touch_topk) -- for layers
    other than the one trained with the cache reward, this uses that layer's
    native (sparse, un-patched) router logits, i.e. the real top-k it would
    use at inference.

    Returns: dict layer -> (expert_probs (E,), flat, per_seq), where
        expert_probs: (E,) mean routing probability per expert (action
            tokens only).
        flat: dict metric name -> 1D tensor, concatenated over all action
            tokens of all sequences (for the aggregate mean/median/min/
            max/std stats).
        per_seq: list (one entry per input sequence, in order) of dict
            metric name -> 1D tensor over that sequence's own action tokens
            (for the token-position case study).
    """
    prob_sum = {l: None for l in layers}
    n_tok = {l: 0 for l in layers}
    flat = {l: {m: [] for m in METRICS} for l in layers}
    per_seq = {l: [] for l in layers}

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        plens = prompt_lens[i:i + batch_size]
        S = max(len(s) for s in batch)
        ids = torch.full((len(batch), S), pad_id, dtype=torch.long)
        valid = torch.zeros(len(batch), S, dtype=torch.bool)
        action = torch.zeros(len(batch), S, dtype=torch.bool)
        for j, (s, pl) in enumerate(zip(batch, plens)):
            ids[j, :len(s)] = torch.tensor(s)
            valid[j, :len(s)] = True
            action[j, pl:len(s)] = True
        ids_d, valid_d = ids.to(device), valid.to(device)
        out = model(input_ids=ids_d, attention_mask=valid_d.long(),
                    output_router_logits=True, use_cache=False)

        for layer in layers:
            if temporal_wrappers is not None and layer in temporal_wrappers:
                # Held (forward-filled) full-E logits actually driving each
                # token's computation, not each token's own fresh logits --
                # see TemporalMoEWrapper._last_held_router_logits.
                logits = temporal_wrappers[layer]._last_held_router_logits \
                    .float().cpu()
            else:
                logits = out.router_logits[layer].view(len(batch), S, -1) \
                    .float().cpu()
            probs = torch.softmax(logits, dim=-1)
            E = probs.shape[-1]

            m = action.unsqueeze(-1)
            batch_prob_sum = (probs * m).sum((0, 1))
            prob_sum[layer] = batch_prob_sum if prob_sum[layer] is None \
                else prob_sum[layer] + batch_prob_sum
            n_tok[layer] += int(action.sum())

            mean_p = probs.mean(-1, keepdim=True)
            diffs = probs - mean_p
            var_t = diffs.pow(2).mean(-1)                          # (b, S)
            m3 = diffs.pow(3).mean(-1)
            skew_t = m3 / var_t.clamp_min(1e-12).pow(1.5)           # Fisher-Pearson
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            kl_t = float(torch.log(torch.tensor(float(E)))) - ent   # KL(p||uniform)

            for j, (s, pl) in enumerate(zip(batch, plens)):
                n = len(s)
                cache = LRUExpertCache(cache_size)
                hit_seq = torch.zeros(n)
                top1_seq = torch.zeros(n, dtype=torch.long)
                topc_seq = [None] * n
                act_idx = []
                for t in range(n):
                    if not valid[j, t]:
                        continue
                    if action[j, t]:
                        cached = cache.experts
                        hit_seq[t] = float(probs[j, t, cached].sum()) if cached else 0.0
                        top1_seq[t] = int(probs[j, t].argmax())
                        # top-cache_size active-expert *set* at this token --
                        # used for run-length/switch-rate instead of the bare
                        # top-1 id, since strict top-1 equality over-counts
                        # switches when two already-warm experts merely trade
                        # rank #1 turn to turn.
                        topc_seq[t] = frozenset(
                            probs[j, t].topk(cache_size).indices.tolist())
                        act_idx.append(t)
                    for e in probs[j, t].topk(touch_topk).indices.tolist():
                        cache.access(e)
                idx = torch.tensor(act_idx, dtype=torch.long)
                seq_metrics = {
                    "variance": var_t[j, idx],
                    "kl_uniform": kl_t[j, idx],
                    "skew": skew_t[j, idx],
                    "hit_ratio": hit_seq[idx],
                    # top-1 expert id per action token, full resolution (not
                    # one of METRICS -- excluded from aggregate stats, used
                    # only for the expert_trace plot/text trace).
                    "expert_id": top1_seq[idx],
                    # top-cache_size expert *set* per action token (same
                    # ordering as expert_id) -- feeds run_lengths/switch_rate.
                    "expert_set": [topc_seq[t] for t in act_idx],
                }
                per_seq[layer].append(seq_metrics)
                for k in METRICS:
                    flat[layer][k].append(seq_metrics[k])
        print(f"[analyze] {min(i + batch_size, len(seqs))}/{len(seqs)}",
              flush=True)

    return {
        layer: ((prob_sum[layer] / n_tok[layer]).cpu(),
               {k: torch.cat(v) for k, v in flat[layer].items()},
               per_seq[layer])
        for layer in layers
    }


def agg_stats(x: torch.Tensor):
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def case_study_points(seq_metrics: dict):
    """4 equally spaced token positions (first, 1/4, 3/4, last) for one
    sequence's per-token METRICS tensors. Returns dict metric -> [4 floats]."""
    n = len(seq_metrics[METRICS[0]])
    if n == 0:
        return None
    last = n - 1
    positions = [0, round(last * 0.25), round(last * 0.75), last]
    return {k: [float(seq_metrics[k][p]) for p in positions] for k in METRICS}


def run_lengths(expert_sets):
    """Contiguous run lengths of identical consecutive top-cache_size active-
    expert *sets* -- e.g. [{1,2},{1,2},{1,2},{3,4}] -> [3,1]. A run continues
    as long as the same k=cache_size expert set stays "hot" turn to turn,
    even if which one is individually top-1 wobbles inside that set --
    comparing bare top-1 ids over-counts switches when two already-warm
    experts merely trade rank #1. Directly operationalizes "does the policy
    hold the same option over contiguous tokens" (temporal consolidation)."""
    sets = list(expert_sets)
    if not sets:
        return []
    runs, cur = [], 1
    for a, b in zip(sets, sets[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


@torch.no_grad()
def compute_perplexity(model, eval_ids, pad_id, device, batch_size=16):
    """Standard teacher-forced next-token perplexity on held-out ground-truth
    sequences (independent of the cache/routing analysis -- no per-token
    routing metrics here, just NLL)."""
    import math
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(eval_ids), batch_size):
        chunk = eval_ids[i:i + batch_size]
        L = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[j, :len(x)] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        lp = F.log_softmax(logits[:, :-1].float(), -1) \
            .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].bool()
        total_nll += -lp[m].sum().item()
        total_tok += int(m.sum())
    return math.exp(total_nll / max(total_tok, 1))


def expert_trace_line(pieces, expert_ids):
    """Join cleaned token pieces into one line, inserting a bare '|' at every
    position where the top-1 expert differs from the previous token -- the
    text equivalent of a color change in expert_trace.png."""
    if not pieces:
        return ""
    parts = [pieces[0]]
    for i in range(1, len(pieces)):
        if expert_ids[i] != expert_ids[i - 1]:
            parts.append("|")
        parts.append(pieces[i])
    return "".join(parts)


def write_expert_trace_text(path, layer, case_idx, token_pieces, per_seq_metrics):
    """--out-dir/expert_trace/sequences.txt: the decoded text of every
    case-study sequence (base and tuned), plus the same text with '|'
    inserted at each top-1-expert change -- so the consolidation visible in
    expert_trace.png can be read as text, token by token."""
    lines = [
        f"# Expert trace as text, layer {layer}.",
        "# '|' marks a token where the top-1 expert differs from the "
        "previous token (text equivalent of a color change in "
        "expert_trace.png's strips).",
        "# Token pieces are raw subword units (BPE/SentencePiece leading-"
        "space markers converted to literal spaces); inserted '|' characters "
        "are not part of the original text.",
        "",
    ]
    for variant in ("base", "tuned"):
        variant_pieces = token_pieces.get(variant, [])
        for si in case_idx:
            pieces = variant_pieces[si] if si < len(variant_pieces) else None
            if not pieces:
                continue
            expert_ids = per_seq_metrics[variant][si]["expert_id"].tolist()
            n = min(len(pieces), len(expert_ids))
            lines.append(f"=== seq {si} ({variant}) ===")
            lines.append("[text] " + "".join(pieces[:n]))
            lines.append("[trace] " + expert_trace_line(pieces[:n], expert_ids[:n]))
            lines.append("")
    path.write_text("\n".join(lines))


def finalize_layer(args, out, layer, checkpoint, probs, flat_metrics,
                   per_seq_metrics, n_seqs, ppl=None, token_pieces=None):
    """Aggregate stats + case study + metrics.json + all plots for ONE
    layer, given both variants' (probs, flat, per_seq) at that layer.
    Writes into `out` (the top-level --out-dir for the trained cache layer,
    or an out/other_layer_N subfolder for the side-effect-on-other-layers
    check). Shared by the single-process (--variant both) and the merge
    (--variant merge) code paths, and by every layer analyzed."""
    results = {}
    for variant in ("base", "tuned"):
        p = probs[variant]
        flat = flat_metrics[variant]
        dist_ent = float(-(p * p.clamp_min(1e-12).log()).sum())
        topc = float(p.sort(descending=True).values[:args.cache_size].sum())
        aggregate = {m: agg_stats(flat[m]) for m in METRICS}

        all_runs = [r for ps in per_seq_metrics[variant]
                   for r in run_lengths(ps["expert_set"])]
        run_len_stats = agg_stats(torch.tensor(all_runs, dtype=torch.float32)) \
            if all_runs else None

        # Switch rate (Henderson et al. 2026, Sec. 2.2, adapted): fraction of
        # adjacent action-token pairs where the top-cache_size active-expert
        # *set* changes, averaged per sequence then across sequences (their
        # definition additionally averages over layers L; here that's the
        # caller's job, since finalize_layer is invoked once per analyzed
        # layer). Set equality (not bare top-1 argmax equality) so two
        # already-warm experts trading rank #1 doesn't count as a switch.
        seq_switch_rates = []
        for ps in per_seq_metrics[variant]:
            sets = ps["expert_set"]
            if len(sets) > 1:
                n_switch = sum(1 for a, b in zip(sets, sets[1:]) if a != b)
                seq_switch_rates.append(n_switch / (len(sets) - 1))
        switch_rate_stats = agg_stats(
            torch.tensor(seq_switch_rates, dtype=torch.float32)) \
            if seq_switch_rates else None

        results[variant] = {
            "aggregate": aggregate,
            "mean_dist_entropy_nats": dist_ent,
            "mean_dist_entropy_frac_of_uniform": dist_ent / torch.log(
                torch.tensor(float(len(p)))).item(),
            f"top{args.cache_size}_mass": topc,
            "expert_probs": p.tolist(),
            "expert_run_length": run_len_stats,
            "switch_rate": switch_rate_stats,
        }
        if ppl is not None and variant in ppl:
            results[variant]["perplexity"] = ppl[variant]
        ppl_str = f" | ppl {ppl[variant]:.3f}" if ppl and variant in ppl else ""
        rl_str = f"{run_len_stats['mean']:.2f}" if run_len_stats else "n/a"
        sr_str = f"{switch_rate_stats['mean']:.3f}" if switch_rate_stats else "n/a"
        print(f"[{variant}] hit_ratio {aggregate['hit_ratio']['mean']:.4f} "
              f"(std {aggregate['hit_ratio']['std']:.4f}) | "
              f"variance {aggregate['variance']['mean']:.2e} | "
              f"kl_uniform {aggregate['kl_uniform']['mean']:.3f} nats | "
              f"skew {aggregate['skew']['mean']:.3f} | "
              f"dist entropy {dist_ent:.3f} nats "
              f"({dist_ent / 2.7726:.1%} of uniform) | "
              f"top-{args.cache_size} mass {topc:.3f} | "
              f"mean expert run length {rl_str} tokens | "
              f"switch rate {sr_str}{ppl_str}")

    # case study: same random sequences (by index) for both variants
    n_case = min(args.n_case_study_seqs, len(per_seq_metrics["base"]),
                 len(per_seq_metrics["tuned"]))
    case_idx = random.Random(args.seed).sample(
        range(min(len(per_seq_metrics["base"]), len(per_seq_metrics["tuned"]))),
        n_case)
    case_study = {}
    for variant in ("base", "tuned"):
        case_study[variant] = [case_study_points(per_seq_metrics[variant][idx])
                               for idx in case_idx]
        results[variant]["case_study"] = case_study[variant]

    (out / "aggregate").mkdir(parents=True, exist_ok=True)
    (out / "case_study").mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump({
            "checkpoint": checkpoint,
            "layer": layer,
            "cache_size": args.cache_size,
            "touch_topk": args.touch_topk,
            "case_study_seq_indices": case_idx,
            **results,
        }, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    E = len(probs["base"])
    x = np.arange(E)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, probs["base"].numpy(), 0.4, label="base", color="#888")
    ax.bar(x + 0.2, probs["tuned"].numpy(), 0.4, label="tuned", color="#d62728")
    ax.axhline(1 / E, ls="--", c="k", lw=0.8, label="uniform")
    ax.set_xlabel(f"expert id (layer {layer})")
    ax.set_ylabel("expected routing probability")
    ax.set_title(f"Per-expert routing mass, {n_seqs} held-out seqs")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "expert_probs.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for variant, color in (("base", "#888"), ("tuned", "#d62728")):
        p = probs[variant].sort(descending=True).values.numpy()
        a1.plot(np.arange(1, E + 1), np.cumsum(p), "o-", color=color,
                label=f"{variant} (H={results[variant]['mean_dist_entropy_nats']:.2f})")
        seq_hit_means = torch.tensor(
            [ps["hit_ratio"].mean() if len(ps["hit_ratio"]) else 0.0
             for ps in per_seq_metrics[variant]])
        a2.hist(seq_hit_means.numpy(), bins=30, alpha=0.6, color=color,
                label=f"{variant} ({results[variant]['aggregate']['hit_ratio']['mean']:.3f})")
    a1.plot(np.arange(1, E + 1), np.arange(1, E + 1) / E, "k--", lw=0.8,
            label="uniform")
    a1.axvline(args.cache_size, c="b", ls=":", lw=0.8)
    a1.set_xlabel("experts (sorted by mass)")
    a1.set_ylabel("cumulative routing mass")
    a1.set_title("Routing skew (Lorenz)")
    a1.legend()
    a2.set_xlabel(f"per-seq mean hit ratio (C={args.cache_size})")
    a2.set_ylabel("# sequences")
    a2.set_title("Soft LRU hit ratio (per-sequence mean)")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out / "skew.png", dpi=150)
    plt.close(fig)

    # --- aggregate/*.png: mean/median/min/max/std, base vs tuned ---------
    cats = ["mean", "median", "min", "max", "std"]
    xg = np.arange(len(cats))
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for offset, (variant, color) in enumerate(
                (("base", "#888"), ("tuned", "#d62728"))):
            vals = [results[variant]["aggregate"][metric][c] for c in cats]
            ax.bar(xg + (offset - 0.5) * 0.35, vals, 0.35, label=variant,
                   color=color)
        ax.set_xticks(xg)
        ax.set_xticklabels(cats)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{metric}: aggregate over all tokens & sequences "
                     f"(layer {layer})")
        ax.legend()
        ax.axhline(0, c="k", lw=0.6)
        fig.tight_layout()
        fig.savefig(out / "aggregate" / f"{metric}.png", dpi=150)
        plt.close(fig)

    # --- case_study/*.png: a stack of small per-sequence histograms, one
    # per random held-out sequence, each showing that sequence's metric
    # value at the 4 token positions (base vs tuned grouped bars) ---------
    pos_labels = ["first", "1/4", "3/4", "last"]
    xp = np.arange(len(pos_labels))
    for metric in METRICS:
        fig, axes = plt.subplots(n_case, 1, figsize=(6, 1.6 * n_case),
                                 sharex=True)
        if n_case == 1:
            axes = [axes]
        for row, (axp, si) in enumerate(zip(axes, case_idx)):
            for offset, (variant, color) in enumerate(
                    (("base", "#888"), ("tuned", "#d62728"))):
                pts = case_study[variant][row]
                vals = pts[metric] if pts is not None else [0.0] * len(pos_labels)
                axp.bar(xp + (offset - 0.5) * 0.35, vals, 0.35,
                       label=variant if row == 0 else None, color=color)
            axp.set_ylabel(f"seq {si}", rotation=0, ha="right", va="center",
                          fontsize=8)
            axp.tick_params(axis="y", labelsize=7)
        axes[-1].set_xticks(xp)
        axes[-1].set_xticklabels(pos_labels)
        axes[-1].set_xlabel("token position in sequence")
        axes[0].legend(fontsize=8, loc="upper right", ncol=2)
        fig.suptitle(f"{metric}: {n_case} random held-out sequences "
                     f"(layer {layer})")
        fig.tight_layout()
        fig.savefig(out / "case_study" / f"{metric}.png", dpi=150)
        plt.close(fig)

    # --- expert_trace/*.png: top-1 expert per token as a color strip, one
    # pair of rows (base, tuned) per random held-out sequence -- contiguous
    # same-color runs = the policy holding the same expert over consecutive
    # tokens (temporal consolidation), in the spirit of the routing-timeline
    # plots in temporally-extended-MoE work (e.g. Henderson et al. 2026). ---
    (out / "expert_trace").mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20", E)
    fig, axes = plt.subplots(2 * n_case, 1, figsize=(9, 0.5 * 2 * n_case))
    for row, si in enumerate(case_idx):
        for k, variant in enumerate(("base", "tuned")):
            ax = axes[2 * row + k]
            eids = per_seq_metrics[variant][si]["expert_id"]
            if len(eids) == 0:
                ax.axis("off")
                continue
            ax.imshow(eids.numpy().reshape(1, -1), aspect="auto", cmap=cmap,
                     vmin=0, vmax=max(E - 1, 1), interpolation="nearest")
            ax.set_yticks([])
            ax.set_ylabel(f"seq {si}\n{variant}", rotation=0, ha="right",
                         va="center", fontsize=7)
            ax.set_xticks([])
    axes[-1].set_xlabel("token position (generated span)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=E - 1))
    fig.colorbar(sm, ax=list(axes), orientation="vertical", fraction=0.03,
                pad=0.02, label="top-1 expert id")
    fig.suptitle(f"Top-1 expert over generated tokens, {n_case} random held-out sequences\n"
                f"(layer {layer}) -- contiguous same-color runs = held expert",
                fontsize=10, x=0.45)
    fig.savefig(out / "expert_trace" / "expert_trace.png", dpi=150,
               bbox_inches="tight")
    plt.close(fig)

    wrote_txt = ""
    if token_pieces is not None:
        write_expert_trace_text(out / "expert_trace" / "sequences.txt", layer,
                                case_idx, token_pieces, per_seq_metrics)
        wrote_txt = ", expert_trace/sequences.txt"

    print(f"[eval] wrote {out}/metrics.json, expert_probs.png, skew.png, "
          f"aggregate/*.png, case_study/*.png, expert_trace/expert_trace.png"
          f"{wrote_txt} (layer {layer})")
    return results


def other_analysis_layers(cache_layer, num_layers, n_each_side=2, spacing=None):
    """4 equally spaced layers around cache_layer: n_each_side before and
    n_each_side after, spaced `spacing` apart (default: num_layers // 8, so
    e.g. cache_layer=16/32 -> spacing=4 -> layers [8, 12, 20, 24]). Used to
    check for side effects of the cache reward on nearby layers that were
    never patched/trained -- their native (sparse) routing is analyzed as-is."""
    if spacing is None:
        spacing = max(1, num_layers // 8)
    offsets = [-spacing * k for k in range(n_each_side, 0, -1)] + \
             [spacing * k for k in range(1, n_each_side + 1)]
    layers = sorted({cache_layer + o for o in offsets
                     if 0 <= cache_layer + o < num_layers
                     and cache_layer + o != cache_layer})
    return layers


def finalize(args, out, checkpoint, probs, flat_metrics, per_seq_metrics, n_seqs,
            ppl=None, token_pieces=None):
    """Top-level orchestrator: finalize_layer() for the trained cache layer
    (into --out-dir, as before) plus each of the 4 other analysis layers
    (into --out-dir/other_layer_<N>/), then the (layer-independent)
    perplexity plot once at the top level. probs/flat_metrics/per_seq_metrics
    are dicts keyed by layer -> {variant: ...} (see main()); token_pieces is
    layer-independent (dict variant -> list of per-sequence token-string
    lists), used to write the text-form expert trace."""
    cache_layer = args.cache_layer
    finalize_layer(args, out, cache_layer, checkpoint, probs[cache_layer],
                   flat_metrics[cache_layer], per_seq_metrics[cache_layer],
                   n_seqs, ppl=ppl, token_pieces=token_pieces)

    for layer in probs:
        if layer == cache_layer:
            continue
        layer_out = out / f"other_layer_{layer}"
        layer_out.mkdir(parents=True, exist_ok=True)
        finalize_layer(args, layer_out, layer, checkpoint, probs[layer],
                       flat_metrics[layer], per_seq_metrics[layer], n_seqs,
                       token_pieces=token_pieces,
                       ppl=None)

    if ppl is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        variants_ = ["base", "tuned"]
        vals = [ppl[v] for v in variants_]
        ax.bar(variants_, vals, color=["#888", "#d62728"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_ylabel("held-out perplexity")
        ax.set_title("Teacher-forced perplexity (ground-truth completions)")
        fig.tight_layout()
        fig.savefig(out / "perplexity.png", dpi=150)
        plt.close(fig)
        print(f"[eval] wrote {out}/perplexity.png")

    other = sorted(l for l in probs if l != cache_layer)
    if other:
        print(f"[eval] other-layer side-effect check written to "
              f"{out}/other_layer_{{{','.join(map(str, other))}}}/")
=======
"""Cache-hit-rate extractor for the MoE cache-consolidation variants.

STUB: this loads a checkpoint, runs held-out sequences through it with
`output_router_logits=True`, and reports the empirical LRU cache-hit rate
at `--cache-layer` (reusing the exact same `cache_emulation_rewards`
simulation used as the training-time reward in finetune_moe_grpo.py). It
does NOT yet implement the full routing-distribution analysis described in
handoff/07-eval-setup.md:

  TODO: working-set concentration plots (how routing mass distributes over
        the LRU's cached experts vs. evicted ones, over time)
  TODO: base vs tuned side-by-side comparison (like run_eval_soft_cache.sh's
        two-GPU base/tuned split) -- today this script evaluates ONE
        variant per invocation, no automatic comparison
  TODO: hold/switch segment analysis for temporal_moe (boundary rate,
        segment length distribution -- TemporalWrapMixin tracks the pieces
        for this, e.g. _last_F, but nothing here reads them yet)
  TODO: any actual plotting (matplotlib figures) -- this only writes raw
        numbers to a JSON summary

Usage:
    python scripts/eval_soft_cache.py --variant cache_sft --out-dir evals/soft_cache
    python scripts/eval_soft_cache.py --variant temporal_moe \
        --checkpoint-dir checkpoints/temporal_moe_tamia --out-dir evals/soft_cache
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_lm_harness import VARIANT_CHECKPOINTS, build_variant_model  # noqa: E402
from finetune_moe_grpo import build_eval_sequences  # noqa: E402
from src.cache_reinforce import cache_emulation_rewards  # noqa: E402


@torch.no_grad()
def compute_cache_hit_rate(model, tokenizer, eval_ids, cache_layer, cache_size,
                           experts_per_token, use_topk, batch_size, device):
    """Runs eval_ids through the model in batches and returns the per-sequence
    and overall LRU cache-hit rate at `cache_layer`.

    Simplification vs. training-time RewardEngine._compute: the reward there
    only scores completion tokens (action_mask excludes the prompt); here we
    score every non-pad token in the sequence (action_mask == valid_mask),
    since these are held-out full conversations, not prompt/completion pairs.
    Temporal-routing checkpoints (temporal_moe) are NOT specially handled --
    this reports raw router-logits cache-hit rate, not the wrapper's actual
    held decisions (see TODO above)."""
    pad_id = tokenizer.pad_token_id
    per_seq_hit_rates = []

    for i in range(0, len(eval_ids), batch_size):
        chunk = eval_ids[i:i + batch_size]
        B = len(chunk)
        S = max(len(s) for s in chunk)
        full_ids = torch.full((B, S), pad_id, dtype=torch.long)
        valid = torch.zeros(B, S, dtype=torch.bool)
        for j, ids in enumerate(chunk):
            full_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            valid[j, :len(ids)] = True
        full_ids, valid = full_ids.to(device), valid.to(device)

        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        router_logits = out.router_logits[cache_layer].view(B, S, -1)
        r_cache_tok, _, hit_rate = cache_emulation_rewards(
            router_logits, valid, valid, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        # r_cache_tok sums to the per-sequence hit fraction on action
        # positions (all valid tokens here) -- see cache_emulation_rewards.
        per_seq_hit_rates.extend(r_cache_tok.sum(-1).cpu().tolist())

    overall = sum(per_seq_hit_rates) / len(per_seq_hit_rates) if per_seq_hit_rates else 0.0
    return per_seq_hit_rates, overall
>>>>>>> 20b5cfc90ca6031a0e330c4c345249e979700d72


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
<<<<<<< HEAD
    ap.add_argument("--checkpoint", default=None,
                    help="adapter checkpoint dir; default = latest in the "
                         "c4_softall save dir")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--n-seqs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512,
                    help="held-out prompt truncation length (tokens)")
    ap.add_argument("--gen-len", type=int, default=512,
                    help="tokens sampled per prompt (T=1.0, as in rollouts); "
                         "each variant is analyzed on its own generations")
    ap.add_argument("--seed", type=int, default=43,
                    help="43: same held-out pool as training, different draw")
    ap.add_argument("--cache-layer", type=int, default=-1,
                    help="-1 = middle layer (must match the layer the cache "
                         "reward was trained on -- the baseline is patched "
                         "dense on this exact same layer, see dense_patch_router)")
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--touch-topk", type=int, default=2,
                    help="experts touching the LRU per token (as in training)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-case-study-seqs", type=int, default=8)
    ap.add_argument("--ppl-seqs", type=int, default=256,
                    help="held-out ground-truth sequences for the teacher-"
                         "forced perplexity check (0 disables it)")
    ap.add_argument("--other-layers-each-side", type=int, default=2,
                    help="also analyze this many equally spaced layers "
                         "before AND after --cache-layer (0 disables), to "
                         "check for side effects of the cache reward on "
                         "layers that were never patched/trained -- written "
                         "to --out-dir/other_layer_<N>/")
    ap.add_argument("--other-layer-spacing", type=int, default=0,
                    help="gap between the extra analysis layers; 0 = auto "
                         "(num_hidden_layers // 8)")
    ap.add_argument("--temporal", action="store_true",
                    help="checkpoint was trained with --temporal (STE "
                         "boundary hold/switch mixin at --cache-layer): "
                         "wrap the model the same way before loading the "
                         "adapter, and skip dense_patch_router (dense "
                         "routing is incompatible with the hold/switch "
                         "mixin) -- the cache layer's metrics are read from "
                         "the wrapper's held (forward-filled) full-E logits "
                         "instead of out.router_logits")
    ap.add_argument("--out-dir", default="eval_softcache")
    ap.add_argument("--variant", choices=["both", "base", "tuned", "merge"],
                    default="both",
                    help="both: run base then tuned sequentially in this "
                         "process (single GPU). base/tuned: run only that "
                         "variant and write a partial state file to "
                         "--out-dir (for a 2-process/2-GPU job, see "
                         "run_eval_soft_cache.sh). merge: load both partial "
                         "files from --out-dir and write metrics.json + all "
                         "plots -- no model/GPU needed.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.variant == "merge":
        base_pt = out / "_partial_base.pt"
        tuned_pt = out / "_partial_tuned.pt"
        assert base_pt.exists() and tuned_pt.exists(), (
            f"missing partial state ({base_pt.exists()=}, {tuned_pt.exists()=}); "
            f"run --variant base and --variant tuned first")
        base = torch.load(base_pt, weights_only=False)
        tuned = torch.load(tuned_pt, weights_only=False)
        if args.cache_layer < 0:
            args.cache_layer = base["cache_layer"]
        args.cache_size = base["cache_size"]
        args.touch_topk = base["touch_topk"]
        layers = sorted(set(base["layers"]) & set(tuned["layers"]))
        probs = {l: {"base": base["layers"][l]["probs"],
                    "tuned": tuned["layers"][l]["probs"]} for l in layers}
        flat_metrics = {l: {"base": base["layers"][l]["flat"],
                            "tuned": tuned["layers"][l]["flat"]} for l in layers}
        per_seq_metrics = {l: {"base": base["layers"][l]["per_seq"],
                               "tuned": tuned["layers"][l]["per_seq"]} for l in layers}
        n_seqs = max(base["n_seqs"], tuned["n_seqs"])
        ppl = None
        if "ppl" in base and "ppl" in tuned:
            ppl = {"base": base["ppl"], "tuned": tuned["ppl"]}
        token_pieces = None
        if "token_pieces" in base and "token_pieces" in tuned:
            token_pieces = {"base": base["token_pieces"],
                            "tuned": tuned["token_pieces"]}
        finalize(args, out, base["checkpoint"], probs, flat_metrics,
                 per_seq_metrics, n_seqs, ppl=ppl, token_pieces=token_pieces)
        base_pt.unlink()
        tuned_pt.unlink()
        return

    if args.checkpoint is None:
        from transformers.trainer_utils import get_last_checkpoint
        save_dir = ("checkpoints/grpo_phi-tiny-moe-instruct_cache_mathcode_"
                    "a0.3_b0.08_kds1.0_c4_softall")
        args.checkpoint = get_last_checkpoint(save_dir)
        assert args.checkpoint, f"no checkpoint under {save_dir}"
    print(f"[eval] checkpoint: {args.checkpoint}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

=======
    ap.add_argument("--variant", required=True,
                    choices=list(VARIANT_CHECKPOINTS.keys()) + ["base"])
    ap.add_argument("--checkpoint-dir", default=None,
                    help="override VARIANT_CHECKPOINTS[variant], e.g. for a "
                         "checkpoint trained via cluv on a non-mila cluster")
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--cache-layer", type=int, default=-1)
    ap.add_argument("--cache-experts-per-token", type=int, default=2)
    ap.add_argument("--cache-topk", action="store_true",
                    help="deterministic top-k routing instead of sampling "
                         "(matches real deployment-time routing)")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--num-eval-seqs", type=int, default=256)
    ap.add_argument("--eval-seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="evals/soft_cache")
    args = ap.parse_args()

    from transformers import AutoTokenizer
>>>>>>> 20b5cfc90ca6031a0e330c4c345249e979700d72
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

<<<<<<< HEAD
    num_layers = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    other_layers = other_analysis_layers(
        args.cache_layer, num_layers, args.other_layers_each_side,
        args.other_layer_spacing or None)
    layers = [args.cache_layer] + other_layers
    print(f"[eval] cache layer {args.cache_layer}, C={args.cache_size}, "
          f"LRU touch top-{args.touch_topk}")
    if other_layers:
        print(f"[eval] also analyzing other layers (native routing, "
              f"side-effect check): {other_layers}")

    def build_variant_model(variant):
        """Returns (model, temporal_wrappers_or_None) for one variant.

        base is always a fresh, adapter-free pretrained model -- never the
        checkpoint's LoRA/temporal-mixin weights. For non-temporal
        checkpoints it still gets dense_patch_router applied at the cache
        layer, matching tuned's routing mechanism there so the two are
        comparable at that one layer (requirement (1) below); for
        --temporal it is left fully native (no mixin at all), since the
        mixin is architectural, not something disabling the adapter can
        strip back out.

        tuned is the checkpoint's adapter applied on top of a fresh model,
        temporal-wrapped first if --temporal (architectural, so it has to
        happen before the adapter loads on top of it).
        """
        m = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
        if variant == "base":
            if not args.temporal:
                dense_patch_router(m, args.cache_layer)
            m.eval()
            return m, None
        tw = None
        if args.temporal:
            from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin
            TemporalWrapMixin.apply(m, TemporalWrapConfig(ste=True))
            # every MoE layer gets wrapped (not just --cache-layer), so the
            # other_layers side-effect analysis also reads held logits.
            tw = {l: m._moe_layers[l] for l in layers}
        peft_cfg = LoraConfig.from_pretrained(args.checkpoint)
        peft_model = get_peft_model(m, peft_cfg)
        load_adapter(peft_model, args.checkpoint)
        peft_model.eval()
        if args.temporal:
            # dense routing (full softmax, all experts active) is
            # incompatible with the hold/switch mixin -- see
            # finetune_moe_grpo.py's own ValueError for the combination.
            print(f"[eval] --temporal: skipping dense_patch_router for "
                  f"tuned, reading held routing from the boundary/"
                  f"hold-switch mixin")
        else:
            # patch the shared base module: dense routing for tuned AND
            # base forwards -- requirement (1): baseline analyzed at the
            # exact same cache layer.
            dense_patch_router(peft_model, args.cache_layer)
        return peft_model, tw

    prompts = build_eval_prompts(tok, args.dataset, args.dataset_split,
                                 args.n_seqs, args.max_len, args.seed)
    print(f"[eval] on-policy: {len(prompts)} prompts "
          f"(mean len {sum(map(len, prompts)) / len(prompts):.0f}) "
          f"+ {args.gen_len} sampled tokens each")

    ppl_eval_ids = None
    if args.ppl_seqs > 0:
        ppl_eval_ids = build_eval_sequences(
            tok, args.dataset, args.dataset_split, args.ppl_seqs,
            args.max_len + args.gen_len, args.seed)
        print(f"[eval] perplexity pool: {len(ppl_eval_ids)} held-out "
              f"ground-truth sequences")

    variants = ("base", "tuned") if args.variant == "both" else (args.variant,)
    probs = {l: {} for l in layers}
    flat_metrics = {l: {} for l in layers}
    per_seq_metrics = {l: {} for l in layers}
    ppl, token_pieces, n_seqs = {}, {}, 0
    for variant in variants:
        variant_model, temporal_wrappers = build_variant_model(variant)
        with torch.no_grad():
            torch.manual_seed(args.seed)
            v_seqs, v_prompt_lens = generate_seqs(
                variant_model, prompts, tok, args.gen_len,
                args.batch_size, device)
            glen = sum(len(s) - p for s, p in zip(v_seqs, v_prompt_lens))
            print(f"[{variant}] mean generated len {glen / len(v_seqs):.0f}")
            # decoded (BPE-marker-cleaned) token pieces of the generated span
            # only, for the text-form expert trace (see write_expert_trace_text)
            token_pieces[variant] = [
                [p.replace("Ġ", " ").replace("▁", " ")
                 for p in tok.convert_ids_to_tokens(s[pl:])]
                for s, pl in zip(v_seqs, v_prompt_lens)
            ]
            result = analyze(
                variant_model, v_seqs, v_prompt_lens, tok.pad_token_id,
                layers, args.cache_size, args.touch_topk,
                args.batch_size, device, temporal_wrappers=temporal_wrappers)
            if ppl_eval_ids is not None:
                ppl[variant] = compute_perplexity(
                    variant_model, ppl_eval_ids, tok.pad_token_id, device,
                    args.batch_size)
                print(f"[{variant}] perplexity {ppl[variant]:.3f}")
        del variant_model
        torch.cuda.empty_cache()
        for l, (p, flat, per_seq) in result.items():
            probs[l][variant] = p
            flat_metrics[l][variant] = flat
            per_seq_metrics[l][variant] = per_seq
        n_seqs = len(v_seqs)

        if args.variant in ("base", "tuned"):
            partial_path = out / f"_partial_{variant}.pt"
            torch.save({
                "checkpoint": args.checkpoint, "cache_layer": args.cache_layer,
                "cache_size": args.cache_size, "touch_topk": args.touch_topk,
                "layers": {l: {"probs": p, "flat": flat, "per_seq": per_seq}
                          for l, (p, flat, per_seq) in result.items()},
                "n_seqs": n_seqs, "token_pieces": token_pieces[variant],
                **({"ppl": ppl[variant]} if variant in ppl else {}),
            }, partial_path)
            print(f"[eval] wrote partial state for variant={variant} to "
                  f"{partial_path}; run --variant merge once both partials "
                  f"exist to produce metrics.json + plots")

    if args.variant == "both":
        finalize(args, out, args.checkpoint, probs, flat_metrics,
                 per_seq_metrics, n_seqs, ppl=ppl or None,
                 token_pieces=token_pieces or None)
=======
    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()
    if model.config.num_hidden_layers is not None:
        cache_layer = args.cache_layer if args.cache_layer >= 0 else \
            model.config.num_hidden_layers + args.cache_layer
    else:
        cache_layer = args.cache_layer

    eval_ids = build_eval_sequences(tok, args.dataset, args.dataset_split,
                                    args.num_eval_seqs, args.eval_seq_len, args.seed)
    print(f"[eval_soft_cache] variant={args.variant} cache_layer={cache_layer} "
         f"cache_size={args.cache_size} n_eval_seqs={len(eval_ids)}", flush=True)

    per_seq, overall = compute_cache_hit_rate(
        model, tok, eval_ids, cache_layer, args.cache_size,
        args.cache_experts_per_token, args.cache_topk, args.batch_size, device)
    print(f"[eval_soft_cache] overall cache-hit rate: {overall:.4f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_soft_cache_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": args.variant,
            "checkpoint_dir": args.checkpoint_dir or VARIANT_CHECKPOINTS.get(args.variant),
            "cache_layer": cache_layer,
            "cache_size": args.cache_size,
            "cache_experts_per_token": args.cache_experts_per_token,
            "cache_topk": args.cache_topk,
            "num_eval_seqs": len(eval_ids),
            "overall_hit_rate": overall,
            "per_seq_hit_rate": per_seq,
        }, f, indent=2)
    print(f"[eval_soft_cache] wrote {out_path}", flush=True)
>>>>>>> 20b5cfc90ca6031a0e330c4c345249e979700d72


if __name__ == "__main__":
    main()
