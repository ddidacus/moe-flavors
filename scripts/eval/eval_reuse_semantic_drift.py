"""Quick eval: does more expert reuse (fewer top-1 expert switches per
sequence) come at the cost of semantic drift from the base model, or does
routing change while the underlying next-token predictions stay close to
base's own?

For each held-out prompt: generate ONE continuation with the BASE model
(greedy, shared across every variant so all comparisons score the exact
same text), then teacher-force the full (prompt, continuation) sequence
through both BASE and VARIANT with output_router_logits=True, and compute,
over the generated (completion) positions only, at --cache-layer:

  1. Expert-reuse switch rate: fraction of adjacent completion tokens
     whose top-1 expert differs (lower = more reuse/contiguous chunking).
     Computed for both base's own routing (switch_rate_base) and the
     variant's routing on the identical text (switch_rate_variant); a
     negative delta_switch_rate = variant} means MORE reuse than base.
  2. Semantic drift: perplexity of the shared continuation under base
     (ppl_base) and under the variant (ppl_variant), plus the mean
     per-token KL(base || variant) over the same completion positions,
     a smoother, distribution-level drift measure that does not depend
     on which token was sampled.

Writes evals/<model>/<date>/eval_reuse_semantic_drift_<variant>.json with
per-prompt arrays and aggregates, plus the per-prompt Pearson correlation
between delta_switch_rate and each drift measure (does more reuse predict
more drift, across held-out prompts, for this one checkpoint).

Usage:
    python scripts/eval/eval_reuse_semantic_drift.py \\
        --model microsoft/Phi-tiny-MoE-instruct --variant base
    python scripts/eval/eval_reuse_semantic_drift.py \\
        --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \\
        --checkpoint-dir ddidacus/olmoe-cache-reward
"""
import argparse
import datetime
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root

import torch

from eval_benchmarks import build_variant_model  # noqa: E402
from eval_cache_conditioning import sample_eval_prompt_texts  # noqa: E402
from eval_complete import (  # noqa: E402
    num_experts_of, resolve_layers, _left_pad_batch, _append_completion,
)


@torch.no_grad()
def generate_base_continuations(base_model, tokenizer, prompt_ids_chunk, gen_len, device):
    """Greedy (deterministic) generation from the base model only, shared
    across every variant so all comparisons score identical text."""
    pad_id = tokenizer.pad_token_id
    prompt_batch, prompt_mask = _left_pad_batch(prompt_ids_chunk, pad_id, device)
    P = prompt_batch.shape[1]
    gen = base_model.generate(
        input_ids=prompt_batch, attention_mask=prompt_mask,
        do_sample=False, max_new_tokens=gen_len, pad_token_id=pad_id,
    )
    completion_ids = gen[:, P:]
    full_ids, valid, action = _append_completion(
        prompt_batch, prompt_mask, completion_ids, pad_id, device)
    return full_ids, valid, action


@torch.no_grad()
def score_sequence(model, full_ids, valid, action, cache_layer):
    """One forward pass: returns (switch_rate per seq, mean_nll per seq,
    log_probs at completion-predicting positions) for one model on one
    batch of already-built (full_ids, valid, action).

    Only ever holds ONE (B, S, vocab) float32 tensor (log_probs) at a time;
    everything else the model returns is freed as soon as it's consumed, to
    keep peak memory bounded for large-vocab models like OLMoE."""
    out = model(input_ids=full_ids, attention_mask=valid.long(),
               output_router_logits=True, use_cache=False)
    B, S = full_ids.shape

    # --- expert-reuse switch rate over completion (action) tokens ---
    router_logits = out.router_logits[cache_layer].view(B, S, -1).float()
    top1 = router_logits.argmax(-1)  # (B, S)
    switch_rates = []
    for b in range(B):
        idx = action[b].nonzero(as_tuple=True)[0]
        seq = top1[b, idx]
        if len(seq) > 1:
            switches = (seq[1:] != seq[:-1]).float().mean().item()
        else:
            switches = 0.0
        switch_rates.append(switches)
    del router_logits, top1

    # --- teacher-forced NLL / log-probs at completion-predicting positions ---
    logits = out.logits[:, :-1, :].float()          # predicts full_ids[:, 1:]
    del out
    targets = full_ids[:, 1:]
    target_mask = action[:, 1:]                      # only completion targets
    log_probs = torch.log_softmax(logits, dim=-1)
    del logits
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, S-1)

    nlls = []
    for b in range(B):
        m = target_mask[b]
        if m.sum() > 0:
            nlls.append((-token_logp[b][m]).mean().item())
        else:
            nlls.append(float("nan"))
    del token_logp

    return switch_rates, nlls, log_probs, target_mask


@torch.no_grad()
def kl_from_base(base_log_probs, variant_log_probs, target_mask):
    """Mean per-token KL(base || variant) over target_mask positions, per
    sequence in the batch. Full-vocabulary KL, no top-k truncation.

    Computed one sequence at a time so we never materialize a second full
    (B, S, vocab) intermediate tensor on top of the two inputs."""
    B = target_mask.shape[0]
    kls = []
    for b in range(B):
        m = target_mask[b]
        if m.sum() > 0:
            blp = base_log_probs[b][m]
            vlp = variant_log_probs[b][m]
            kl = (blp.exp() * (blp - vlp)).sum(-1).mean().item()
            kls.append(kl)
        else:
            kls.append(float("nan"))
    return kls


def pearson_r(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]  # drop NaNs
    if len(pairs) < 3:
        return None
    xs2, ys2 = zip(*pairs)
    n = len(xs2)
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs2, ys2))
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--cache-layer", type=int, default=-1)
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--out-dir-root", default="evals")
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(args.model)
    num_experts_of(config)  # sanity check the model family is recognized
    cache_layer, _ = resolve_layers(config.num_hidden_layers, args.cache_layer)

    out_dir = Path(args.out_dir_root) / args.model.split("/")[-1] / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_reuse_semantic_drift_{args.variant}.json"
    print(f"[reuse_drift] model={args.model} variant={args.variant} "
         f"cache_layer={cache_layer} out={out_path}", flush=True)

    base_model = build_variant_model("base", args.model, device, None)
    base_model.eval()
    if args.variant == "base":
        variant_model = base_model
    else:
        variant_model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
        variant_model.eval()

    prompt_texts = sample_eval_prompt_texts(tok, args.dataset, args.dataset_split,
                                            args.num_prompts, args.seed)
    prompt_ids = [tok(t, truncation=True, max_length=args.prompt_len,
                      add_special_tokens=False)["input_ids"] for t in prompt_texts]

    all_switch_base, all_switch_variant = [], []
    all_ppl_base, all_ppl_variant = [], []
    all_kl = []

    for i in range(0, len(prompt_ids), args.batch_size):
        chunk = prompt_ids[i:i + args.batch_size]
        full_ids, valid, action = generate_base_continuations(
            base_model, tok, chunk, args.gen_len, device)

        switch_base, nll_base, logp_base, target_mask = score_sequence(
            base_model, full_ids, valid, action, cache_layer)
        switch_variant, nll_variant, logp_variant, _ = score_sequence(
            variant_model, full_ids, valid, action, cache_layer)
        kls = kl_from_base(logp_base, logp_variant, target_mask)

        all_switch_base.extend(switch_base)
        all_switch_variant.extend(switch_variant)
        all_ppl_base.extend([torch.tensor(n).exp().item() if n == n else float("nan") for n in nll_base])
        all_ppl_variant.extend([torch.tensor(n).exp().item() if n == n else float("nan") for n in nll_variant])
        all_kl.extend(kls)

        print(f"[reuse_drift] {min(i + args.batch_size, len(prompt_ids))}/{len(prompt_ids)}", flush=True)

    def clean(vals):
        return [v for v in vals if v == v]

    delta_switch = [v - b for v, b in zip(all_switch_variant, all_switch_base)]
    delta_ppl = [v - b for v, b in zip(all_ppl_variant, all_ppl_base) if v == v and b == b]

    def agg(vals):
        vals = clean(vals)
        return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}

    result = {
        "model": args.model, "variant": args.variant, "cache_layer": cache_layer,
        "n_prompts": len(prompt_ids), "gen_len": args.gen_len,
        "switch_rate_base": {**agg(all_switch_base), "per_seq": all_switch_base},
        "switch_rate_variant": {**agg(all_switch_variant), "per_seq": all_switch_variant},
        "delta_switch_rate": {**agg(delta_switch), "per_seq": delta_switch},
        "ppl_base": {**agg(all_ppl_base), "per_seq": all_ppl_base},
        "ppl_variant": {**agg(all_ppl_variant), "per_seq": all_ppl_variant},
        "delta_ppl": {**agg(delta_ppl), "per_seq": delta_ppl},
        "kl_from_base": {**agg(all_kl), "per_seq": all_kl},
        "corr_delta_switch_vs_delta_ppl": pearson_r(delta_switch, [v - b for v, b in zip(all_ppl_variant, all_ppl_base)]),
        "corr_delta_switch_vs_kl": pearson_r(delta_switch, all_kl),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[reuse_drift] done. switch_rate base={agg(all_switch_base)['mean']:.4f} "
         f"variant={agg(all_switch_variant)['mean']:.4f} "
         f"delta_ppl={agg(delta_ppl)['mean']:.4f} "
         f"kl={agg(all_kl)['mean']:.6f} "
         f"corr(dswitch,dppl)={result['corr_delta_switch_vs_delta_ppl']} "
         f"corr(dswitch,kl)={result['corr_delta_switch_vs_kl']} -> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
