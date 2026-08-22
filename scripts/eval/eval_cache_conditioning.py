"""Eval: does the prompt-conditioned cache-size adapter (scripts/train/
train_prompt_conditioned.py) actually change routing behavior when given a
different "[CACHE_SIZE=X]" prefix, vs. the untouched base model?

For a FIXED set of held-out prompts (sampled once, reused across every
cache size and both models for an apples-to-apples comparison), each
prompt is prepended with a "[CACHE_SIZE=X]" prefix for X in --cache-sizes,
generated on-policy (temperature=1.0, deterministic top-k routing), and
scored at the same cache layer used in training:

  * hit rate: LRU cache-hit fraction on the generated tokens (src.
    cache_reinforce.cache_emulation_rewards, same as finetune_moe_grpo.py's
    RewardEngine._compute / eval_router.py's compute_cache_hit_rate).
  * unique experts: count of distinct experts drawn over the generated
    tokens (NOT provided by cache_emulation_rewards -- its LRUExpertCache
    is capacity-limited to the current working set, not a full-history
    set -- computed here from the per-token expert draws it returns).

Both metrics are aggregated (mean/std) per (model, cache_size) and plotted
as a grouped bar chart (hit rate) and a grouped box plot (unique experts),
base vs. fine-tuned within each cache-size group.

Usage:
    python scripts/eval/eval_cache_conditioning.py \\
        --checkpoint-dir checkpoints/prompt_conditioned_test_mila \\
        --out-dir evals/cache_conditioning
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.cache_reinforce import cache_emulation_rewards

# peft 0.19 x transformers 5.8 bug: on adapter-checkpoint load, peft's v4->v5
# key conversion calls WeightConverter with a removed kwarg
# ('distributed_operation') and crashes. Same workaround as finetune_moe_
# grpo.py / train_prompt_conditioned.py / eval_benchmarks.py -- duplicated
# here rather than imported from eval_benchmarks.py, which pulls in the
# full lm-eval-harness dependency chain (+ HF_ALLOW_CODE_EVAL env setup)
# just for this ~10-line patch.
import peft.utils.transformers_weight_conversion as _pwc

_orig_convert = _pwc.convert_peft_adapter_state_dict_for_transformers

def _convert_only_if_legacy(model, peft_config, adapter_state_dict, adapter_name):
    legacy = any(".w1." in k or ".w2." in k or ".w3." in k
                 or "block_sparse_moe" in k for k in adapter_state_dict)
    if not legacy:
        return adapter_state_dict
    return _orig_convert(model=model, peft_config=peft_config,
                         adapter_state_dict=adapter_state_dict,
                         adapter_name=adapter_name)

_pwc.convert_peft_adapter_state_dict_for_transformers = _convert_only_if_legacy


CACHE_PREFIX_TEMPLATE = "[CACHE_SIZE={size}]"
EVAL_POOL_PER_SPLIT = 1000  # matches train_prompt_conditioned.py's reserved eval pool


def load_adapter(peft_model, ckpt_dir):
    """Manual adapter load (peft 0.19 can't load ParamWrapper/target_parameters
    adapters via its own state-dict path) -- identical to eval_benchmarks.py/
    eval_router.py/train_prompt_conditioned.py's loader."""
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
    print(f"[eval_cache_conditioning] loaded {len(remapped)} adapter tensors from {ckpt_dir}")


def build_model(base_model_name, device, checkpoint_dir=None):
    """checkpoint_dir=None -> plain base model. Otherwise -> base model +
    LoRA adapter loaded from checkpoint_dir (resolved to its latest
    checkpoint-NNN subdir if it's a run directory)."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    if checkpoint_dir is None:
        return m

    from peft import LoraConfig, get_peft_model
    from transformers.trainer_utils import get_last_checkpoint
    ckpt_dir = get_last_checkpoint(checkpoint_dir) or checkpoint_dir
    assert Path(ckpt_dir, "adapter_config.json").exists(), \
        f"no adapter checkpoint found under {checkpoint_dir}"

    peft_cfg = LoraConfig.from_pretrained(ckpt_dir)
    peft_model = get_peft_model(m, peft_cfg)
    load_adapter(peft_model, ckpt_dir)
    return peft_model


def sample_eval_prompt_texts(tokenizer, dataset_name, split, n_total, seed,
                             pool_per_split=EVAL_POOL_PER_SPLIT):
    """N held-out prompts (chat template + add_generation_prompt=True, no
    reference answer), rendered to TEXT (not yet cache-size-prefixed or
    tokenized) -- sampled ONCE from the same reserved eval pool convention
    as train_prompt_conditioned.py's build_eval_prompts_conditioned, so the
    caller can reuse the identical prompt set across every cache size and
    model instead of drawing a different subset per size."""
    import random
    from src.nemotron_data import load_split_stream

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max(n_total // len(splits), 1)
    rng = random.Random(seed)
    use_chat = tokenizer.chat_template is not None
    texts = []
    for sp in splits:
        ds = load_split_stream(dataset_name, sp)
        pool = []
        for row in ds:
            msgs = [m for m in row["messages"] if m["content"].strip()]
            if any(m["role"] == "assistant" for m in msgs):
                pool.append(msgs)
            if len(pool) >= pool_per_split:
                break
        for msgs in rng.sample(pool, min(per_split, len(pool))):
            user_msgs = []
            for m in msgs:
                if m["role"] == "assistant":
                    break
                user_msgs.append(m)
            if use_chat:
                text = tokenizer.apply_chat_template(
                    user_msgs, add_generation_prompt=True, tokenize=False)
            else:
                text = "\n".join(m["content"] for m in user_msgs)
            texts.append(text)
    return texts[:n_total]


def _find_layer_module(model, layer_idx):
    """Same as eval_router.py -- works whether model is bare or PEFT-wrapped."""
    suffix = f"layers.{layer_idx}"
    matches = [(name, mod) for name, mod in model.named_modules() if name.endswith(suffix)]
    if not matches:
        raise ValueError(f"no submodule ending in '{suffix}' found in model")
    matches.sort(key=lambda nm: len(nm[0]))
    return matches[0][1]


def _sequence_stack_distances(experts_row, valid_row, action_row):
    """Mattson et al. (1970) LRU stack-distance algorithm for one sequence.

    experts_row: (S, K) expert ids drawn at every position (already CPU
    ints/lists). valid_row/action_row: (S,) bools -- prompt tokens (valid,
    not action) WARM the recency stack but aren't recorded, matching the
    hit-rate/unique-experts convention elsewhere in this script.

    Walks positions in order maintaining `stack`, a most-recent-first list
    of distinct expert ids ever seen. For each expert access at a VALID
    position: if the expert is already in the stack, its 1-indexed depth
    (before it's moved to the front) is its stack distance -- an LRU cache
    of capacity C would have held it iff distance <= C, for ANY C, so this
    one pass implies the hit-rate at every cache size at once. If the
    expert has never been seen before in this sequence, its distance is
    None (a "compulsory miss" -- no finite cache could have held it, since
    it wasn't accessed yet). Only accesses at ACTION (completion)
    positions are returned; the stack itself is still updated (warmed) by
    prompt-position accesses that precede them.

    The recency stack itself is capped in size only by the number of
    distinct experts ever seen (<= num_experts, e.g. 16), so list.index/
    remove here cost at most O(num_experts) per access -- trivial for the
    sequence lengths and expert counts involved.
    """
    stack = []
    distances = []
    for s in range(len(valid_row)):
        if not valid_row[s]:
            continue
        for e in experts_row[s]:
            e = int(e)
            if e in stack:
                d = stack.index(e) + 1
                stack.remove(e)
            else:
                d = None
            stack.insert(0, e)
            if action_row[s]:
                distances.append(d)
    return distances


@torch.no_grad()
def compute_cache_metrics(model, tokenizer, prompt_ids, cache_layer, cache_size,
                          experts_per_token, use_topk, gen_len, batch_size, device):
    """Generates a completion for each of prompt_ids (on-policy, T=1.0),
    scored at cache_layer -- returns, per sequence: the LRU cache-hit
    fraction (at the fixed `cache_size` passed in) and the count of
    DISTINCT experts drawn, both restricted to the generated (completion)
    tokens only (same masking as finetune_moe_grpo.py's RewardEngine.
    _compute / eval_router.py's compute_cache_hit_rate -- prompt tokens
    warm the cache but aren't scored); plus, per sequence, the mean LRU
    stack distance and compulsory-miss rate over the same completion
    tokens -- a CACHE-SIZE-INDEPENDENT locality summary (Mattson et al.
    1970's stack algorithm): an access to expert e has stack distance d if
    e was the d-th most-recently-used distinct expert (1-indexed) at that
    point, or is a "compulsory miss" (no finite cache could have held it)
    if e has never been accessed before in this sequence. An access is an
    LRU-hit at capacity C iff its stack distance <= C, so this single pass
    implies the hit-rate at EVERY cache size at once, unlike cache_size
    above which only tells you the hit rate at one fixed capacity."""
    pad_id = tokenizer.pad_token_id
    per_seq_hit_rates = []
    per_seq_unique_experts = []
    per_seq_mean_stack_distance = []
    per_seq_compulsory_miss_rate = []

    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        B = len(chunk)
        P = max(len(p) for p in chunk)
        prompt_batch = torch.full((B, P), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((B, P), dtype=torch.long)
        for j, p in enumerate(chunk):
            prompt_batch[j, P - len(p):] = torch.tensor(p, dtype=torch.long)
            prompt_mask[j, P - len(p):] = 1
        prompt_batch, prompt_mask = prompt_batch.to(device), prompt_mask.to(device)

        gen = model.generate(
            input_ids=prompt_batch, attention_mask=prompt_mask,
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=gen_len, pad_token_id=pad_id,
        )
        completion_ids = gen[:, P:]

        S = P + completion_ids.shape[1]
        full_ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros((B, S), dtype=torch.bool, device=device)
        action = torch.zeros((B, S), dtype=torch.bool, device=device)
        full_ids[:, :P] = prompt_batch
        valid[:, :P] = prompt_mask.bool()
        full_ids[:, P:] = completion_ids
        comp_valid = completion_ids != pad_id
        comp_len = comp_valid.float().flip(-1).cumsum(-1).flip(-1).bool() | comp_valid
        valid[:, P:] = comp_len if comp_len.any() else comp_valid
        action[:, P:] = valid[:, P:]

        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        router_logits = out.router_logits[cache_layer].view(B, S, -1)
        r_cache_tok, experts, _ = cache_emulation_rewards(
            router_logits, valid, action, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        hit_fracs = r_cache_tok.sum(-1).cpu().tolist()
        per_seq_hit_rates.extend(hit_fracs)

        experts_cpu = experts.cpu()
        valid_cpu = valid.cpu()
        action_cpu = action.cpu()
        for b in range(B):
            drawn = experts_cpu[b][action_cpu[b]].flatten().tolist()
            per_seq_unique_experts.append(len(set(drawn)))

            distances = _sequence_stack_distances(
                experts_cpu[b].tolist(), valid_cpu[b].tolist(), action_cpu[b].tolist())
            finite = [d for d in distances if d is not None]
            per_seq_mean_stack_distance.append(
                statistics.fmean(finite) if finite else float("nan"))
            per_seq_compulsory_miss_rate.append(
                (len(distances) - len(finite)) / len(distances) if distances else float("nan"))

    return (per_seq_hit_rates, per_seq_unique_experts,
           per_seq_mean_stack_distance, per_seq_compulsory_miss_rate)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results, cache_sizes, variant_order, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_variants = len(variant_order)
    colors = {v: plt.cm.tab10.colors[i] for i, v in enumerate(variant_order)}
    display_labels = {"base": "base", "fine_tuned": "fine-tuned",
                      "fine_tuned_no_prefix": "fine-tuned (no prefix)"}
    labels = {v: display_labels.get(v, v) for v in variant_order}

    # --- grouped bar: hit rate ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    n_groups = len(cache_sizes)
    width = 0.8 / n_variants
    x = list(range(n_groups))
    for j, model_label in enumerate(variant_order):
        means = [results[model_label][cs]["hit_rate_mean"] for cs in cache_sizes]
        stds = [results[model_label][cs]["hit_rate_std"] for cs in cache_sizes]
        offset = (j - (n_variants - 1) / 2) * width
        ax.bar([xi + offset for xi in x], means, width, yerr=stds, capsize=4,
              label=labels[model_label], color=colors[model_label])
    ax.set_xticks(x)
    ax.set_xticklabels([f"cache={cs}" for cs in cache_sizes])
    ax.set_ylabel("cache hit rate")
    ax.set_title("Cache hit rate by conditioned cache size")
    ax.legend(loc="best")
    fig.tight_layout()
    hit_rate_path = plots_dir / "hit_rate_bar.png"
    fig.savefig(hit_rate_path, dpi=150)
    plt.close(fig)
    print(f"[eval_cache_conditioning] wrote {hit_rate_path}")

    # --- grouped box: unique experts ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    group_width = n_variants + 1  # n_variants boxes + 1 gap per cache-size group
    for i, cs in enumerate(cache_sizes):
        for j, model_label in enumerate(variant_order):
            data = results[model_label][cs]["per_seq_unique_experts"]
            pos = i * group_width + j + 1
            bp = ax.boxplot([data], positions=[pos], widths=0.7, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(colors[model_label])
    ax.set_xticks([i * group_width + (n_variants + 1) / 2 for i in range(n_groups)])
    ax.set_xticklabels([f"cache={cs}" for cs in cache_sizes])
    ax.set_ylabel("unique experts used per prompt")
    ax.set_title("Unique experts activated by conditioned cache size")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[v]) for v in variant_order]
    ax.legend(handles, [labels[v] for v in variant_order], loc="best")
    fig.tight_layout()
    experts_path = plots_dir / "unique_experts_box.png"
    fig.savefig(experts_path, dpi=150)
    plt.close(fig)
    print(f"[eval_cache_conditioning] wrote {experts_path}")

    # --- grouped box: stack distance (cache-size-INDEPENDENT locality) ------
    # Mattson stack distance: how many distinct experts were touched since
    # an expert was last used, at the moment it's reused. Doesn't reference
    # any assumed cache capacity (unlike hit_rate/unique_experts above,
    # which are both computed at the fixed `cache_size` this group
    # represents) -- lower is more cache-friendly at ANY capacity.
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cs in enumerate(cache_sizes):
        for j, model_label in enumerate(variant_order):
            data = [d for d in results[model_label][cs]["per_seq_stack_distance"] if d == d]
            pos = i * group_width + j + 1
            bp = ax.boxplot([data], positions=[pos], widths=0.7, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(colors[model_label])
    ax.set_xticks([i * group_width + (n_variants + 1) / 2 for i in range(n_groups)])
    ax.set_xticklabels([f"cache={cs}" for cs in cache_sizes])
    ax.set_ylabel("mean LRU stack distance per prompt (lower = more cache-friendly)")
    ax.set_title("Stack distance (cache-size-independent locality) by conditioned cache size")
    ax.legend(handles, [labels[v] for v in variant_order], loc="best")
    fig.tight_layout()
    stack_path = plots_dir / "stack_distance_box.png"
    fig.savefig(stack_path, dpi=150)
    plt.close(fig)
    print(f"[eval_cache_conditioning] wrote {stack_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--checkpoint-dir", default="checkpoints/prompt_conditioned_test_mila")
    ap.add_argument("--cache-sizes", type=str, default="2,4,8")
    ap.add_argument("--cache-layer", type=int, default=-1, help="-1 = auto (middle layer)")
    ap.add_argument("--cache-experts-per-token", type=int, default=2)
    ap.add_argument("--cache-topk", action="store_true", default=True,
                    help="deterministic top-k routing (matches real "
                         "deployment-time routing); on by default for a "
                         "reproducible fixed-prompt-set comparison")
    ap.add_argument("--no-cache-topk", dest="cache_topk", action="store_false")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--num-eval-prompts", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="evals/cache_conditioning")
    args = ap.parse_args()

    cache_sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]

    from transformers import AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_texts = sample_eval_prompt_texts(
        tok, args.dataset, args.dataset_split, args.num_eval_prompts, args.seed)
    print(f"[eval_cache_conditioning] sampled {len(base_texts)} held-out prompts "
         f"(reused across every cache size and both models)", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    def dump_results_json(results, complete):
        # Written after EVERY (model, cache_size) combo, not just at the
        # end -- a run that hits the SLURM time cap mid-way (each combo can
        # take 15-40+ min: 256 prompts x up to gen_len tokens, on-policy)
        # still leaves usable partial results instead of losing the whole
        # run. Atomic write (temp file + os.replace) so a reader never sees
        # a half-written file.
        import os
        payload = {
            "model": args.model,
            "checkpoint_dir": args.checkpoint_dir,
            "cache_layer": cache_layer,
            "cache_experts_per_token": args.cache_experts_per_token,
            "cache_topk": args.cache_topk,
            "cache_sizes": cache_sizes,
            "num_eval_prompts": len(base_texts),
            "gen_len": args.gen_len,
            "seed": args.seed,
            "complete": complete,
            "results": {ml: {str(cs): v for cs, v in per_size.items()}
                       for ml, per_size in results.items()},
        }
        tmp = out_path.with_suffix(f".json.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, out_path)

    # Each entry: (result label, checkpoint dir for build_model, whether to
    # prepend "[CACHE_SIZE=X]" to the prompt). "fine_tuned_no_prefix" reuses
    # the SAME loaded adapter as "fine_tuned" (no second checkpoint load) --
    # it isolates whether the adapter changed the model's routing at all
    # independent of the conditioning signal, vs. needing the prefix to
    # actually see a hit-rate/expert-count difference from base.
    model_loads = [("base", None, [("base", True)]),
                  (args.checkpoint_dir, args.checkpoint_dir,
                   [("fine_tuned", True), ("fine_tuned_no_prefix", False)])]

    results = {}
    cache_layer = None
    for _, checkpoint_dir, prompt_variants in model_loads:
        model = build_model(args.model, device, checkpoint_dir)
        model.eval()
        if cache_layer is None:
            num_layers = model.config.num_hidden_layers
            cache_layer = args.cache_layer if args.cache_layer >= 0 else num_layers // 2
            print(f"[eval_cache_conditioning] cache_layer={cache_layer}/{num_layers}", flush=True)

        for model_label, use_prefix in prompt_variants:
            results[model_label] = {}
            for cache_size in cache_sizes:
                if use_prefix:
                    prefix = CACHE_PREFIX_TEMPLATE.format(size=cache_size)
                    texts = [f"{prefix} {t}" for t in base_texts]
                else:
                    texts = base_texts
                prompt_ids = [
                    tok(t, truncation=True, max_length=args.prompt_len,
                       add_special_tokens=False)["input_ids"]
                    for t in texts
                ]
                (hit_rates, unique_experts, mean_stack_distances,
                 compulsory_miss_rates) = compute_cache_metrics(
                    model, tok, prompt_ids, cache_layer, cache_size,
                    args.cache_experts_per_token, args.cache_topk, args.gen_len,
                    args.batch_size, device)

                def _agg(vals):
                    finite = [v for v in vals if v == v]  # drop NaN
                    mean = statistics.fmean(finite) if finite else float("nan")
                    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
                    return mean, std

                hit_mean, hit_std = _agg(hit_rates)
                ue_mean, ue_std = _agg(unique_experts)
                sd_mean, sd_std = _agg(mean_stack_distances)
                cm_mean, cm_std = _agg(compulsory_miss_rates)
                results[model_label][cache_size] = {
                    "hit_rate_mean": hit_mean, "hit_rate_std": hit_std,
                    "unique_experts_mean": ue_mean, "unique_experts_std": ue_std,
                    "stack_distance_mean": sd_mean, "stack_distance_std": sd_std,
                    "compulsory_miss_rate_mean": cm_mean, "compulsory_miss_rate_std": cm_std,
                    "per_seq_hit_rate": hit_rates,
                    "per_seq_unique_experts": unique_experts,
                    "per_seq_stack_distance": mean_stack_distances,
                    "per_seq_compulsory_miss_rate": compulsory_miss_rates,
                }
                print(f"[eval_cache_conditioning] {model_label} cache={cache_size}: "
                     f"hit_rate={hit_mean:.4f}+/-{hit_std:.4f} "
                     f"unique_experts={ue_mean:.2f}+/-{ue_std:.2f} "
                     f"stack_distance={sd_mean:.2f}+/-{sd_std:.2f} "
                     f"compulsory_miss_rate={cm_mean:.4f}+/-{cm_std:.4f}", flush=True)
                dump_results_json(results, complete=False)

        del model
        torch.cuda.empty_cache()

    variant_order = [ml for _, _, pv in model_loads for ml, _ in pv]

    dump_results_json(results, complete=True)
    print(f"[eval_cache_conditioning] wrote {out_path}", flush=True)

    header = "cache_size  " + "  ".join(
        f"{v} hit_rate / unique_exp / stack_dist" for v in variant_order)
    print(f"\n{header}")
    for cs in cache_sizes:
        row = [f"{cs:<11}"]
        for v in variant_order:
            r = results[v][cs]
            row.append(f"{r['hit_rate_mean']:.4f}+/-{r['hit_rate_std']:.4f} / "
                      f"{r['unique_experts_mean']:.2f}+/-{r['unique_experts_std']:.2f} / "
                      f"{r['stack_distance_mean']:.2f}+/-{r['stack_distance_std']:.2f}")
        print("  ".join(row))

    make_plots(results, cache_sizes, variant_order, out_dir)


if __name__ == "__main__":
    main()
