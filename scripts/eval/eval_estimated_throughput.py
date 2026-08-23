"""Estimated offloaded-inference latency: prefill compute time plus the
disk/RAM transfer cost of the cache misses that prefill would incur under
an LRU expert cache, matching the standard MoE-offloading latency model
(compute overlapped or serial with expert loads -- see Eliseev & Mazur
2023, "Fast Inference of MoE via Offloading"):

    time_total = time_fwd + misses * offload_time_per_expert

Two independent measurements feed that formula:

1. offload_time_per_expert (the constant): wall-clock time to load ONE
   expert's weight tensors from the model's on-disk safetensors checkpoint
   into a fresh CPU tensor, repeated 1024 times (round-robining over every
   expert at --cache-layer so the benchmark isn't just re-reading the same
   bytes), reported as mean +/- std in milliseconds. This is a disk/page-
   cache read + memcpy cost, not a training/inference measurement -- it
   does not touch the loaded nn.Module at all, only the raw checkpoint
   file(s), matching how a real offloading engine would stream an expert's
   weights in. NOTE: after the first pass over all experts the OS page
   cache is warm, so this measures steady-state (cached-checkpoint) reload
   latency, not cold-disk latency -- the realistic case for a serving
   process that keeps the checkpoint file mmapped/warm.

2. time_fwd and misses (per prompt): for 256 random held-out prompts (same
   pool/sampling as eval_router.py/eval_cache_conditioning.py), a single
   teacher-forced forward pass over the prompt (no generation -- this is
   the prefill compute cost), timed with CUDA events, plus the LRU
   cache-miss count that same forward pass incurs (src.cache_reinforce.
   cache_emulation_rewards, action mask = the whole prompt, since prefill
   is exactly what's being processed here -- unlike eval_router.py's
   convention of scoring generated tokens only).

Usage:
    python scripts/eval/eval_estimated_throughput.py \\
        --variant base --out-dir evals/estimated_throughput
    python scripts/eval/eval_estimated_throughput.py \\
        --variant cache_sft --cache-size 4 --out-dir evals/estimated_throughput
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root

import torch

from eval_benchmarks import build_variant_model  # noqa: E402
from finetune_moe_grpo import build_eval_prompts  # noqa: E402
from src.cache_reinforce import cache_emulation_rewards  # noqa: E402


# ---------------------------------------------------------------------------
# Part 1: per-expert disk-load constant.
# ---------------------------------------------------------------------------

# Matches both Phi-tiny-MoE's on-disk naming (model.layers.N.block_sparse_moe.
# experts.I.w1/w2/w3.weight) and OLMoE's (model.layers.N.mlp.experts.I.
# gate_proj/up_proj/down_proj.weight) -- both store one safetensors entry per
# (layer, expert, projection), never a fused per-layer tensor, on disk.
_EXPERT_TENSOR_RE = re.compile(r"layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts?\.(\d+)\.")


def find_expert_tensor_groups(model_name, layer_idx):
    """Returns {expert_idx: [(shard_filename, tensor_name), ...]} for every
    per-expert tensor at `layer_idx` in `model_name`'s on-disk checkpoint,
    read from the safetensors shard index (no full model download)."""
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    groups = {}
    for name, shard in weight_map.items():
        m = _EXPERT_TENSOR_RE.search(name)
        if m and int(m.group(1)) == layer_idx:
            groups.setdefault(int(m.group(2)), []).append((shard, name))
    if not groups:
        raise ValueError(f"no per-expert tensors found at layer {layer_idx} "
                         f"in {model_name}'s checkpoint index")
    return groups


def benchmark_expert_load_ms(model_name, layer_idx, num_trials=1024):
    """Loads one expert's weight tensors from the on-disk checkpoint into a
    fresh CPU tensor, `num_trials` times (round-robining over every expert
    found at `layer_idx`), returning the per-trial times in milliseconds."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    groups = find_expert_tensor_groups(model_name, layer_idx)
    expert_ids = sorted(groups.keys())

    # resolve + open every shard this layer's experts live in once, up front
    # (a real offloading engine keeps the checkpoint file(s) open/mmapped
    # for the life of the serving process -- open() itself is not part of
    # the per-expert transfer cost being measured).
    shard_names = sorted({shard for tensors in groups.values() for shard, _ in tensors})
    shard_paths = {s: hf_hub_download(model_name, s) for s in shard_names}
    handles = {s: safe_open(p, framework="pt", device="cpu") for s, p in shard_paths.items()}

    times_ms = []
    for i in range(num_trials):
        expert_id = expert_ids[i % len(expert_ids)]
        t0 = time.perf_counter()
        for shard, tensor_name in groups[expert_id]:
            handles[shard].get_tensor(tensor_name)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


# ---------------------------------------------------------------------------
# Part 2: per-prompt prefill latency + cache misses.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_prefill_latency_and_misses(model, tokenizer, prompt_ids, cache_layer,
                                       cache_size, experts_per_token, use_topk,
                                       batch_size, device):
    """For each prompt, one teacher-forced forward pass (no generation --
    this is the prefill cost), timed with CUDA events, plus the LRU
    cache-miss count that pass incurs across the WHOLE prompt (unlike
    eval_router.py's generated-tokens-only convention -- here the prompt
    itself is exactly what's being computed/timed)."""
    pad_id = tokenizer.pad_token_id
    per_seq_fwd_ms = []
    per_seq_misses = []

    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        B = len(chunk)
        P = max(len(p) for p in chunk)
        input_ids = torch.full((B, P), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, P), dtype=torch.long)
        for j, p in enumerate(chunk):
            input_ids[j, :len(p)] = torch.tensor(p, dtype=torch.long)
            attn_mask[j, :len(p)] = 1
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_router_logits=True, use_cache=False)
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)
        per_seq_fwd_ms.extend([elapsed_ms / B] * B)

        valid = attn_mask.bool()
        router_logits = out.router_logits[cache_layer].view(B, P, -1)
        r_cache_tok, _, _ = cache_emulation_rewards(
            router_logits, valid, valid, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        hit_fracs = r_cache_tok.sum(-1).cpu()
        T = valid.sum(-1).float().cpu().clamp(min=1)
        misses = (1.0 - hit_fracs) * T * experts_per_token
        per_seq_misses.extend(misses.tolist())

    return per_seq_fwd_ms, per_seq_misses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", default="base",
                    help="build_variant_model only special-cases 'base' (no "
                         "adapter) and 'temporal_moe' (mixin wrap); any other "
                         "string takes the generic LoRA-adapter load path, so "
                         "this need not be one of eval_benchmarks.ALL_VARIANTS "
                         "as long as --checkpoint-dir is given")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="required unless --variant base; overrides "
                         "VARIANT_CHECKPOINTS[variant] when variant is a "
                         "registered name")
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--cache-layer", type=int, default=-1)
    ap.add_argument("--cache-experts-per-token", type=int, default=2)
    ap.add_argument("--cache-topk", action="store_true",
                    help="deterministic top-k routing instead of sampling")
    ap.add_argument("--num-expert-load-trials", type=int, default=1024)
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--num-eval-prompts", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="default: evals/<model_name>/eval_estimated_throughput/ "
                         "(one results_<variant>.json per variant)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoConfig
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    num_hidden_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    cache_layer = args.cache_layer if args.cache_layer >= 0 else num_hidden_layers // 2

    print(f"[eval_estimated_throughput] benchmarking per-expert disk-load time "
         f"at layer {cache_layer} ({args.num_expert_load_trials} trials) ...", flush=True)
    load_times_ms = benchmark_expert_load_ms(args.model, cache_layer, args.num_expert_load_trials)
    load_mean = statistics.fmean(load_times_ms)
    load_std = statistics.pstdev(load_times_ms)
    print(f"[eval_estimated_throughput] expert load time: {load_mean:.4f} +/- "
         f"{load_std:.4f} ms/expert", flush=True)

    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()

    prompt_ids = build_eval_prompts(tok, args.dataset, args.dataset_split,
                                    args.num_eval_prompts, args.prompt_len, args.seed)
    print(f"[eval_estimated_throughput] variant={args.variant} cache_layer={cache_layer} "
         f"cache_size={args.cache_size} n_prompts={len(prompt_ids)}", flush=True)

    fwd_ms, misses = compute_prefill_latency_and_misses(
        model, tok, prompt_ids, cache_layer, args.cache_size,
        args.cache_experts_per_token, args.cache_topk, args.batch_size, device)

    total_ms = [f + m * load_mean for f, m in zip(fwd_ms, misses)]

    fwd_mean, fwd_std = statistics.fmean(fwd_ms), statistics.pstdev(fwd_ms)
    miss_mean, miss_std = statistics.fmean(misses), statistics.pstdev(misses)
    total_mean, total_std = statistics.fmean(total_ms), statistics.pstdev(total_ms)

    print(f"[eval_estimated_throughput] fwd pass: {fwd_mean:.2f} +/- {fwd_std:.2f} ms/prompt", flush=True)
    print(f"[eval_estimated_throughput] cache misses: {miss_mean:.2f} +/- {miss_std:.2f} /prompt", flush=True)
    print(f"[eval_estimated_throughput] estimated total: {total_mean:.2f} +/- "
         f"{total_std:.2f} ms/prompt", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("evals") / args.model.split("/")[-1] / "eval_estimated_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "variant": args.variant,
            "checkpoint_dir": args.checkpoint_dir, "cache_layer": cache_layer,
            "cache_size": args.cache_size,
            "cache_experts_per_token": args.cache_experts_per_token,
            "cache_topk": args.cache_topk, "n_prompts": len(prompt_ids),
            "expert_load_ms": {"mean": load_mean, "std": load_std,
                              "n_trials": args.num_expert_load_trials,
                              "per_trial": load_times_ms},
            "fwd_ms": {"mean": fwd_mean, "std": fwd_std, "per_seq": fwd_ms},
            "cache_misses": {"mean": miss_mean, "std": miss_std, "per_seq": misses},
            "estimated_total_ms": {"mean": total_mean, "std": total_std, "per_seq": total_ms},
        }, f, indent=2)
    print(f"[eval_estimated_throughput] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
