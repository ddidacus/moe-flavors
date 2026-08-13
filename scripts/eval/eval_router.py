"""Cache-hit-rate extractor for the MoE cache-consolidation variants.

On-policy: generates completions from the model itself (temperature=1.0,
top_p=1.0, top_k=0 -- matching GRPOConfig's training-time sampling exactly)
from held-out prompts, then scores the cache-hit rate only on the
generated tokens (output_router_logits over the full prompt+completion
sequence, masked to completion positions) -- the exact same forward/mask/
cache_emulation_rewards call as finetune_moe_grpo.py's RewardEngine._compute.
Earlier versions of this script were teacher-forced (scored every token of
held-out ground-truth conversations, prompt included) -- that measures a
different distribution than the training reward and gave misleadingly flat
numbers across all variants (~0.26 for both base and cache_sft, despite
cache_sft's training-time reward climbing to ~0.50); see
handoff/05-sweep-results.md's soft-cache off-policy eval finding.

Layer convention: --cache-layer -1 ("auto") resolves to num_hidden_layers
// 2 (the middle layer), matching finetune_moe_grpo.py's training-time
default exactly. An earlier version of this script resolved -1 via
Python-style negative indexing (num_hidden_layers + (-1) = the LAST layer),
silently measuring an untrained router and producing flat, uninformative
hit rates for every variant -- see results/results.tex's soft-cache-metrics
note for the before/after numbers.

STUB: does NOT yet implement the full routing-distribution analysis
described in handoff/07-eval-setup.md:

  TODO: working-set concentration plots (how routing mass distributes over
        the LRU's cached experts vs. evicted ones, over time)
  TODO: base vs tuned side-by-side comparison (like run_router.sh's
        two-GPU base/tuned split) -- today this script evaluates ONE
        variant per invocation, no automatic comparison
  TODO: hold/switch segment analysis for temporal_moe (boundary rate,
        segment length distribution -- TemporalWrapMixin tracks the pieces
        for this, e.g. _last_F, but nothing here reads them yet)
  TODO: temporal_moe's effective held decisions (not raw router_logits) --
        see finetune_moe_grpo.py's RewardEngine._compute temporal_wrappers
        branch; not read here yet. This is the likely reason temporal_moe's
        measured hit rate stays flat near the untrained baseline even at
        the corrected layer: its real routing decisions are the mixin's
        held/switched choices, not the raw per-token router logits this
        script reads.

Usage:
    python scripts/eval/eval_router.py --variant cache_sft --out-dir evals/router
    python scripts/eval/eval_router.py --variant temporal_moe \
        --checkpoint-dir checkpoints/temporal_moe_tamia --out-dir evals/router
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root
from eval_benchmarks import VARIANT_CHECKPOINTS, ALL_VARIANTS, build_variant_model  # noqa: E402
from finetune_moe_grpo import build_eval_prompts  # noqa: E402
from src.cache_reinforce import cache_emulation_rewards  # noqa: E402


# ---------------------------------------------------------------------------
# Layer timing: wall-clock milliseconds spent inside the single cached MoE
# layer's forward, isolated via a forward pre/post hook pair timed with CUDA
# events (GPU ops are async, so plain time.time() around the call would
# mostly measure kernel-launch overhead, not actual compute). Locating the
# layer by name suffix rather than a fixed attribute path keeps this working
# whether `model` is the bare base model (model.model.layers.N) or a
# PEFT-wrapped one (model.base_model.model.model.layers.N).
# ---------------------------------------------------------------------------

def _find_layer_module(model, layer_idx):
    suffix = f"layers.{layer_idx}"
    matches = [(name, mod) for name, mod in model.named_modules() if name.endswith(suffix)]
    if not matches:
        raise ValueError(f"no submodule ending in '{suffix}' found in model")
    matches.sort(key=lambda nm: len(nm[0]))  # shortest matching path = the real layer
    return matches[0][1]


class LayerTimer:
    """Context manager: times one forward pass through `layer` in milliseconds."""

    def __init__(self, layer):
        self.layer = layer
        self._start_evt = torch.cuda.Event(enable_timing=True)
        self._end_evt = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._pre = self.layer.register_forward_pre_hook(
            lambda module, inp: self._start_evt.record())
        self._post = self.layer.register_forward_hook(
            lambda module, inp, out: self._end_evt.record())
        return self

    def __exit__(self, *exc_info):
        self._pre.remove()
        self._post.remove()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self):
        return self._start_evt.elapsed_time(self._end_evt)


# ---------------------------------------------------------------------------
# Core metric: generate on-policy completions for a batch of prompts, then
# replay them through the model with output_router_logits=True to score the
# LRU cache-hit rate, cache-miss count, and per-layer throughput on the
# generated tokens only.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cache_hit_rate(model, tokenizer, prompt_ids, cache_layer, cache_size,
                           experts_per_token, use_topk, gen_len, batch_size, device):
    """Generates a completion for each of prompt_ids (on-policy, T=1.0) and
    returns, per sequence: the LRU cache-hit rate, the raw cache-miss count,
    and the cache layer's forward latency (ms/sample) -- all scored/timed on
    the generated (completion) tokens only, mirroring finetune_moe_grpo.py's
    RewardEngine._compute exactly (prompt tokens warm the cache but earn no
    reward and aren't timed)."""
    pad_id = tokenizer.pad_token_id
    layer_module = _find_layer_module(model, cache_layer)
    per_seq_hit_rates = []
    per_seq_cache_misses = []
    per_sample_throughput_ms = []

    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        B = len(chunk)
        P = max(len(p) for p in chunk)
        prompt_batch = torch.full((B, P), pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((B, P), dtype=torch.long)
        for j, p in enumerate(chunk):
            # left-pad prompts so generation starts at the same column B-wide
            prompt_batch[j, P - len(p):] = torch.tensor(p, dtype=torch.long)
            prompt_mask[j, P - len(p):] = 1
        prompt_batch, prompt_mask = prompt_batch.to(device), prompt_mask.to(device)

        gen = model.generate(
            input_ids=prompt_batch, attention_mask=prompt_mask,
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=gen_len, pad_token_id=pad_id,
        )
        completion_ids = gen[:, P:]  # (B, gen_len), right-padded by generate()

        S = P + completion_ids.shape[1]
        full_ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros((B, S), dtype=torch.bool, device=device)
        action = torch.zeros((B, S), dtype=torch.bool, device=device)
        full_ids[:, :P] = prompt_batch
        valid[:, :P] = prompt_mask.bool()
        full_ids[:, P:] = completion_ids
        comp_valid = completion_ids != pad_id
        # eos may legitimately appear as a real token; only pad tail is invalid
        comp_len = comp_valid.float().flip(-1).cumsum(-1).flip(-1).bool() | comp_valid
        valid[:, P:] = comp_len if comp_len.any() else comp_valid
        action[:, P:] = valid[:, P:]

        with LayerTimer(layer_module) as timer:
            out = model(input_ids=full_ids, attention_mask=valid.long(),
                        output_router_logits=True, use_cache=False)
        # ms/sample: the batch shares one forward pass, so divide the
        # layer's total wall-clock time evenly across its B samples.
        per_sample_throughput_ms.extend([timer.elapsed_ms / B] * B)

        router_logits = out.router_logits[cache_layer].view(B, S, -1)
        r_cache_tok, _, hit_rate = cache_emulation_rewards(
            router_logits, valid, action, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        # r_cache_tok sums to the per-sequence hit fraction on action
        # (completion) positions -- see cache_emulation_rewards. Convert
        # that fraction back into a raw miss count: T completion tokens x
        # experts_per_token accesses each, of which (1 - hit_fraction) missed.
        hit_fracs = r_cache_tok.sum(-1).cpu()
        T = action.sum(-1).cpu().clamp(min=1)
        misses = (1.0 - hit_fracs) * T * experts_per_token
        per_seq_hit_rates.extend(hit_fracs.tolist())
        per_seq_cache_misses.extend(misses.tolist())

    overall = sum(per_seq_hit_rates) / len(per_seq_hit_rates) if per_seq_hit_rates else 0.0
    return per_seq_hit_rates, overall, per_seq_cache_misses, per_sample_throughput_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", required=True, choices=ALL_VARIANTS)
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
    ap.add_argument("--prompt-len", type=int, default=1024,
                    help="prompt truncation length (tokens)")
    ap.add_argument("--gen-len", type=int, default=1024,
                    help="tokens generated per prompt (T=1.0, matching "
                         "GRPOConfig's training-time sampling) -- only "
                         "these tokens are scored for cache-hit rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="evals/router")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()
    if model.config.num_hidden_layers is not None:
        # -1 means "auto": match finetune_moe_grpo.py's training-time default
        # of num_hidden_layers // 2 (the middle layer), NOT Python-style
        # negative indexing (which resolves to the last layer and silently
        # measures an untrained router -- see module docstring).
        cache_layer = args.cache_layer if args.cache_layer >= 0 else \
            model.config.num_hidden_layers // 2
    else:
        cache_layer = args.cache_layer

    prompt_ids = build_eval_prompts(tok, args.dataset, args.dataset_split,
                                    args.num_eval_seqs, args.prompt_len, args.seed)
    print(f"[eval_router] variant={args.variant} cache_layer={cache_layer} "
         f"cache_size={args.cache_size} n_eval_seqs={len(prompt_ids)} "
         f"gen_len={args.gen_len} (on-policy, T=1.0)", flush=True)

    per_seq, overall, per_seq_misses, per_sample_ms = compute_cache_hit_rate(
        model, tok, prompt_ids, cache_layer, args.cache_size,
        args.cache_experts_per_token, args.cache_topk, args.gen_len,
        args.batch_size, device)

    miss_mean = statistics.fmean(per_seq_misses)
    miss_std = statistics.pstdev(per_seq_misses) if len(per_seq_misses) > 1 else 0.0
    throughput_mean_ms = statistics.fmean(per_sample_ms)
    print(f"[eval_router] overall cache-hit rate: {overall:.4f}", flush=True)
    print(f"[eval_router] cache misses/sample: {miss_mean:.2f} +/- {miss_std:.2f}", flush=True)
    print(f"[eval_router] throughput: {throughput_mean_ms:.2f} ms/sample "
         f"through layer {cache_layer}", flush=True)

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
            "on_policy": True,
            "gen_len": args.gen_len,
            "num_eval_seqs": len(prompt_ids),
            "overall_hit_rate": overall,
            "per_seq_hit_rate": per_seq,
            "cache_misses_mean": miss_mean,
            "cache_misses_std": miss_std,
            "per_seq_cache_misses": per_seq_misses,
            "throughput_ms_per_sample": throughput_mean_ms,
            "per_sample_throughput_ms": per_sample_ms,
        }, f, indent=2)
    print(f"[eval_router] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
