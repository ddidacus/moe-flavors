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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
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
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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


if __name__ == "__main__":
    main()
