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

STUB: does NOT yet implement the full routing-distribution analysis
described in handoff/07-eval-setup.md:

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
  TODO: temporal_moe's effective held decisions (not raw router_logits) --
        see finetune_moe_grpo.py's RewardEngine._compute temporal_wrappers
        branch; not read here yet, same limitation as before

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
from finetune_moe_grpo import build_eval_prompts  # noqa: E402
from src.cache_reinforce import cache_emulation_rewards  # noqa: E402


@torch.no_grad()
def compute_cache_hit_rate(model, tokenizer, prompt_ids, cache_layer, cache_size,
                           experts_per_token, use_topk, gen_len, batch_size, device):
    """Generates a completion for each of prompt_ids (on-policy, T=1.0) and
    returns the per-sequence and overall LRU cache-hit rate at `cache_layer`,
    scored only on the generated (completion) tokens -- mirrors
    finetune_moe_grpo.py's RewardEngine._compute exactly (prompt tokens warm
    the cache but earn no reward)."""
    pad_id = tokenizer.pad_token_id
    per_seq_hit_rates = []

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

        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        router_logits = out.router_logits[cache_layer].view(B, S, -1)
        r_cache_tok, _, hit_rate = cache_emulation_rewards(
            router_logits, valid, action, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        # r_cache_tok sums to the per-sequence hit fraction on action
        # (completion) positions -- see cache_emulation_rewards.
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
    ap.add_argument("--prompt-len", type=int, default=1024,
                    help="prompt truncation length (tokens)")
    ap.add_argument("--gen-len", type=int, default=1024,
                    help="tokens generated per prompt (T=1.0, matching "
                         "GRPOConfig's training-time sampling) -- only "
                         "these tokens are scored for cache-hit rate")
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

    prompt_ids = build_eval_prompts(tok, args.dataset, args.dataset_split,
                                    args.num_eval_seqs, args.prompt_len, args.seed)
    print(f"[eval_soft_cache] variant={args.variant} cache_layer={cache_layer} "
         f"cache_size={args.cache_size} n_eval_seqs={len(prompt_ids)} "
         f"gen_len={args.gen_len} (on-policy, T=1.0)", flush=True)

    per_seq, overall = compute_cache_hit_rate(
        model, tok, prompt_ids, cache_layer, args.cache_size,
        args.cache_experts_per_token, args.cache_topk, args.gen_len,
        args.batch_size, device)
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
            "on_policy": True,
            "gen_len": args.gen_len,
            "num_eval_seqs": len(prompt_ids),
            "overall_hit_rate": overall,
            "per_seq_hit_rate": per_seq,
        }, f, indent=2)
    print(f"[eval_soft_cache] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
