"""Monolithic eval: runs one (model, variant) checkpoint through five
analyses in a single process and writes results under
`evals/<model_name>/<date>/eval_<part>.json` (+ one PNG for part 3):

  1. eval_harness.json        -- lm-eval-harness downstream tasks (MMLU,
                                 MMMLU, GSM8K, HumanEval, MATH), sized to
                                 ~1024 total examples (not lm_eval's raw
                                 per-subtask --limit semantics: MMLU's 57
                                 subject subtasks each get limit//57 so the
                                 total stays ~1024; MMMLU uses eval_
                                 benchmarks.build_mmmlu_samples(n_total=
                                 1024) directly; GSM8K/HumanEval/MATH are
                                 flat tasks so limit=1024 is already total).
  2. eval_routing_distribution.json -- per-expert token-assignment
                                 distribution at --cache-layer over 1024
                                 held-out Nemotron prompts (teacher-forced,
                                 no generation): avg tokens/expert,
                                 skewness, and "expert contribution index"
                                 (ECI = normalized routing entropy
                                 H(p)/log(E), 1.0=uniform, ~0=collapsed).
  3. eval_routing_viz.json + expert_trace.png -- for 16 of those prompts,
                                 on-policy generation, top-1 expert id per
                                 generated token as a color strip, one row
                                 per prompt per layer, at --cache-layer and
                                 two "quartile" layers (num_hidden_layers//4
                                 and 3*num_hidden_layers//4) -- generalizes
                                 across Phi-tiny-MoE (32 layers -> [8,16,24])
                                 and OLMoE (16 layers -> [4,8,12]).
  4. eval_throughput.json     -- per-prompt estimated offloaded-inference
                                 latency: standalone prefill forward pass
                                 (timed) + a full on-policy generate() call
                                 (timed) + cache misses across the WHOLE
                                 sequence (prefill+decode, not decode-only)
                                 x the per-expert disk-load constant (see
                                 eval_estimated_throughput.py). Also reports
                                 tokens/second.
  5. eval_cache_hit_ratio.json -- on-policy cache hit rate at --cache-layer/
                                 --cache-size over the same 1024 prompts
                                 (straight reuse of eval_router.py's
                                 compute_cache_hit_rate).

Cache layer resolution matches every training run in this repo (all pass
--cache-layer -1): `num_hidden_layers // 2` -- layer 16 for Phi-tiny-MoE
(32 layers), layer 8 for OLMoE-1B-7B-0125-Instruct (16 layers).

Usage:
    python scripts/eval/eval_complete.py \\
        --model microsoft/Phi-tiny-MoE-instruct --variant base
    python scripts/eval/eval_complete.py \\
        --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \\
        --checkpoint-dir checkpoints/cache_reward_olmoe_tamia \\
        --cache-size 16 --cache-experts-per-token 8 --cache-topk
"""
import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root

import torch

import eval_benchmarks  # noqa: E402
from eval_benchmarks import build_variant_model  # noqa: E402
from eval_router import compute_cache_hit_rate  # noqa: E402
from eval_estimated_throughput import benchmark_expert_load_ms  # noqa: E402
from finetune_moe_grpo import build_eval_prompts  # noqa: E402
from src.cache_reinforce import cache_emulation_rewards  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def num_experts_of(config):
    """Attribute name differs by model family: phimoe uses
    num_local_experts, olmoe uses num_experts."""
    n = getattr(config, "num_local_experts", None)
    if n is None:
        n = getattr(config, "num_experts", None)
    if n is None:
        raise ValueError("could not resolve num_experts from model config")
    return n


def resolve_layers(num_hidden_layers, cache_layer_arg):
    """cache_layer: -1 -> num_hidden_layers // 2 (the trained layer, matches
    every training run in this repo). other_layers: quartile spacing
    (num_hidden_layers//4, 3*num_hidden_layers//4), generalizing across
    very different depths (32 vs 16 layers)."""
    cache_layer = cache_layer_arg if cache_layer_arg >= 0 else num_hidden_layers // 2
    other_layers = [num_hidden_layers // 4, (3 * num_hidden_layers) // 4]
    return cache_layer, other_layers


@torch.no_grad()
def generate_batch(model, tokenizer, prompt_ids_chunk, gen_len, device):
    """Left-pad + on-policy generate (T=1.0) for one batch of prompts,
    returning (full_ids, valid, action, P) -- action is the completion-only
    mask; pass `valid` instead where whole-sequence scoring is wanted.
    Same mechanics as eval_router.py::compute_cache_hit_rate /
    eval_cache_conditioning.py, factored out since this script needs it
    for parts 3 and 4 (part 5 calls compute_cache_hit_rate directly)."""
    pad_id = tokenizer.pad_token_id
    B = len(prompt_ids_chunk)
    P = max(len(p) for p in prompt_ids_chunk)
    prompt_batch = torch.full((B, P), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((B, P), dtype=torch.long)
    for j, p in enumerate(prompt_ids_chunk):
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
    return full_ids, valid, action, P


# ---------------------------------------------------------------------------
# Part 1: lm-eval-harness suite, sized to ~1024 total examples.
# ---------------------------------------------------------------------------

def run_harness(model_name, variant, checkpoint_dir, batch_size, total_budget,
                num_seeds, seed, out_path):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_variant_model(variant, model_name, device, checkpoint_dir)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device)

    n_mmlu_subjects = len(eval_benchmarks.MMLU_SUBJECT_SIZES)
    mmlu_limit = max(1, total_budget // n_mmlu_subjects)
    seeds = [seed + i for i in range(max(1, num_seeds))]

    mmlu_results = lm_eval.simple_evaluate(
        model=lm, tasks=["mmlu"], num_fewshot=None, limit=mmlu_limit,
        log_samples=False, random_seed=seeds[0], numpy_random_seed=seeds[0],
        torch_random_seed=seeds[0], fewshot_random_seed=seeds[0],
    )["results"]

    mmmlu_samples = eval_benchmarks.build_mmmlu_samples(seeds[0], n_total=total_budget)
    mmmlu_leaf_results = lm_eval.simple_evaluate(
        model=lm, tasks=list(mmmlu_samples.keys()), num_fewshot=None,
        samples=mmmlu_samples, log_samples=False,
        random_seed=seeds[0], numpy_random_seed=seeds[0],
        torch_random_seed=seeds[0], fewshot_random_seed=seeds[0],
    )["results"]
    n_sampled = sum(len(v) for v in mmmlu_samples.values())
    mmmlu_acc = sum(mmmlu_leaf_results[t]["acc,none"] * len(mmmlu_samples[t])
                    for t in mmmlu_samples) / n_sampled
    mmmlu_pooled = {"acc,none": mmmlu_acc, "n_sampled": n_sampled,
                    "n_leaf_tasks_touched": len(mmmlu_samples)}

    det_results = {"mmlu": mmlu_results["mmlu"], "mmmlu": mmmlu_pooled}

    per_seed = {}
    for s in seeds:
        seed_results = lm_eval.simple_evaluate(
            model=lm, tasks=["gsm8k", "humaneval"], num_fewshot=None,
            limit=total_budget, gen_kwargs=eval_benchmarks.SAMPLING_KWARGS,
            confirm_run_unsafe_code=True, log_samples=False,
            random_seed=s, numpy_random_seed=s, torch_random_seed=s,
            fewshot_random_seed=s,
        )["results"]
        math_results = lm_eval.simple_evaluate(
            model=lm, tasks=["hendrycks_math"],
            num_fewshot=eval_benchmarks.MATH_NUM_FEWSHOT, limit=total_budget,
            gen_kwargs=eval_benchmarks.SAMPLING_KWARGS, log_samples=False,
            random_seed=s, numpy_random_seed=s, torch_random_seed=s,
            fewshot_random_seed=s,
        )["results"]
        seed_results.update(math_results)
        per_seed[s] = seed_results

    stochastic_agg = eval_benchmarks._aggregate_stochastic(
        per_seed, eval_benchmarks.STOCHASTIC_TASKS)
    results = {**det_results, **stochastic_agg}

    del model
    torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "seeds": seeds,
            "mmlu_limit_per_subtask": mmlu_limit, "mmmlu_n_total": total_budget,
            "flat_task_limit": total_budget,
            "deterministic_results": det_results,
            "per_seed_stochastic_results": per_seed, "results": results,
        }, f, indent=2)
    print(f"[eval_complete][1/5] wrote {out_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Part 2: routing distribution (avg tokens/expert, skewness, ECI).
# ---------------------------------------------------------------------------

def _skewness(counts):
    x = torch.as_tensor(counts, dtype=torch.float64)
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(1e-12)
    return float(((x - mean) / std).pow(3).mean())


def _eci(counts, num_experts):
    x = torch.as_tensor(counts, dtype=torch.float64)
    p = x / x.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    return float(ent / math.log(num_experts))


@torch.no_grad()
def run_routing_distribution(model, tokenizer, prompt_ids, cache_layer,
                             experts_per_token, num_experts, batch_size,
                             device, out_path, model_name, variant):
    pad_id = tokenizer.pad_token_id
    counts = torch.zeros(num_experts, dtype=torch.long)
    n_valid_tokens = 0

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

        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_router_logits=True, use_cache=False)
        router_logits = out.router_logits[cache_layer].view(B, P, -1).float()
        probs = torch.softmax(router_logits, dim=-1)
        top_idx = probs.topk(experts_per_token, dim=-1).indices  # (B,P,k)
        valid = attn_mask.bool()
        picked = top_idx[valid]  # (n_valid_tokens, k)
        counts += torch.bincount(picked.reshape(-1).cpu(), minlength=num_experts)
        n_valid_tokens += int(valid.sum())
        print(f"[eval_complete][2/5] {min(i + batch_size, len(prompt_ids))}/"
             f"{len(prompt_ids)}", flush=True)

    counts_list = counts.tolist()
    avg_tokens_per_expert = statistics.fmean(counts_list)
    skew = _skewness(counts_list)
    eci = _eci(counts_list, num_experts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "cache_layer": cache_layer,
            "num_experts": num_experts, "experts_per_token": experts_per_token,
            "n_prompts": len(prompt_ids), "n_valid_tokens": n_valid_tokens,
            "avg_tokens_per_expert": avg_tokens_per_expert,
            "skewness": skew, "eci": eci, "per_expert_counts": counts_list,
        }, f, indent=2)
    print(f"[eval_complete][2/5] avg_tokens/expert={avg_tokens_per_expert:.1f} "
         f"skew={skew:.3f} eci={eci:.3f} -> wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Part 3: expert-choice visualization (color strip per prompt per layer).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_routing_viz(model, tokenizer, prompt_ids, layers, num_experts,
                    gen_len, device, out_path, png_path, model_name, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_prompts = len(prompt_ids)
    per_prompt_layer_ids = {layer: [None] * n_prompts for layer in layers}

    for idx, p in enumerate(prompt_ids):
        full_ids, valid, action, P = generate_batch(model, tokenizer, [p], gen_len, device)
        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        S = full_ids.shape[1]
        act = action[0]
        for layer in layers:
            router_logits = out.router_logits[layer].view(1, S, -1).float()
            top1 = router_logits[0].argmax(-1)  # (S,)
            per_prompt_layer_ids[layer][idx] = top1[act].cpu()
        print(f"[eval_complete][3/5] {idx + 1}/{n_prompts}", flush=True)

    cmap = plt.get_cmap("tab20", num_experts)
    n_rows = n_prompts * len(layers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 0.35 * n_rows))
    if n_rows == 1:
        axes = [axes]
    row = 0
    for pidx in range(n_prompts):
        for layer in layers:
            ax = axes[row]
            eids = per_prompt_layer_ids[layer][pidx]
            if eids.numel() == 0:
                ax.axis("off")
            else:
                ax.imshow(eids.numpy().reshape(1, -1), aspect="auto", cmap=cmap,
                         vmin=0, vmax=max(num_experts - 1, 1), interpolation="nearest")
            ax.set_yticks([])
            ax.set_ylabel(f"p{pidx}\nL{layer}", rotation=0, ha="right",
                         va="center", fontsize=6)
            ax.set_xticks([])
            row += 1
    axes[-1].set_xlabel("token position (generated span)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_experts - 1))
    fig.colorbar(sm, ax=list(axes), orientation="vertical", fraction=0.03,
                pad=0.02, label="top-1 expert id")
    fig.suptitle(f"Top-1 expert over generated tokens, {n_prompts} prompts x "
                f"layers {layers}\n(model={model_name.split('/')[-1]}, variant={variant})",
                fontsize=9, x=0.45)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "layers": layers,
            "n_prompts": n_prompts, "num_experts": num_experts,
            "png": str(png_path),
            "per_layer_expert_ids": {
                str(layer): [ids.tolist() for ids in per_prompt_layer_ids[layer]]
                for layer in layers
            },
        }, f, indent=2)
    print(f"[eval_complete][3/5] wrote {out_path} and {png_path}", flush=True)


# ---------------------------------------------------------------------------
# Part 4: throughput (prefill + decode + whole-sequence offload misses).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_throughput(model, tokenizer, prompt_ids, cache_layer, cache_size,
                   experts_per_token, use_topk, gen_len, batch_size, device,
                   model_name, num_expert_load_trials, out_path, variant):
    pad_id = tokenizer.pad_token_id
    print(f"[eval_complete][4/5] benchmarking per-expert disk-load time "
         f"({num_expert_load_trials} trials) ...", flush=True)
    load_times_ms = benchmark_expert_load_ms(model_name, cache_layer, num_expert_load_trials)
    load_mean = statistics.fmean(load_times_ms)
    load_std = statistics.pstdev(load_times_ms)

    fwd_ms_list, decode_ms_list, misses_list, tok_per_sec_list = [], [], [], []

    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        B = len(chunk)
        P = max(len(p) for p in chunk)
        input_ids = torch.full((B, P), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, P), dtype=torch.long)
        for j, p in enumerate(chunk):
            # left-pad: required for correct batched model.generate() below
            # (right-padding, fine for a standalone forward pass, breaks
            # causal continuation once real generation is involved).
            input_ids[j, P - len(p):] = torch.tensor(p, dtype=torch.long)
            attn_mask[j, P - len(p):] = 1
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        # standalone prefill timing (teacher-forced forward, no generation)
        start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        end_evt.record()
        torch.cuda.synchronize()
        fwd_ms = start_evt.elapsed_time(end_evt)

        # full on-policy generate call, timed (includes its own internal
        # prefill + every decode step) -- decode_ms derived as the residual
        # over the standalone prefill measurement above. Reuses this same
        # generate() call's output for the miss-counting forward pass below
        # (not a second untimed generate call) so both numbers reflect the
        # same sampled completion.
        gen_start, gen_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        gen_start.record()
        gen = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                             do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=gen_len, pad_token_id=pad_id)
        gen_end.record()
        torch.cuda.synchronize()
        generate_total_ms = gen_start.elapsed_time(gen_end)
        decode_ms = max(generate_total_ms - fwd_ms, 0.0)

        completion_ids = gen[:, P:]
        S = P + completion_ids.shape[1]
        full_ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros((B, S), dtype=torch.bool, device=device)
        action = torch.zeros((B, S), dtype=torch.bool, device=device)
        full_ids[:, :P] = input_ids
        valid[:, :P] = attn_mask.bool()
        full_ids[:, P:] = completion_ids
        comp_valid = completion_ids != pad_id
        comp_len = comp_valid.float().flip(-1).cumsum(-1).flip(-1).bool() | comp_valid
        valid[:, P:] = comp_len if comp_len.any() else comp_valid
        action[:, P:] = valid[:, P:]

        # whole-sequence (prefill + decode) cache misses
        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        S = full_ids.shape[1]
        router_logits = out.router_logits[cache_layer].view(B, S, -1)
        r_cache_tok, _, _ = cache_emulation_rewards(
            router_logits, valid, valid, cache_size=cache_size,
            experts_per_token=experts_per_token, use_topk=use_topk,
        )
        hit_fracs = r_cache_tok.sum(-1).cpu()
        T = valid.sum(-1).float().cpu().clamp(min=1)
        misses = (1.0 - hit_fracs) * T * experts_per_token
        comp_tokens = action.sum(-1).float().cpu().clamp(min=1)

        fwd_ms_per_sample = fwd_ms / B
        decode_ms_per_sample = decode_ms / B
        for b in range(B):
            total_ms = fwd_ms_per_sample + decode_ms_per_sample + float(misses[b]) * load_mean
            fwd_ms_list.append(fwd_ms_per_sample)
            decode_ms_list.append(decode_ms_per_sample)
            misses_list.append(float(misses[b]))
            tok_per_sec_list.append(float(comp_tokens[b]) / (total_ms / 1000.0))
        print(f"[eval_complete][4/5] {min(i + batch_size, len(prompt_ids))}/"
             f"{len(prompt_ids)}", flush=True)

    total_ms_list = [f + d + m * load_mean for f, d, m in
                     zip(fwd_ms_list, decode_ms_list, misses_list)]

    def _agg(vals):
        return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "cache_layer": cache_layer,
            "cache_size": cache_size, "experts_per_token": experts_per_token,
            "n_prompts": len(prompt_ids),
            "expert_load_ms": {"mean": load_mean, "std": load_std,
                              "n_trials": num_expert_load_trials},
            "fwd_ms": {**_agg(fwd_ms_list), "per_seq": fwd_ms_list},
            "decode_ms": {**_agg(decode_ms_list), "per_seq": decode_ms_list},
            "cache_misses": {**_agg(misses_list), "per_seq": misses_list},
            "estimated_total_ms": {**_agg(total_ms_list), "per_seq": total_ms_list},
            "tokens_per_second": {**_agg(tok_per_sec_list), "per_seq": tok_per_sec_list},
        }, f, indent=2)
    print(f"[eval_complete][4/5] total={statistics.fmean(total_ms_list):.1f}ms "
         f"tok/s={statistics.fmean(tok_per_sec_list):.2f} -> wrote {out_path}", flush=True)
    return {"tokens_per_second_mean": statistics.fmean(tok_per_sec_list)}


# ---------------------------------------------------------------------------
# Part 5: cache hit ratio (straight reuse of eval_router.py).
# ---------------------------------------------------------------------------

def run_cache_hit_ratio(model, tokenizer, prompt_ids, cache_layer, cache_size,
                        experts_per_token, use_topk, gen_len, batch_size,
                        device, out_path, model_name, variant):
    per_seq, overall, per_seq_misses, per_sample_ms = compute_cache_hit_rate(
        model, tokenizer, prompt_ids, cache_layer, cache_size,
        experts_per_token, use_topk, gen_len, batch_size, device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name, "variant": variant, "cache_layer": cache_layer,
            "cache_size": cache_size, "experts_per_token": experts_per_token,
            "n_prompts": len(prompt_ids), "overall_hit_rate": overall,
            "per_seq_hit_rate": per_seq, "per_seq_cache_misses": per_seq_misses,
        }, f, indent=2)
    print(f"[eval_complete][5/5] overall_hit_rate={overall:.4f} -> wrote {out_path}", flush=True)
    return {"overall_hit_rate": overall}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--cache-layer", type=int, default=-1)
    ap.add_argument("--cache-experts-per-token", type=int, default=2)
    ap.add_argument("--cache-topk", action="store_true")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default="math,code")
    ap.add_argument("--num-eval-prompts", type=int, default=1024)
    ap.add_argument("--num-viz-prompts", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-expert-load-trials", type=int, default=1024)
    ap.add_argument("--harness-total-budget", type=int, default=1024)
    ap.add_argument("--harness-num-seeds", type=int, default=1)
    ap.add_argument("--skip-parts", default="",
                    help="comma list of part numbers to skip, e.g. '1' to "
                         "skip the slow lm-eval-harness suite")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--out-dir-root", default="evals")
    args = ap.parse_args()
    skip = {int(x) for x in args.skip_parts.split(",") if x.strip()}

    from transformers import AutoTokenizer, AutoConfig
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(args.model)
    num_hidden_layers = config.num_hidden_layers
    num_experts = num_experts_of(config)
    cache_layer, other_layers = resolve_layers(num_hidden_layers, args.cache_layer)
    viz_layers = sorted({cache_layer, *other_layers})

    out_dir = Path(args.out_dir_root) / args.model.split("/")[-1] / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_complete] model={args.model} variant={args.variant} "
         f"num_hidden_layers={num_hidden_layers} num_experts={num_experts} "
         f"cache_layer={cache_layer} viz_layers={viz_layers} out_dir={out_dir}",
         flush=True)

    summary = {}

    if 1 not in skip:
        run_harness(args.model, args.variant, args.checkpoint_dir, args.batch_size,
                   args.harness_total_budget, args.harness_num_seeds, args.seed,
                   out_dir / "eval_harness.json")

    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()

    prompt_ids = build_eval_prompts(tok, args.dataset, args.dataset_split,
                                    args.num_eval_prompts, args.prompt_len, args.seed)
    viz_prompt_ids = prompt_ids[:args.num_viz_prompts]
    print(f"[eval_complete] sampled {len(prompt_ids)} shared prompts "
         f"({len(viz_prompt_ids)} for viz)", flush=True)

    if 2 not in skip:
        run_routing_distribution(model, tok, prompt_ids, cache_layer,
                                 args.cache_experts_per_token, num_experts,
                                 args.batch_size, device,
                                 out_dir / "eval_routing_distribution.json",
                                 args.model, args.variant)

    if 3 not in skip:
        run_routing_viz(model, tok, viz_prompt_ids, viz_layers, num_experts,
                        args.gen_len, device, out_dir / "eval_routing_viz.json",
                        out_dir / "expert_trace.png", args.model, args.variant)

    if 4 not in skip:
        r4 = run_throughput(model, tok, prompt_ids, cache_layer, args.cache_size,
                            args.cache_experts_per_token, args.cache_topk,
                            args.gen_len, args.batch_size, device, args.model,
                            args.num_expert_load_trials,
                            out_dir / "eval_throughput.json", args.variant)
        summary["tokens_per_second_mean"] = r4["tokens_per_second_mean"]

    if 5 not in skip:
        r5 = run_cache_hit_ratio(model, tok, prompt_ids, cache_layer, args.cache_size,
                                 args.cache_experts_per_token, args.cache_topk,
                                 args.gen_len, args.batch_size, device,
                                 out_dir / "eval_cache_hit_ratio.json",
                                 args.model, args.variant)
        summary["overall_hit_rate"] = r5["overall_hit_rate"]

    print(f"[eval_complete] done. summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
