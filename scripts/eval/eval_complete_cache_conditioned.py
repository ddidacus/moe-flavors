"""Cache-size-conditioned variant of eval_complete.py: sweeps every cache
size the prompt-conditioned checkpoint was trained on (e.g. `2,4,8` for
Phi-tiny-MoE, `8,16,32` for OLMoE -- see scripts/cluv/
train_prompt_conditioned_olmoe.sh), prefixing every prompt with
`[CACHE_SIZE=X]` (matching training-time conditioning, since this
checkpoint never saw an unconditioned prompt) before running the same
parts 2-5 as eval_complete.py, once per cache size.

DEVIATION from eval_complete.py: part 1 (lm-eval-harness) runs only ONCE,
unconditioned. lm_eval's own task/prompt templates (MMLU, GSM8K, etc.) are
not under our control at the per-example level, so there's no clean way to
prepend our `[CACHE_SIZE=X]` prefix inside its prompt construction --
faking it would require monkeypatching lm_eval internals for a benefit
that's unclear anyway (the harness tasks aren't cache-hit-rate-relevant to
begin with). Written once to `eval_harness.json` at the top level (not
nested under a cache-size subdirectory).

Output layout: `evals/<model_name>/<date>/eval_harness.json` (once) plus,
per cache size, `evals/<model_name>/<date>/cache<N>/eval_<part>.json` for
parts 2-5, plus two scaling plots at the top level:
`scaling_hit_ratio.png` (x = cache size, descending, y = hit ratio) and
`scaling_tokens_per_second.png` (same x axis, y = tokens/sec) -- two
separate figures sharing the x-axis convention, not one dual-axis plot.

Usage:
    python scripts/eval/eval_complete_cache_conditioned.py \\
        --model microsoft/Phi-tiny-MoE-instruct \\
        --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \\
        --cache-sizes 2,4,8
    python scripts/eval/eval_complete_cache_conditioned.py \\
        --model allenai/OLMoE-1B-7B-0125-Instruct \\
        --checkpoint-dir checkpoints/prompt_conditioned_olmoe_tamia \\
        --cache-sizes 8,16,32 --cache-experts-per-token 8 --cache-topk
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root

from eval_benchmarks import build_variant_model  # noqa: E402
from eval_cache_conditioning import sample_eval_prompt_texts  # noqa: E402
from eval_complete import (  # noqa: E402
    num_experts_of, resolve_layers, run_harness, run_routing_distribution,
    run_routing_viz, run_throughput, run_cache_hit_ratio,
)
from train_prompt_conditioned import CACHE_PREFIX_TEMPLATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", default="prompt_conditioned")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--cache-sizes", default="2,4,8",
                    help="comma list; use the set the checkpoint was "
                         "actually trained on (2,4,8 for phi-tiny-moe, "
                         "8,16,32 for OLMoE)")
    ap.add_argument("--cache-experts-per-token", type=int, default=2)
    ap.add_argument("--cache-topk", action="store_true")
    ap.add_argument("--cache-layer", type=int, default=-1)
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
                    help="comma list of part numbers to skip (applied per "
                         "cache size for parts 2-5; part 1 always runs at "
                         "most once regardless)")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--out-dir-root", default="evals")
    args = ap.parse_args()
    skip = {int(x) for x in args.skip_parts.split(",") if x.strip()}
    cache_sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]

    from transformers import AutoTokenizer, AutoConfig
    import torch
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
    print(f"[eval_complete_cc] model={args.model} cache_sizes={cache_sizes} "
         f"cache_layer={cache_layer} viz_layers={viz_layers} out_dir={out_dir}",
         flush=True)

    if 1 not in skip:
        run_harness(args.model, args.variant, args.checkpoint_dir, args.batch_size,
                   args.harness_total_budget, args.harness_num_seeds, args.seed,
                   out_dir / "eval_harness.json")

    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()

    base_texts = sample_eval_prompt_texts(tok, args.dataset, args.dataset_split,
                                          args.num_eval_prompts, args.seed)
    print(f"[eval_complete_cc] sampled {len(base_texts)} shared base prompts "
         f"(unprefixed)", flush=True)

    hit_ratio_by_size = {}
    tok_per_sec_by_size = {}

    for cache_size in cache_sizes:
        prefix = CACHE_PREFIX_TEMPLATE.format(size=cache_size)
        prompt_ids = [
            tok(f"{prefix} {t}", truncation=True, max_length=args.prompt_len,
               add_special_tokens=False)["input_ids"]
            for t in base_texts
        ]
        viz_prompt_ids = prompt_ids[:args.num_viz_prompts]
        cs_dir = out_dir / f"cache{cache_size}"
        cs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval_complete_cc] === cache_size={cache_size} "
             f"({len(prompt_ids)} prefixed prompts) ===", flush=True)

        if 2 not in skip:
            run_routing_distribution(model, tok, prompt_ids, cache_layer,
                                     args.cache_experts_per_token, num_experts,
                                     args.batch_size, device,
                                     cs_dir / "eval_routing_distribution.json",
                                     args.model, f"{args.variant}_cache{cache_size}")

        if 3 not in skip:
            run_routing_viz(model, tok, viz_prompt_ids, viz_layers, num_experts,
                            args.gen_len, device, cs_dir / "eval_routing_viz.json",
                            cs_dir / "expert_trace.png", args.model,
                            f"{args.variant}_cache{cache_size}")

        if 4 not in skip:
            r4 = run_throughput(model, tok, prompt_ids, cache_layer, cache_size,
                                args.cache_experts_per_token, args.cache_topk,
                                args.gen_len, args.batch_size, device, args.model,
                                args.num_expert_load_trials,
                                cs_dir / "eval_throughput.json",
                                f"{args.variant}_cache{cache_size}")
            tok_per_sec_by_size[cache_size] = r4["tokens_per_second_mean"]

        if 5 not in skip:
            r5 = run_cache_hit_ratio(model, tok, prompt_ids, cache_layer, cache_size,
                                     args.cache_experts_per_token, args.cache_topk,
                                     args.gen_len, args.batch_size, device,
                                     cs_dir / "eval_cache_hit_ratio.json",
                                     args.model, f"{args.variant}_cache{cache_size}")
            hit_ratio_by_size[cache_size] = r5["overall_hit_rate"]

    if hit_ratio_by_size or tok_per_sec_by_size:
        _make_scaling_plots(out_dir, cache_sizes, hit_ratio_by_size,
                            tok_per_sec_by_size, args.model, args.variant)

    print(f"[eval_complete_cc] done. hit_ratio_by_size={hit_ratio_by_size} "
         f"tok_per_sec_by_size={tok_per_sec_by_size}", flush=True)


def _make_scaling_plots(out_dir, cache_sizes, hit_ratio_by_size,
                        tok_per_sec_by_size, model_name, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes_desc = sorted(cache_sizes, reverse=True)
    title_suffix = f"{model_name.split('/')[-1]} / {variant}"

    if hit_ratio_by_size:
        xs = [s for s in sizes_desc if s in hit_ratio_by_size]
        ys = [hit_ratio_by_size[s] for s in xs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([str(x) for x in xs], ys, marker="o")
        ax.set_xlabel("cache size (descending)")
        ax.set_ylabel("cache hit ratio")
        ax.set_title(f"Hit ratio vs. cache size\n{title_suffix}", fontsize=9)
        fig.tight_layout()
        path = out_dir / "scaling_hit_ratio.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[eval_complete_cc] wrote {path}", flush=True)

    if tok_per_sec_by_size:
        xs = [s for s in sizes_desc if s in tok_per_sec_by_size]
        ys = [tok_per_sec_by_size[s] for s in xs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([str(x) for x in xs], ys, marker="o", color="tab:orange")
        ax.set_xlabel("cache size (descending)")
        ax.set_ylabel("tokens / second")
        ax.set_title(f"Throughput vs. cache size\n{title_suffix}", fontsize=9)
        fig.tight_layout()
        path = out_dir / "scaling_tokens_per_second.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[eval_complete_cc] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
