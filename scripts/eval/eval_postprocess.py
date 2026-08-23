"""Post-processing for the two eval pipelines (eval_benchmarks.py,
eval_router.py): reads every results_*.json under a benchmarks dir and a
router dir, and writes:

  - benchmarks_table.md   lm-eval-harness scores per variant per task, with
                          a per-row average and per-column bold-max
  - router_table.md       soft-cache metrics per variant (hit rate, hit-rate
                          std, cache-miss count [mean +/- std], and the
                          cached layer's forward throughput in ms/sample)
  - plots/avg_score.png       bar chart of average benchmark score per variant
  - plots/cache_hit_rate.png  bar chart of cache hit rate (+/- std) per variant
  - plots/score_vs_hitrate.png  scatter: x = cache hit rate, y = average
                          benchmark score, one point per variant

No GPU/model needed -- this only reads the JSON files written by the two
eval scripts (CPU-only, mirrors eval_benchmarks.py's --variant merge).

Usage:
    python scripts/eval/eval_postprocess.py \
        --benchmarks-dir evals/2026-08-03 \
        --router-dir evals/router_2026-08-10 \
        --out-dir results
"""
import argparse
import json
from pathlib import Path

# Variant/task registry, duplicated from eval_benchmarks.py rather than
# imported from it -- that module pulls in torch/lm_eval/peft at import
# time, which this CPU-only, no-GPU-needed script has no reason to load.
ALL_VARIANTS = ["base", "sft_baseline", "cache_sft", "temporal_moe",
                "controller_baseline", "melinoe", "sft_then_dapo"]
TASKS = ["mmlu", "mmmlu", "gsm8k", "humaneval", "hendrycks_math"]

# Display label and primary-metric key per benchmark task -- same convention
# as eval_benchmarks.py's merge().
TASK_LABELS = {"mmlu": "MMLU", "mmmlu": "MMMLU", "gsm8k": "GSM8K",
               "humaneval": "HumanEval", "hendrycks_math": "MATH"}
PRIMARY_METRIC = {"mmlu": "acc,none", "mmmlu": "acc,none",
                  "gsm8k": "exact_match,flexible-extract",
                  "humaneval": "pass@1,create_test",
                  "hendrycks_math": "exact_match,none"}
VARIANT_LABELS = {"base": "Base", "sft_baseline": "SFT baseline",
                  "cache_sft": "Cache SFT", "temporal_moe": "Temporal MoE",
                  "controller_baseline": "Controller baseline",
                  "melinoe": "Melinoe", "sft_then_dapo": "SFT + DAPO"}


# ---------------------------------------------------------------------------
# Loading: pull the one number lm-eval-harness / eval_router.py actually
# report per variant out of their raw JSON dumps.
# ---------------------------------------------------------------------------

def load_benchmarks(bench_dir):
    """variant -> {task: score}, scanning bench_dir for results_<variant>.json."""
    scores = {}
    for variant in ALL_VARIANTS:
        p = Path(bench_dir) / f"results_{variant}.json"
        if not p.exists():
            continue
        results = json.load(open(p))["results"]
        row = {}
        for task in TASKS:
            entry = results.get(task, {}).get(PRIMARY_METRIC[task])
            if entry is None:
                continue
            row[task] = entry["mean"] if isinstance(entry, dict) else entry
        scores[variant] = row
    return scores


def load_router(router_dir):
    """variant -> {hit_rate, hit_rate_std, cache_misses_mean,
    cache_misses_std, throughput_ms, cache_layer}, scanning router_dir for
    results_soft_cache_<variant>.json. cache_misses_*/throughput_ms are None
    for results written before eval_router.py started measuring them."""
    import statistics

    metrics = {}
    for variant in ALL_VARIANTS:
        p = Path(router_dir) / f"results_soft_cache_{variant}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        per_seq = d["per_seq_hit_rate"]
        metrics[variant] = {
            "hit_rate": d["overall_hit_rate"],
            "hit_rate_std": statistics.pstdev(per_seq) if len(per_seq) > 1 else 0.0,
            "cache_misses_mean": d.get("cache_misses_mean"),
            "cache_misses_std": d.get("cache_misses_std"),
            "throughput_ms": d.get("throughput_ms_per_sample"),
            "cache_layer": d["cache_layer"],
        }
    return metrics


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def write_benchmarks_table(scores, out_path):
    variants = [v for v in ALL_VARIANTS if v in scores]
    cols = TASKS + ["average"]
    for v in variants:
        present = [scores[v][t] for t in TASKS if t in scores[v]]
        scores[v]["average"] = sum(present) / len(present) if present else None

    # per-column max, for bolding -- ties all get bolded.
    col_max = {c: max((scores[v][c] for v in variants if scores[v].get(c) is not None),
                      default=None) for c in cols}

    header = ["Method"] + [TASK_LABELS.get(c, c.capitalize()) for c in cols]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for v in variants:
        row = [VARIANT_LABELS.get(v, v)]
        for c in cols:
            val = scores[v].get(c)
            if val is None:
                row.append("-")
                continue
            cell = f"{val:.4f}"
            if col_max[c] is not None and abs(val - col_max[c]) < 1e-9:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"[postprocess] wrote {out_path}")


def write_router_table(metrics, out_path):
    """Hit rate/std, cache-miss count (mean +/- std over eval samples), the
    cached layer's forward throughput (ms/sample), and the layer index --
    run length/eval-seq-count/experts-per-token were dropped since they're
    fixed eval-configuration knobs, not per-method results."""
    variants = [v for v in ALL_VARIANTS if v in metrics]
    col_max = {"hit_rate": max((metrics[v]["hit_rate"] for v in variants), default=None)}

    header = ["Method", "Hit rate", "Hit-rate std", "Cache misses (mean±std)",
              "Throughput (ms/sample)", "Cache layer"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for v in variants:
        m = metrics[v]
        hit_cell = f"{m['hit_rate']:.4f}"
        if col_max["hit_rate"] is not None and abs(m["hit_rate"] - col_max["hit_rate"]) < 1e-9:
            hit_cell = f"**{hit_cell}**"
        misses_cell = (f"{m['cache_misses_mean']:.2f} ± {m['cache_misses_std']:.2f}"
                      if m["cache_misses_mean"] is not None else "-")
        throughput_cell = (f"{m['throughput_ms']:.2f}"
                          if m["throughput_ms"] is not None else "-")
        row = [VARIANT_LABELS.get(v, v), hit_cell, f"{m['hit_rate_std']:.4f}",
              misses_cell, throughput_cell, str(m["cache_layer"])]
        lines.append("| " + " | ".join(row) + " |")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"[postprocess] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(scores, metrics, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in ALL_VARIANTS if v in scores]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    colors = plt.cm.tab10.colors

    # Average benchmark score per variant.
    fig, ax = plt.subplots(figsize=(7, 4))
    avgs = [scores[v].get("average") for v in variants]
    ax.bar(labels, avgs, color=[colors[i % 10] for i in range(len(variants))])
    ax.set_ylabel("Average LM-eval-harness score")
    ax.set_title("Average benchmark score per method")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "avg_score.png", dpi=150)
    plt.close(fig)

    # Cache hit rate (+/- std) per variant, only for variants with router data.
    router_variants = [v for v in variants if v in metrics]
    if router_variants:
        fig, ax = plt.subplots(figsize=(7, 4))
        hit_rates = [metrics[v]["hit_rate"] for v in router_variants]
        errs = [metrics[v]["hit_rate_std"] for v in router_variants]
        ax.bar([VARIANT_LABELS.get(v, v) for v in router_variants], hit_rates,
              yerr=errs, capsize=4,
              color=[colors[i % 10] for i in range(len(router_variants))])
        ax.set_ylabel("Overall cache hit rate (on-policy)")
        ax.set_title("Soft-cache hit rate per method")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(plots_dir / "cache_hit_rate.png", dpi=150)
        plt.close(fig)

    # Scatter: average benchmark score vs. cache hit rate, one point/variant.
    if router_variants:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, v in enumerate(router_variants):
            x = metrics[v]["hit_rate"]
            y = scores[v].get("average")
            if y is None:
                continue
            ax.scatter(x, y, s=80, color=colors[i % 10], label=VARIANT_LABELS.get(v, v))
        ax.set_xlabel("Overall cache hit rate (on-policy eval)")
        ax.set_ylabel("Average LM-eval-harness score")
        ax.set_title("Benchmark score vs. cache hit rate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "score_vs_hitrate.png", dpi=150)
        plt.close(fig)

    print(f"[postprocess] wrote plots to {plots_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks-dir", default="evals/benchmarks",
                    help="directory of results_<variant>.json from eval_benchmarks.py")
    ap.add_argument("--router-dir", default="evals/router",
                    help="directory of results_soft_cache_<variant>.json from eval_router.py")
    ap.add_argument("--out-dir", default="results",
                    help="where to write the markdown tables and plots/ dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_benchmarks(args.benchmarks_dir)
    metrics = load_router(args.router_dir)
    if not scores:
        print(f"[postprocess] no benchmark results found under {args.benchmarks_dir}")
    if not metrics:
        print(f"[postprocess] no router results found under {args.router_dir}")

    if scores:
        write_benchmarks_table(scores, out_dir / "benchmarks_table.md")
    if metrics:
        write_router_table(metrics, out_dir / "router_table.md")
    if scores and metrics:
        make_plots(scores, metrics, out_dir / "plots")


if __name__ == "__main__":
    main()
