"""Token-length distribution of the Nemotron-v2 dataset's prompts and
completions -- run this BEFORE picking --prompt-len/--completion-len for
the training scripts in this directory, so the choice is based on where
the data actually falls rather than a guess.

Samples N rows per split (streaming, no full download), tokenizes the
prompt (every user/system turn concatenated before the first assistant
turn -- same construction as sample_filtered_prompts) and the completion
(the first assistant turn) with the target tokenizer, and reports the
percentile distribution of each in tokens. No GPU needed.

Usage:
    python scripts/train/analyze_dataset_lengths.py
    python scripts/train/analyze_dataset_lengths.py --dataset-split math,code \
        --samples-per-split 2000 --out-dir results/dataset_lengths
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# The 9 splits every cluv train_*.sh trains on ("allsplits" in this repo's
# shorthand) -- see src/nemotron_data.py / finetune_moe_grpo.py's docstrings.
ALL_SPLITS = ["stem", "chat", "math", "code", "multilingual_ja",
             "multilingual_de", "multilingual_it", "multilingual_es",
             "multilingual_fr"]

PERCENTILES = [50, 75, 90, 95, 99, 100]


# ---------------------------------------------------------------------------
# Sampling + tokenization: same prompt/completion construction as
# src.nemotron_data.sample_filtered_prompts, but every scanned row is kept
# (this is a measurement pass, not a training-data filter).
# ---------------------------------------------------------------------------

def collect_lengths(tokenizer, dataset_name, splits, samples_per_split):
    from src.nemotron_data import load_split_stream

    prompt_lens, completion_lens = [], []
    per_split = {}
    for sp in splits:
        ds = load_split_stream(dataset_name, sp)
        sp_prompt_lens, sp_completion_lens = [], []
        for row in ds:
            if len(sp_prompt_lens) >= samples_per_split:
                break
            parts = []
            target_text = None
            for m in row["messages"]:
                if m["role"] == "assistant":
                    target_text = m["content"]
                    break
                if m["content"].strip():
                    parts.append(m["content"])
            text = "\n".join(parts).strip()
            if not (text and target_text and target_text.strip()):
                continue
            p_len = len(tokenizer(text)["input_ids"])
            c_len = len(tokenizer(target_text.strip(),
                                  add_special_tokens=False)["input_ids"])
            sp_prompt_lens.append(p_len)
            sp_completion_lens.append(c_len)
        per_split[sp] = {"prompt": sp_prompt_lens, "completion": sp_completion_lens}
        prompt_lens.extend(sp_prompt_lens)
        completion_lens.extend(sp_completion_lens)
        print(f"[analyze] {sp}: {len(sp_prompt_lens)} rows sampled", flush=True)
    return prompt_lens, completion_lens, per_split


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[idx]


def summarize(values):
    return {f"p{p}": percentile(values, p) for p in PERCENTILES} | {
        "mean": sum(values) / len(values) if values else None,
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Reporting: printed table + optional JSON dump + histogram plot.
# ---------------------------------------------------------------------------

def print_table(name, values):
    s = summarize(values)
    print(f"\n[{name}] n={s['n']} mean={s['mean']:.1f}", flush=True)
    header = "  " + "  ".join(f"p{p}" for p in PERCENTILES)
    row = "  " + "  ".join(str(s[f"p{p}"]) for p in PERCENTILES)
    print(header)
    print(row)


def make_histogram(prompt_lens, completion_lens, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title in [(axes[0], prompt_lens, "Prompt length (tokens)"),
                              (axes[1], completion_lens, "Completion length (tokens)")]:
        ax.hist(values, bins=60, color="#4c72b0")
        for p in (90, 95, 99):
            ax.axvline(percentile(values, p), color="crimson", linestyle="--",
                      linewidth=1, label=f"p{p}={percentile(values, p)}")
        ax.set_title(title)
        ax.set_xlabel("tokens")
        ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--dataset-split", default=",".join(ALL_SPLITS))
    ap.add_argument("--samples-per-split", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/dataset_lengths")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    splits = [s.strip() for s in args.dataset_split.split(",") if s.strip()]
    prompt_lens, completion_lens, per_split = collect_lengths(
        tok, args.dataset, splits, args.samples_per_split)

    print_table("prompt", prompt_lens)
    print_table("completion", completion_lens)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model, "dataset": args.dataset, "splits": splits,
        "samples_per_split": args.samples_per_split,
        "prompt": summarize(prompt_lens), "completion": summarize(completion_lens),
        "per_split": {sp: {"prompt": summarize(d["prompt"]),
                           "completion": summarize(d["completion"])}
                     for sp, d in per_split.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] wrote {out_dir / 'summary.json'}", flush=True)

    make_histogram(prompt_lens, completion_lens, out_dir / "length_histogram.png")


if __name__ == "__main__":
    main()
