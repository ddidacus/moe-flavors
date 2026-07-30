"""One-off: materialize a compact local copy of the Nemotron-v2 splits we
train/eval on, reading directly from the already-downloaded HF hub cache
(bypasses the gated-dataset Hub API check entirely -- loads the raw cached
parquet files via the `parquet` builder instead of `load_dataset(repo_id)`,
so it works even with an invalid/expired HF_TOKEN as long as the files were
downloaded once before while properly authenticated).

Keeps only the `messages` column (the only field build_prompt_dataset /
build_eval_sequences actually read) and caps each split at --max-rows,
matching the MAX_SCAN_PER_SPLIT bound already used by the reservoir sampler
in finetune_moe_grpo.py -- this is meant to reproduce the same sampling
pool, not the full multi-GB dataset.

Usage: python scripts/prepare_local_dataset.py
       (then see pyproject.toml's [tool.cluv] data_source/datasets_path,
       and src/nemotron_data.py for how training scripts load this)
"""
import argparse
import glob
from pathlib import Path

from datasets import Dataset, load_dataset

SPLITS = ["stem", "chat", "math", "code", "multilingual_ja", "multilingual_de",
          "multilingual_it", "multilingual_es", "multilingual_fr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-cache-dir", default=None,
                    help="path to the datasets--nvidia--Nemotron-Post-Training-"
                         "Dataset-v2 hub cache dir; default: $HF_HOME/hub/... "
                         "or ~/.cache/huggingface/hub/...")
    ap.add_argument("--max-rows-per-split", type=int, default=50_000)
    ap.add_argument("--out-dir", default="data/nemotron_v2")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="subset of SPLITS to (re)build; default: all")
    args = ap.parse_args()
    splits = args.splits or SPLITS

    if args.hub_cache_dir:
        cache_dir = Path(args.hub_cache_dir)
    else:
        import os
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache/huggingface")
        cache_dir = Path(hf_home) / "hub" / "datasets--nvidia--Nemotron-Post-Training-Dataset-v2"
    snapshots = sorted((cache_dir / "snapshots").iterdir())
    assert snapshots, f"no snapshot found under {cache_dir}/snapshots"
    snapshot = snapshots[-1]
    print(f"[prepare_local_dataset] reading from {snapshot}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in splits:
        files = sorted(glob.glob(str(snapshot / "data" / f"{sp}-*.parquet")))
        assert files, f"no parquet files found for split {sp} under {snapshot}/data"
        # streaming=True avoids datasets' full on-disk conversion cache for
        # the whole (multi-GB) split before we get to trim it down -- that
        # blew mila's $HOME quota. Collect only what we need, in memory.
        stream = load_dataset("parquet", data_files=files, split="train", streaming=True)
        rows = []
        for row in stream:
            rows.append({"messages": row["messages"]})
            if len(rows) >= args.max_rows_per_split:
                break
        ds = Dataset.from_list(rows)
        split_dir = out_dir / sp
        ds.save_to_disk(str(split_dir))
        print(f"[prepare_local_dataset] {sp}: {len(ds)} rows -> {split_dir}")

    print(f"[prepare_local_dataset] done -> {out_dir}")


if __name__ == "__main__":
    main()
