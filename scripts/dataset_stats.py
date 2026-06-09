"""Show prompt-length distribution stats for each Nemotron v2 split."""

import argparse
import numpy as np
from datasets import load_dataset


SPLITS = ["stem", "chat", "math", "code"]


def prompt_lengths(ds):
    lengths = []
    for row in ds:
        prompt = "\n".join(
            m["content"] for m in row["messages"] if m["role"] == "user"
        )
        lengths.append(len(prompt))
    return np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap per split (default: load all)")
    args = parser.parse_args()

    for split in SPLITS:
        selector = split if args.max_samples is None else f"{split}[:{args.max_samples}]"
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                          split=selector, streaming=False)
        lens = prompt_lengths(ds)

        print(f"\n=== {split} ({len(lens)} samples) ===")
        print(f"  min    : {int(np.min(lens)):>8,}")
        print(f"  max    : {int(np.max(lens)):>8,}")
        print(f"  mean   : {np.mean(lens):>12,.1f}")
        print(f"  median : {np.median(lens):>12,.1f}")
        print(f"  std    : {np.std(lens):>12,.1f}")


if __name__ == "__main__":
    main()
