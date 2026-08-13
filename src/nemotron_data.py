"""Shared dataset-split loader for the Nemotron-v2 training/eval pipelines.

Loads from a local pre-materialized copy (see scripts/prepare_local_dataset.py)
when available -- bypasses the gated-dataset Hub API check entirely, so it
works even with an invalid/expired HF_TOKEN, and works fully offline on
clusters with no internet (tamia, rorqual, narval, vulcan). Falls back to
streaming from the Hub otherwise.
"""
import os
from pathlib import Path

NEMOTRON_DATASET_NAME = "nvidia/Nemotron-Post-Training-Dataset-v2"


def _local_dataset_root() -> Path | None:
    """Local materialized dir, if present. Layout: <root>/<split>/ (one
    datasets.load_from_disk() dir per split) -- matches
    scripts/prepare_local_dataset.py's --out-dir.

    Checked in order: NEMOTRON_LOCAL_DATA env var (explicit override) ->
    <project_root>/data/nemotron_v2 (mila, symlinked to $SCRATCH) ->
    $SCRATCH/datasets/nemotron_v2 (every other cluster: cluv's
    datasets_path pushes the data there via `cluv sync`, but -- unlike
    results_path -- does NOT auto-symlink it into the project, so this is
    checked directly rather than relying on shell expansion of $SCRATCH
    inside an exported env var)."""
    candidates = []
    if "NEMOTRON_LOCAL_DATA" in os.environ:
        candidates.append(Path(os.environ["NEMOTRON_LOCAL_DATA"]))
    candidates.append(Path(__file__).resolve().parent.parent / "data/nemotron_v2")
    if "SCRATCH" in os.environ:
        candidates.append(Path(os.environ["SCRATCH"]) / "datasets/nemotron_v2")
    for root in candidates:
        if root.is_dir():
            return root
    return None


def load_split_stream(dataset_name: str, split: str):
    """Drop-in replacement for load_dataset(dataset_name, split=split,
    streaming=True) -- returns something iterable row-by-row (each row a
    dict with at least 'messages'). Uses the local materialized copy for
    NEMOTRON_DATASET_NAME when available, else streams from the Hub."""
    if dataset_name == NEMOTRON_DATASET_NAME:
        local_root = _local_dataset_root()
        if local_root is not None:
            split_dir = local_root / split
            if split_dir.is_dir():
                from datasets import load_from_disk
                return load_from_disk(str(split_dir))
    from datasets import load_dataset
    return load_dataset(dataset_name, split=split, streaming=True)


def sample_filtered_prompts(tokenizer, dataset_name, split, max_samples,
                            prompt_len, seed, skip_first=0,
                            scan_batch_size=1000, max_scan_per_split=50_000):
    """Reservoir-samples (Algorithm R) up to max_samples (prompt_ids,
    target_text) pairs across `split` (a comma-separated list of split
    names), evenly divided per split. Every finetune_moe_*.py training
    script's build_prompt_dataset/build_sft_dataset used to duplicate this
    scan loop; it now lives here once.

    Filtering, not truncation: rows whose prompt tokenizes to more than
    prompt_len tokens are dropped rather than clipped, so every returned
    prompt is a complete, untruncated conversation turn (the reservoir may
    end up with fewer than max_samples // len(splits) items per split if
    many candidates are filtered out). target_text is returned as a raw,
    untruncated string -- each caller truncates/formats the completion side
    its own way (e.g. as message-list "completion" for SFTTrainer vs.
    tokenized target_ids for the GRPO/controller/melinoe NLL losses).

    The length filter requires tokenizing every scanned row (unlike the old
    truncate-after-sampling code, which only tokenized the much smaller
    finalized reservoir) -- to keep that affordable, prompts are tokenized
    in scan_batch_size-row batches via the fast tokenizer's batch call
    rather than one row at a time. One-row-at-a-time tokenization across
    every scanned row (up to max_scan_per_split x num_splits, e.g. 9 splits
    x 50k = 450k individual calls) is what used to blow past accelerate's
    600s multi-GPU rendezvous timeout; batching keeps the same total token
    count but a fraction of the Python/call overhead.
    """
    import random

    rng = random.Random(seed)
    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
    sampled = []

    for sp in splits:
        ds = load_split_stream(dataset_name, sp)
        reservoir = []  # (prompt_ids, target_text), already length-filtered
        seen = 0
        scanned = 0
        batch_texts, batch_targets = [], []

        def flush_batch():
            nonlocal seen
            if not batch_texts:
                return
            for ids, target_text in zip(tokenizer(list(batch_texts))["input_ids"],
                                        batch_targets):
                if len(ids) > prompt_len:
                    continue  # filtered out: prompt too long, not truncated
                item = (ids, target_text)
                if len(reservoir) < per_split:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, seen)
                    if j < per_split:
                        reservoir[j] = item
                seen += 1
            batch_texts.clear()
            batch_targets.clear()

        for r_idx, row in enumerate(ds):
            if r_idx < skip_first:
                continue
            if scanned >= max(max_scan_per_split, per_split):
                break
            scanned += 1
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
            batch_texts.append(text)
            batch_targets.append(target_text.strip())
            if len(batch_texts) >= scan_batch_size:
                flush_batch()
        flush_batch()
        sampled.extend(reservoir)

    return sampled
