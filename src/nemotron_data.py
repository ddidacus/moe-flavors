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
