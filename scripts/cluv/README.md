# cluv job scripts

Per-cluster SLURM job scripts used by `cluv submit <cluster> -- <program> [args...]`,
wired up via `job_script_path` in `pyproject.toml` (`[tool.cluv.clusters.*]`).

Each `<cluster>_job.sh` only sets the `#SBATCH` resource header for that
cluster's GPUs, then sources `common.sh`, which sets up the venv/caches,
forces `WANDB_MODE=disabled` (compute nodes on these clusters have no
internet — see below), and execs the program passed after `--` with a
3-attempt retry for fast (<10min) startup failures (shared-FS flakiness).
Non-GPU settings (account, timelimit, requeue) come from `[tool.cluv.env]`
/ `[tool.cluv.clusters.*].env` in `pyproject.toml` — no need to duplicate
them here.

## wandb

`common.sh` unconditionally sets `WANDB_MODE=disabled` for every cluv job,
regardless of any `WANDB_MODE` set in `pyproject.toml`. Even wandb's
"offline" mode does DNS/network probing on init that can hang or error on
a node with no internet (tamia, rorqual, narval; also vulcan, to be safe),
so cluv jobs rely on the SLURM stdout log (`%x_%j.out`, in the project
root) instead of wandb. Metrics/progress should be `print`ed or logged to
that file. This does not affect the existing mila `sbatch` scripts
(`scripts/run_finetune_moe_*.sh`), which are submitted directly, not
through cluv, and keep using wandb online as before.

`scripts/job.sh` is the generic fallback cluv uses for any cluster without
its own `job_script_path` entry (e.g. mila, killarney, trillium — mila jobs
are submitted directly with `sbatch` today, not through cluv).

## Cluster GPU summary

Researched 2026-07-24 from alliancecan.ca service pages + Mila's DRAC docs
(docs.alliancecan.ca itself was blocked by bot protection when fetched
directly — figures below are secondary-sourced, cross-check with
`cluv status` / `sinfo -N` before trusting them for a large run).

| Cluster  | GPU model    | GPUs/node | Notes |
|----------|--------------|-----------|-------|
| tamia    | H100 80GB (H200 also available) | 4, whole-node only | PAICE, Université Laval. No internet on compute nodes. |
| rorqual  | H100 80GB    | 4 | Successor to Béluga, ÉTS Montréal. No internet — `module load httpproxy` for wandb if needed. |
| narval   | A100 40GB    | 4 | ÉTS Montréal. Older/lower-throughput than the H100 clusters. No internet. Access depends on supervisor affiliation. |
| vulcan   | L40S 48GB    | 4 (64 CPU cores/node, confirmed) | University of Alberta / Amii, AI-dedicated. Less VRAM than Mila's A100L — may need smaller batch size. |
| fir      | H100 80GB    | 4 | Successor to Cedar. Unrestricted internet. |
| nibi     | H100 80GB    | 4 | Successor to Graham (also has unused AMD MI300A nodes). Unrestricted internet. |

All six job scripts request `--exclusive` (whole node) instead of guessing
exact CPU-core/memory-per-node figures, since every training script here
already uses `accelerate launch --num_processes 4` (4 GPUs) and tamIA in
particular mandates whole-node allocation anyway.

## Usage

```bash
# submit to a specific cluster, program args after `--`
cluv submit tamia -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/finetune_moe_sft.py --model microsoft/Phi-tiny-MoE-instruct ...

# --autocommit must come BEFORE the cluster name -- cluv's own sbatch_args
# parser (argparse.REMAINDER) silently swallows it if placed after, see
# train_/eval_ scripts below for the correct order.
cluv submit --autocommit rorqual --time=1-00:00:00 -- accelerate launch ...

# see scripts/train_small_scale.sh for the CLUSTER=<name> wrapper used
# to fan all 4 small-scale training jobs out to any of these clusters at once
CLUSTER=fir bash scripts/train_small_scale.sh
```

## train_/eval_ scripts (one variant/job at a time)

Standalone, per-variant alternative to `scripts/train_small_scale.sh` --
useful for resubmitting a single failed variant, or running eval without
retraining everything. Each takes `CLUSTER` as a required env var.

| Script | What it submits | Saves / reads |
|---|---|---|
| `train_sft_baseline.sh` | `finetune_moe_sft.py` | `checkpoints/sft_baseline_<CLUSTER>` |
| `train_cache_sft.sh` | `finetune_moe_grpo.py` (soft cache reward) | `checkpoints/cache_sft_<CLUSTER>` |
| `train_temporal_moe.sh` | `finetune_moe_grpo.py` (+ `--temporal`) | `checkpoints/temporal_moe_<CLUSTER>` |
| `train_controller_baseline.sh` | `finetune_moe_controller.py` | `checkpoints/controller_baseline_<CLUSTER>` |
| `eval_lm_harness.sh [variant...]` | `scripts/eval_lm_harness.py` (lm-eval-harness: MMLU/MMMLU/GSM8K/HumanEval/MATH) | reads `checkpoints/<variant>_<CLUSTER>`, writes `evals/<CLUSTER>/<date>/` |
| `eval_soft_cache.sh [variant...]` | `scripts/eval_soft_cache.py` (LRU cache-hit-rate; **stub**, see its docstring) | reads `checkpoints/<variant>_<CLUSTER>`, writes `evals/<CLUSTER>/soft_cache_<date>/` |

The two `eval_*.sh` scripts default to all 5 variants (`base`,
`sft_baseline`, `cache_sft`, `temporal_moe`, `controller_baseline`) run in
parallel, GPU-pinned, within one job (`_eval_lm_harness_run.sh` /
`_eval_soft_cache_run.sh` are the actual remote drivers -- not meant to be
invoked directly; kept as real files rather than inline shell strings
because `cluv submit` breaks on nested quotes/parens passed as program args).

```bash
CLUSTER=fir bash scripts/cluv/train_cache_sft.sh
CLUSTER=fir bash scripts/cluv/eval_lm_harness.sh cache_sft temporal_moe
CLUSTER=fir bash scripts/cluv/eval_soft_cache.sh cache_sft temporal_moe
```

`eval_lm_harness.py` and `eval_soft_cache.py` both accept a
`--checkpoint-dir` override (added specifically for cluv checkpoints, whose
`<variant>_<CLUSTER>` naming differs from the mila `run_finetune_moe_*.sh`
`VARIANT_CHECKPOINTS` convention) -- the `eval_*.sh` wrappers pass it
automatically, you shouldn't need to set it by hand.
