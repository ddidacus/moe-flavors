# MoE Playground

Mixture-of-Experts experiments on the [nemotron-moe-exam](https://huggingface.co/datasets/ddidacus/nemotron-moe-exam) dataset. Four experiment families compare different MoE routing strategies: vanilla top-k, temporal boundary-aware chunking, DeepSeek-style shared+routed experts, and LoRA fine-tuning of a pre-trained MoE model with optional temporal wrapping.

## Cache-consolidation / temporal-mixin / controller experiments (Phi-tiny-MoE)

The current active experiment line (see `handoff/` for full write-ups):
four LoRA fine-tunes of `microsoft/Phi-tiny-MoE-instruct` compared against
each other and the untouched base model -- plain SFT, GRPO + a cache-hit
reward, the same + a temporal hold/switch mixin, and an Option-Critic MoE
controller reimplementation (Shen & Henderson 2026).

**Run both experimental setups in order** -- small scale first (fast
end-to-end check, ~2k sequences, a few hours), then large scale (~10k
sequences, the two GRPO-based runs take 2-3.4 days each):

```bash
# 1. Small scale -- run this first
bash scripts/train_small_scale.sh

# 2. Large scale -- once small scale looks right
bash scripts/train_large_scale.sh
```

Each command submits 4 independent SLURM jobs (one per model:
`sft_baseline`, `cache_sft`, `temporal_moe`, `controller_baseline`) and
prints their job IDs. Full spec (dataset, sequence length, per-model
batch/lr, step counts, GPU-hour estimates) in
[`handoff/06-training-setup.md`](handoff/06-training-setup.md) (large) and
[`handoff/08-training-setup-small.md`](handoff/08-training-setup-small.md)
(small).

Once checkpoints exist, evaluate with:

```bash
# quantitative + qualitative eval, per checkpoint -- see handoff/07-eval-setup.md
sbatch scripts/eval/run_benchmarks.sh <variant1> [variant2]   # downstream tasks
sbatch scripts/eval/run_router.sh --checkpoint <path>          # routing/cache metrics
python scripts/eval/eval_postprocess.py                       # tables + plots from evals/
```

## Complete per-checkpoint eval (`eval_complete.py`)

Runs one `(model, variant)` checkpoint through five analyses in a single
process -- lm-eval-harness downstream tasks, per-expert routing
distribution (avg tokens/expert, skewness, expert contribution index),
an expert-choice visualization, an estimated offloaded-inference
throughput, and the on-policy cache hit ratio -- and writes results under
`evals/<model_name>/<date>/<variant>/eval_<part>.json` (+ one PNG). See
the script's own docstring for the full breakdown of what each part
measures. `eval_complete_cache_conditioned.py` is the same five analyses
run once per cache size for a `prompt_conditioned` checkpoint (which
expects a `[CACHE_SIZE=X]` prompt prefix at inference time), plus two
scaling plots (hit ratio and tokens/second vs. cache size).

### Running directly with bash

```bash
source .venv/bin/activate

# base model, no checkpoint
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant base

# a trained checkpoint -- --cache-size/--cache-experts-per-token/--cache-topk
# should match how it was actually trained (see scripts/cluv/train_*.sh)
python scripts/eval/eval_complete.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \
    --checkpoint-dir checkpoints/cache_reward_olmoe_tamia \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk

# cache-size-conditioned checkpoint, swept across every size it was trained on
python scripts/eval/eval_complete_cache_conditioned.py \
    --model microsoft/Phi-tiny-MoE-instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \
    --cache-sizes 2,4,8 --cache-experts-per-token 2 --cache-topk
```

Useful flags for a quick smoke test before a full run: `--num-eval-prompts
8 --num-viz-prompts 4 --num-expert-load-trials 16 --harness-total-budget 8
--harness-num-seeds 1 --gen-len 32 --batch-size 4`. `--skip-parts 1` skips
the slow lm-eval-harness suite (part 1) if you only want the routing/cache
metrics. Needs GPU with >=40GB VRAM for the full-size run (A100/A100L/
L40S/H100 or equivalent); a real run of all 5 parts on 1024 prompts has
taken 9-12 hours per checkpoint in practice.

### Submitting as SLURM jobs (no cluv)

`scripts/slurm/` has plain `sbatch` job scripts, independent of this
repo's own cluv-based tooling (`scripts/cluv/`, which is specific to the
cluster this project runs on) -- edit the `#SBATCH --partition`/
`--account`/`--gres` lines for your own cluster:

```bash
# one (model, variant) per job, single GPU
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant base

# the full sweep (every trained variant, both models), 2 scripts packed
# per job on separate GPUs -- edit the checkpoint paths inside the script
# first to match where your checkpoints live
bash scripts/slurm/submit_eval_sweep.sh

# prompt_conditioned checkpoints (cache-size sweep), one job per model
bash scripts/slurm/submit_eval_sweep_cache_conditioned.sh
```

`scripts/slurm/eval_complete_pair.sbatch` (used by both sweep scripts)
runs two eval jobs in parallel within one SLURM allocation, each pinned to
its own GPU via `scripts/eval/run_pair.sh` -- requests 2 GPUs + 8 CPUs
total (4 CPUs per script).

## Project structure

```
src/
  vanilla_moe.py           # Top-k MoE: Router, ExpertMLP, BatchedExperts, SegmentedExperts, MoEMixin
  temporal_moe.py          # Temporal MoE: ChunkingRouter (delta-state termination), ratio loss, MoEMixin
  deepseek_moe.py          # DeepSeek-style MoE: shared + routed experts, MoEMixin
  temporal_moe_wrapper.py  # Wraps an existing HF MoE model to add temporal boundary routing

scripts/
  train_moe.py             # Training script for experiments 1-3 (MoE from scratch on a dense model)
  finetune_moe.py          # Fine-tuning script for experiment 4 (LoRA + optional temporal wrapping)
  eval_harness.py          # Evaluate any checkpoint via lm-evaluation-harness
  run_vanilla_moe.sh       # SLURM: experiment 1
  run_temporal_moe.sh      # SLURM: experiment 2 (set LEARNABLE_N=1 for learnable N variant)
  run_deepseek_moe.sh      # SLURM: experiment 3
  run_finetune_moe.sh      # SLURM: experiment 4 (set TEMPORAL=1 for temporal variant)
```

## Checkpointing and preemption

All scripts handle SLURM preemption via `SIGUSR1`/`SIGTERM` signal handlers. When preempted, the current training state is saved. On resubmission, training resumes from the latest complete checkpoint (`--resume-from auto`). Checkpoints use a rotation policy (`keep_last=1`) to save disk space. Incomplete checkpoints (missing `.complete` marker) are skipped on resume.

W&B run IDs are persisted in checkpoint metadata for seamless run continuation across preemptions.

## Setup

```bash
uv sync
```
