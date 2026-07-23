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
sbatch scripts/run_eval_lm_harness.sh <variant1> [variant2]   # downstream tasks
sbatch scripts/run_eval_soft_cache.sh --checkpoint <path>     # routing/cache metrics
```

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
