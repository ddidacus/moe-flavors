# MoE Playground

Mixture-of-Experts experiments on the [nemotron-moe-exam](https://huggingface.co/datasets/ddidacus/nemotron-moe-exam) dataset. Four experiment families compare different MoE routing strategies: vanilla top-k, temporal boundary-aware chunking, DeepSeek-style shared+routed experts, and LoRA fine-tuning of a pre-trained MoE model with optional temporal wrapping.

## Experiments

### 1. Vanilla MoE

Standard per-token top-k gating with a Switch Transformer load-balancing loss. Each FFN layer in a dense model (Qwen3-0.6B) is replaced with N small expert MLPs and a learned router.

```bash
sbatch scripts/run_vanilla_moe.sh
```

### 2. Temporal MoE

Boundary-aware chunking router that detects segment boundaries in the token sequence via delta-state termination (option-critic style). Tokens within a segment share the same expert routing. A **ratio loss** encourages meaningful segmentation, with variable N per layer. Supports a **learnable-N** variant where N is optimized during training.

```bash
# Fixed N
sbatch scripts/run_temporal_moe.sh

# Learnable N variant
LEARNABLE_N=1 sbatch scripts/run_temporal_moe.sh
```

### 3. DeepSeek-style MoE

Combines always-active shared experts with top-k routed experts, following the DeepSeek MoE architecture. Shared experts process all tokens (no gating), while routed experts are selected per-token via top-k routing.

```bash
sbatch scripts/run_deepseek_moe.sh
```

### 4. Fine-tune pre-trained MoE (gpt-oss-20b + LoRA)

LoRA fine-tuning of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (32 experts, top-4, 20B params). Two variants: baseline LoRA fine-tuning, and LoRA + temporal boundary routing on top of the existing MoE experts.

```bash
# Baseline LoRA fine-tuning
sbatch scripts/run_finetune_moe.sh

# LoRA + temporal boundary routing
TEMPORAL=1 sbatch scripts/run_finetune_moe.sh
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

## How it works

### MoE from scratch (experiments 1-3)

1. A dense pretrained LM (Qwen3-0.6B) is loaded.
2. Every FFN/MLP layer is replaced with an MoE layer: N expert MLPs + a router.
3. The model is trained on nemotron-moe-exam with the MoE routing and any auxiliary losses.

**Vanilla:** Per-token top-k routing with load-balancing auxiliary loss.

**Temporal:** The `ChunkingRouter` computes boundary probabilities from delta states (consecutive hidden state differences). Positions where `p_t >= 0.5` start new segments; routing decisions are forward-filled within each segment via `cummax`. A ratio loss encourages the router to produce segments of target length N. The learnable-N variant optimizes N during training via a softplus-parameterized scalar per layer.

**DeepSeek:** Shared experts (always-active, no gating) are combined with routed experts (top-k selected per token). The shared experts stabilize training while routed experts specialize.

### LoRA fine-tuning (experiment 4)

1. gpt-oss-20b is loaded with its existing MoE architecture intact (32 experts, top-4).
2. LoRA adapters are applied to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
3. Optionally, temporal boundary routing is wrapped around the existing MoE blocks — only the boundary prediction layers and LoRA adapters are trained; the original experts remain frozen.

## Key arguments

### `train_moe.py` (experiments 1-3)

| Flag | Default | Description |
|------|---------|-------------|
| `--moe-type` | `temporal` | `vanilla`, `temporal`, `deepseek`, or `temporal-wrap` |
| `--model` | `Qwen/Qwen3-0.6B` | Base dense model to convert |
| `--num-experts` | `8` | Number of expert MLPs per layer |
| `--top-k` | `2` | Experts active per token |
| `--expert-dim` | auto | Per-expert intermediate dim |
| `--ratio-loss-N` | `3` | Target segment length (one per layer, or single value for all) |
| `--learnable-N` | off | Make N a learnable parameter |
| `--num-shared-experts` | `2` | Shared experts (DeepSeek mode only) |

### `finetune_moe.py` (experiment 4)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openai/gpt-oss-20b` | Pre-trained MoE model |
| `--temporal` | off | Add temporal boundary routing |
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--lora-target-modules` | `q/k/v/o_proj` | Modules to apply LoRA to |

## Checkpointing and preemption

All scripts handle SLURM preemption via `SIGUSR1`/`SIGTERM` signal handlers. When preempted, the current training state is saved. On resubmission, training resumes from the latest complete checkpoint (`--resume-from auto`). Checkpoints use a rotation policy (`keep_last=1`) to save disk space. Incomplete checkpoints (missing `.complete` marker) are skipped on resume.

W&B run IDs are persisted in checkpoint metadata for seamless run continuation across preemptions.

## Setup

```bash
uv sync
```

## Evaluate a checkpoint

```bash
python scripts/eval_harness.py --checkpoint-dir checkpoints/vanilla_moe_64e_k8/step_1000
```

Metrics are logged to W&B and optionally saved as JSON (`--output-dir`).
