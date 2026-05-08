# MoE Playground

Proof-of-concept for **boundary-aware Mixture-of-Experts (MoE)** applied to pretrained dense language models. Instead of routing each token independently, a learned chunking router detects segment boundaries in the token sequence and forward-fills routing decisions across each segment, so tokens within the same chunk share the same experts.

## How it works

1. A dense pretrained LM (default: Qwen2-0.5B) is loaded.
2. Every FFN/MLP layer is replaced with a `ChunkingMoELayer` containing N expert copies of the original FFN plus a boundary-aware router.
3. The **ChunkingRouter** computes boundary probabilities between adjacent tokens using learned query/key projections and cosine distance. Tokens where `p_t >= threshold` start a new segment; routing decisions are forward-filled within each segment via `cummax`.
4. A **ratio loss** encourages the router to produce meaningful segmentation and is folded into the model's forward pass for clean DDP compatibility.

## Project structure

```
src/
  temporal_moe.py    # ChunkingRouter, ChunkingMoELayer, MoEMixin, MoEConfig
  vanilla_moe.py     # Baseline top-k MoE (no chunking) for comparison
scripts/
  moe_mixin_poc.py   # Training script (FineWeb data, wandb logging, boundary eval)
  moe_chunking.sh    # SLURM launch script (multi-GPU via accelerate)
```

## Setup

```bash
uv sync
```

## Running

**Multi-GPU (SLURM):**

```bash
sbatch scripts/moe_chunking.sh
```

**Single GPU:**

```bash
python scripts/moe_mixin_poc.py
```

**Key arguments** (see `scripts/moe_mixin_poc.py --help`):

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2-0.5B` | HuggingFace model to convert |
| `--num-experts` | `8` | Number of expert copies per MLP layer |
| `--top-k` | `2` | Experts active per token |
| `--seq-len` | `256` | Sequence length |
| `--batch-size` | `16` | Per-device batch size |
| `--num-steps` | `100000` | Total training steps |
| `--lr` | `1e-4` | Learning rate |
| `--eval-every` | `50` | Steps between boundary visualizations |

Training logs (loss curves, boundary visualizations, expert assignment plots) are sent to [Weights & Biases](https://wandb.ai) under the `moe-chunking-poc` project.
