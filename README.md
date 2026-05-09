# MoE Playground

Proof-of-concept for **Mixture-of-Experts (MoE)** applied to pretrained dense language models. Supports two MoE variants:

- **Temporal (chunking):** A learned chunking router detects segment boundaries in the token sequence via cosine distance and forward-fills routing decisions across each segment, so tokens within the same chunk share the same experts.
- **Vanilla:** Standard per-token top-k gating with a load-balancing auxiliary loss.

## How it works

1. A dense pretrained LM (default: Qwen3-0.6B) is loaded.
2. Every FFN/MLP layer is replaced with an MoE layer containing N expert copies of the original FFN plus a router.
3. **Temporal mode:** The `ChunkingRouter` computes boundary probabilities between adjacent tokens using learned query/key projections and cosine distance. Tokens where `p_t >= threshold` start a new segment; routing decisions are forward-filled within each segment via `cummax`. A **ratio loss** encourages the router to produce meaningful segmentation and is folded into the model's forward pass for clean DDP compatibility.
4. **Vanilla mode:** A standard `Router` computes per-token top-k expert assignments with a Switch Transformer style load-balancing loss.

## Project structure

```
src/
  temporal_moe.py    # ChunkingRouter, ChunkingMoELayer, MoEMixin, MoEConfig
  vanilla_moe.py     # Baseline top-k MoE (no chunking) for comparison
scripts/
  moe_mixin_poc.py   # Training script (Nemotron SFT data, wandb logging, boundary eval)
  moe_chunking.sh    # SLURM launch script (multi-GPU via accelerate)
```

## Dataset

Training uses the [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) SFT dataset (gated, requires HF access). Messages are concatenated into a single text per sample for causal LM training. Available splits: `code`, `math`, `stem`, `chat`, and multilingual variants.

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
python scripts/moe_mixin_poc.py --moe-type temporal
```

**Key arguments** (see `scripts/moe_mixin_poc.py --help`):

| Flag | Default | Description |
|------|---------|-------------|
| `--moe-type` | `temporal` | MoE variant: `vanilla` or `temporal` |
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace model to convert |
| `--num-experts` | `8` | Number of expert copies per MLP layer |
| `--top-k` | `2` | Experts active per token |
| `--ratio-loss-N` | `3` | N parameter for temporal ratio loss (target segment length) |
| `--ratio-loss-alpha` | `0.3` | Weight for temporal ratio loss |
| `--dataset-split` | `code` | Nemotron SFT split to use |
| `--num-samples` | `100000` | Number of samples to load |
| `--seq-len` | `256` | Sequence length |
| `--batch-size` | `16` | Per-device batch size |
| `--num-steps` | `100000` | Total training steps |
| `--lr` | `1e-4` | Learning rate |
| `--eval-every` | `50` | Steps between boundary visualizations (temporal only) |

Training logs (loss curves, boundary visualizations, expert assignment plots) are sent to [Weights & Biases](https://wandb.ai) under the `moe-chunking-poc` project.
