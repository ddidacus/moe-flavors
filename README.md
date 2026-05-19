# MoE Playground

Proof-of-concept for **Mixture-of-Experts (MoE)** applied to pretrained dense language models. Supports two MoE variants:

- **Temporal (chunking):** A learned chunking router detects segment boundaries in the token sequence via cosine distance and forward-fills routing decisions across each segment, so tokens within the same chunk share the same experts.
- **Vanilla:** Standard per-token top-k gating with a load-balancing auxiliary loss.

## How it works

1. A dense pretrained LM (default: Qwen3-0.6B) is loaded.
2. Every FFN/MLP layer is replaced with an MoE layer containing N expert copies of the original FFN plus a router.
3. **Temporal mode:** The `ChunkingRouter` computes boundary probabilities between adjacent tokens using learned query/key projections and cosine distance. Tokens where `p_t >= threshold` start a new segment; routing decisions are forward-filled within each segment via `cummax`. A **ratio loss** encourages the router to produce meaningful segmentation and is folded into the model's forward pass for clean DDP compatibility.
4. **Vanilla mode:** A standard `Router` computes per-token top-k expert assignments with a Switch Transformer style load-balancing loss.

## Research questions

1. **Performance degradation.** Do we observe any performance degradation with temporally consistent routing? (perplexity, eval harness performance)
2. **Switch rates.** How do switch rates compare between vanilla MoE and temporally consistent MoE?
3. **Routing patterns.** Can we observe routing patterns more consistently with temporally consistent MoEs?

## Project structure

```
src/
  temporal_moe.py      # ChunkingRouter, ChunkingMoELayer, MoEMixin, MoEConfig
  vanilla_moe.py       # Baseline top-k MoE (no chunking) for comparison
scripts/
  moe_mixin_poc.py     # Training script (Nemotron SFT data, wandb logging, boundary eval)
  e2e_temporal.sh      # SLURM end-to-end: train temporal MoE + eval harness
  e2e_vanilla.sh       # SLURM end-to-end: train vanilla MoE + eval harness
  eval_harness.py      # Standalone evaluation of any saved checkpoint via lm-evaluation-harness
  eval_baseline.sh     # SLURM eval of the dense baseline (Qwen3-0.6B, no MoE)
  sweep_ratio_loss.sh  # Grid sweep over ratio_loss_N and ratio_loss_alpha
```

## Dataset

Training uses the [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) SFT dataset (gated, requires HF access). Messages are concatenated into a single text per sample for causal LM training. Available splits: `code`, `math`, `stem`, `chat`, and multilingual variants.

## Setup

```bash
uv sync
```

## Running

**End-to-end (SLURM) — train + eval:**

```bash
sbatch scripts/e2e_temporal.sh   # temporal MoE
sbatch scripts/e2e_vanilla.sh    # vanilla MoE
```

**Evaluate the dense baseline (no MoE):**

```bash
sbatch scripts/eval_baseline.sh
```

**Evaluate a saved checkpoint:**

```bash
python scripts/eval_harness.py --checkpoint-dir checkpoints/temporal_12345/step_30000
```

**Hyperparameter sweep (ratio loss):**

```bash
bash scripts/sweep_ratio_loss.sh
```

**Single GPU (manual):**

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
| `--ratio-loss-N` | `3` | N parameter(s) for temporal ratio loss — one per layer, or a single value broadcast to all |
| `--ratio-loss-alpha` | `0.3` | Weight for temporal ratio loss |
| `--entropy-threshold` | `0.1` | Floor for p_t entropy penalty (0 disables) |
| `--entropy-alpha` | `1.0` | Weight for entropy penalty |
| `--entropy-warmup-steps` | `0` | Steps before enabling the entropy penalty |
| `--dataset-splits` | `code` | Nemotron SFT splits (multiple allowed) |
| `--num-samples` | `100000` | Number of samples to load |
| `--seq-len` | `256` | Sequence length |
| `--batch-size` | `16` | Per-device batch size |
| `--num-steps` | `100000` | Total training steps |
| `--lr` | `1e-4` | Learning rate |
| `--log-every` | `10` | Steps between W&B metric logs |
| `--eval-every` | `1000` | Steps between perplexity, harness, and visualization evals |
| `--harness-tasks` | MMLU math subset | lm-eval-harness tasks to run during eval |
| `--harness-limit` | `8` | Samples per harness task (during training evals) |
| `--save-dir` | — | Directory for model checkpoints |
| `--save-every` | `0` | Checkpoint interval (0 = only at end) |
| `--wandb-run-name` | auto | Custom W&B run name |

Training logs (loss curves, boundary visualizations, expert assignment plots, switch rates) are sent to [Weights & Biases](https://wandb.ai) under the `moe-chunking-poc` project.
