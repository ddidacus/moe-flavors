"""Analyze router gate weight matrices across MoE layers.

For each layer, extracts the gate weight matrix (num_experts x hidden_dim),
performs PCA to 2D, and computes pairwise cosine similarity between expert
weight vectors. Produces:
  - PCA scatter plots of expert weight vectors per layer (grid + individual)
  - Cosine similarity heatmaps per layer
  - Summary statistics (avg pairwise similarity per layer)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.extract_routing_vectors import load_moe_model


def extract_gate_weights(model):
    """Return list of (layer_idx, weight_tensor) for each MoE layer's gate."""
    weights = []
    for layer_idx, moe_layer in enumerate(model._moe_layers):
        gate = moe_layer.router.gate
        if hasattr(gate, "weight"):
            w = gate.weight.detach().float().cpu()
        elif isinstance(gate, torch.nn.Linear):
            w = gate.weight.detach().float().cpu()
        else:
            print(f"  Layer {layer_idx}: cannot extract gate weights, skipping")
            continue
        weights.append((layer_idx, w))
    return weights


def pca_2d(X):
    """PCA projection to 2D. X: (n, d) -> (n, 2)."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    proj = X_centered @ Vt[:2].T
    explained = S[:2] ** 2 / (S ** 2).sum()
    return proj.numpy(), explained.numpy()


def pairwise_cosine_similarity(X):
    """Return (n, n) pairwise cosine similarity matrix."""
    X_norm = F.normalize(X, dim=1)
    return (X_norm @ X_norm.T).numpy()


def plot_pca_grid(gate_weights, out_dir):
    """Grid of PCA scatter plots, one per layer."""
    num_layers = len(gate_weights)
    n_cols = min(4, num_layers)
    n_rows = (num_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    cmap = plt.cm.get_cmap("tab20")

    for i, (layer_idx, W) in enumerate(gate_weights):
        ax = axes[i]
        num_experts = W.shape[0]
        proj, explained = pca_2d(W)

        colors = [cmap(e / max(num_experts - 1, 1)) for e in range(num_experts)]
        for e in range(num_experts):
            ax.scatter(proj[e, 0], proj[e, 1], c=[colors[e]], s=80,
                       edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(f"E{e}", (proj[e, 0], proj[e, 1]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

        var_str = f"{explained[0]:.0%}+{explained[1]:.0%}"
        ax.set_title(f"Layer {layer_idx} ({var_str})", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("PCA of Router Gate Expert Weight Vectors", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "gate_pca_grid.png", dpi=150)
    plt.close(fig)


def plot_similarity_grid(gate_weights, out_dir):
    """Grid of cosine similarity heatmaps, one per layer."""
    num_layers = len(gate_weights)
    n_cols = min(4, num_layers)
    n_rows = (num_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (layer_idx, W) in enumerate(gate_weights):
        ax = axes[i]
        sim = pairwise_cosine_similarity(W)
        num_experts = W.shape[0]

        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(num_experts))
        ax.set_xticklabels([f"E{e}" for e in range(num_experts)], fontsize=7)
        ax.set_yticklabels([f"E{e}" for e in range(num_experts)], fontsize=7)

        for row in range(num_experts):
            for col in range(num_experts):
                ax.text(col, row, f"{sim[row, col]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(sim[row, col]) > 0.5 else "black")

    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Pairwise Cosine Similarity of Gate Expert Vectors", fontsize=13)
    fig.colorbar(im, ax=axes[:num_layers].tolist(), shrink=0.6, label="Cosine Similarity")
    plt.tight_layout()
    fig.savefig(out_dir / "gate_similarity_grid.png", dpi=150)
    plt.close(fig)


def plot_avg_similarity_by_layer(gate_weights, out_dir):
    """Line plot of average pairwise cosine similarity per layer."""
    layers = []
    avg_sims = []

    for layer_idx, W in gate_weights:
        sim = pairwise_cosine_similarity(W)
        num_experts = W.shape[0]
        mask = ~np.eye(num_experts, dtype=bool)
        avg_sims.append(sim[mask].mean())
        layers.append(layer_idx)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, avg_sims, marker="o", markersize=4, color="#E91E63")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg Pairwise Cosine Similarity")
    ax.set_title("Expert Gate Vector Similarity by Layer")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_dir / "gate_avg_similarity_by_layer.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="PCA and similarity analysis of router gate weight vectors",
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to MoE checkpoint directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <checkpoint-dir>/gate_analysis/)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "gate_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_dir} ...")
    model, tokenizer, moe_type = load_moe_model(str(checkpoint_dir), args.device)
    print(f"MoE type: {moe_type}")

    gate_weights = extract_gate_weights(model)
    print(f"Extracted gate weights from {len(gate_weights)} layers")

    if not gate_weights:
        print("No gate weights found, exiting.")
        return

    num_experts = gate_weights[0][1].shape[0]
    hidden_dim = gate_weights[0][1].shape[1]
    print(f"  {num_experts} experts, hidden_dim={hidden_dim}")

    print("\n--- Per-layer average pairwise cosine similarity ---")
    all_avg_sims = []
    for layer_idx, W in gate_weights:
        sim = pairwise_cosine_similarity(W)
        mask = ~np.eye(num_experts, dtype=bool)
        avg_sim = sim[mask].mean()
        min_sim = sim[mask].min()
        max_sim = sim[mask].max()
        all_avg_sims.append(avg_sim)
        print(f"  Layer {layer_idx:2d}: avg={avg_sim:+.4f}  min={min_sim:+.4f}  max={max_sim:+.4f}")

    global_avg = np.mean(all_avg_sims)
    print(f"\n  Global avg across layers: {global_avg:+.4f}")
    print(f"  (lower = more differentiated expert representations)")

    print("\nGenerating plots ...")
    plot_pca_grid(gate_weights, out_dir)
    plot_similarity_grid(gate_weights, out_dir)
    plot_avg_similarity_by_layer(gate_weights, out_dir)

    summary = {
        "moe_type": moe_type,
        "num_layers": len(gate_weights),
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "per_layer_avg_similarity": {idx: float(s) for (idx, _), s in zip(gate_weights, all_avg_sims)},
        "global_avg_similarity": float(global_avg),
    }
    torch.save(summary, out_dir / "gate_analysis_summary.pt")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
