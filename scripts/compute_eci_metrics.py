"""Compute Expert Contribution Index (ECI) metrics from extracted routing data.

Loads the output of `extract_routing_vectors.py --mode eci` and produces:
  1. Global ECI: E_{x in D_tau}[G^(l)(x)_i] per (layer, domain, expert)
  2. ECI heatmaps, specialization plots, and per-sample visualizations
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_global_eci(eci_data, domain_token_counts):
    """Token-count-weighted ECI across all samples.

    Args:
        eci_data: (S, D, L, E) per-sample per-domain avg routing probs
        domain_token_counts: (S, D) token counts per domain per sample

    Returns:
        (L, D, E) global ECI tensor
    """
    weights = domain_token_counts.float()
    weights_sum = weights.sum(dim=0, keepdim=True).clamp(min=1)
    weights_norm = weights / weights_sum  # (S, D)

    weighted = eci_data * weights_norm.unsqueeze(-1).unsqueeze(-1)  # (S, D, L, E)
    eci = weighted.sum(dim=0)  # (D, L, E)
    return eci.permute(1, 0, 2)  # (L, D, E)


def plot_eci_heatmaps(eci, domain_names, out_dir):
    """One heatmap per domain: layers x experts."""
    num_layers, num_domains, num_experts = eci.shape
    fig, axes = plt.subplots(1, num_domains, figsize=(5 * num_domains, max(6, num_layers * 0.3)))

    for d in range(num_domains):
        ax = axes[d] if num_domains > 1 else axes
        data = eci[:, d, :].numpy()
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        ax.set_title(f"ECI — {domain_names[d]}")
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 10)))
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "eci_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_top_experts(eci, domain_names, out_dir):
    """Categorical heatmap: which expert has highest ECI per (layer, domain)."""
    num_layers, num_domains, num_experts = eci.shape
    top = eci.argmax(dim=-1).numpy()  # (L, D)

    fig, ax = plt.subplots(figsize=(max(4, num_domains * 1.5), max(6, num_layers * 0.3)))
    cmap = plt.cm.get_cmap("tab20", num_experts)
    im = ax.imshow(top, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=num_experts - 1)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Layer")
    ax.set_title("Top Expert per Domain per Layer")
    ax.set_xticks(range(num_domains))
    ax.set_xticklabels(domain_names)
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 10)))
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_ticks(range(num_experts))
    cbar.set_ticklabels([f"E{i}" for i in range(num_experts)])

    plt.tight_layout()
    fig.savefig(out_dir / "eci_top_experts.png", dpi=150)
    plt.close(fig)


def compute_jaccard_similarity(eci, domain_names, out_dir, top_k=None):
    """Jaccard similarity of top-k expert sets between domain pairs, per layer.

    Jaccard_l(t1, t2) = |E_l,t1 ∩ E_l,t2| / |E_l,t1 ∪ E_l,t2|
    (Equation 13 from arXiv:2601.03425)
    """
    num_layers, num_domains, num_experts = eci.shape
    if top_k is None:
        top_k = max(1, num_experts // 4)

    pairs = [(i, j) for i in range(num_domains) for j in range(i + 1, num_domains)]
    jaccard = torch.zeros(num_layers, len(pairs))

    for l in range(num_layers):
        top_sets = []
        for d in range(num_domains):
            topk_idx = eci[l, d].topk(min(top_k, num_experts)).indices
            top_sets.append(set(topk_idx.tolist()))
        for p, (i, j) in enumerate(pairs):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccard[l, p] = inter / union if union > 0 else 0.0

    pair_names = [f"{domain_names[i]} vs {domain_names[j]}" for i, j in pairs]

    fig, ax = plt.subplots(figsize=(10, 4))
    for p, name in enumerate(pair_names):
        ax.plot(range(num_layers), jaccard[:, p].numpy(), marker="o", markersize=3, label=name)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title(f"Top-{top_k} Expert Overlap Between Domains (per Layer)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "jaccard_similarity_by_layer.png", dpi=150)
    plt.close(fig)

    print(f"\nJaccard similarity (top-{top_k}, avg across layers):")
    for p, name in enumerate(pair_names):
        print(f"  {name}: {jaccard[:, p].mean().item():.4f}")

    return jaccard, pair_names


def compute_gini_coefficient(eci, domain_names, out_dir):
    """Gini coefficient of expert contributions per layer.

    Gini(c_bar^l) = sum_i sum_j |c_bar_i - c_bar_j| / (2 * E * sum_i c_bar_i)
    (Equation 14 from arXiv:2601.03425)
    """
    num_layers, num_domains, num_experts = eci.shape
    avg_eci = eci.mean(dim=1)  # (L, E) — average ECI across domains

    gini = torch.zeros(num_layers)
    for l in range(num_layers):
        c = avg_eci[l]
        diffs = (c.unsqueeze(0) - c.unsqueeze(1)).abs().sum()
        total = c.sum()
        gini[l] = diffs / (2 * num_experts * total) if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(num_layers), gini.numpy(), marker="o", markersize=3, color="#E91E63")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Expert Contribution Inequality by Layer")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "gini_coefficient_by_layer.png", dpi=150)
    plt.close(fig)

    print(f"\nGini coefficient (avg across layers): {gini.mean().item():.4f}")

    return gini


def compute_entropy_by_domain(eci, domain_names, out_dir):
    """Average entropy of expert routing distribution per domain per layer."""
    num_layers, num_domains, num_experts = eci.shape
    eci_clamped = eci.clamp(min=1e-12)
    entropy = -(eci_clamped * eci_clamped.log()).sum(dim=-1)  # (L, D)

    fig, ax = plt.subplots(figsize=(10, 4))
    domain_colors = {"chat": "#2196F3", "code": "#4CAF50", "math": "#FF9800"}
    for d, name in enumerate(domain_names):
        color = domain_colors.get(name, None)
        ax.plot(range(num_layers), entropy[:, d].numpy(),
                marker="o", markersize=3, label=name, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Avg Expert Distribution Entropy by Domain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.log(num_experts), color="gray", linestyle="--", alpha=0.5,
               label=f"max entropy (log {num_experts})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "entropy_by_domain.png", dpi=150)
    plt.close(fig)

    print("\nEntropy by domain (avg across layers):")
    for d, name in enumerate(domain_names):
        print(f"  {name}: {entropy[:, d].mean().item():.4f} nats")

    return entropy


def _pick_layers(num_layers, count=3):
    if num_layers <= count:
        return list(range(num_layers))
    step = (num_layers - 1) / (count - 1)
    return [int(round(i * step)) for i in range(count)]


def plot_viz_samples(data, out_dir, num_plots=20, seed=42):
    """Per-sample plots showing expert distributions within each domain segment."""
    viz_probs = data["viz_routing_probs"]       # (V, max_len, L, E)
    viz_masks = data["viz_domain_masks"]         # (V, max_len)
    viz_attn = data["viz_attention_masks"]       # (V, max_len)
    domain_names = data["meta"]["domain_names"]
    num_viz = viz_probs.shape[0]
    num_experts = data["meta"]["num_experts"]
    num_layers = data["meta"]["num_layers"]

    actual_plots = min(num_plots, num_viz)
    rng = random.Random(seed)
    indices = list(range(num_viz))
    rng.shuffle(indices)
    indices = sorted(indices[:actual_plots])

    layers_to_show = _pick_layers(num_layers)
    domain_colors = {"chat": "#2196F3", "code": "#4CAF50", "math": "#FF9800"}
    viz_dir = out_dir / "viz_samples"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in tqdm(indices, desc="Plotting viz samples"):
        attn = viz_attn[sample_idx]
        seq_len = attn.sum().item()
        if seq_len == 0:
            continue

        domain_mask = viz_masks[sample_idx, :seq_len]
        probs = viz_probs[sample_idx, :seq_len]  # (seq_len, L, E)

        num_panels = 1 + len(layers_to_show)
        fig, axes = plt.subplots(num_panels, 1,
                                 figsize=(14, 2 + 2.5 * len(layers_to_show)),
                                 gridspec_kw={"height_ratios": [0.5] + [1] * len(layers_to_show)})

        ax_strip = axes[0]
        strip = np.zeros((1, seq_len, 3))
        for t in range(seq_len):
            d = domain_mask[t].item()
            if d >= 0:
                color = matplotlib.colors.to_rgb(domain_colors[domain_names[d]])
                strip[0, t] = color
            else:
                strip[0, t] = (0.85, 0.85, 0.85)
        ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
        ax_strip.set_yticks([])
        ax_strip.set_title(f"Sample {sample_idx} — Domain Segmentation")

        handles = [plt.Rectangle((0, 0), 1, 1, fc=domain_colors[d]) for d in domain_names]
        ax_strip.legend(handles, domain_names, loc="upper right", ncol=len(domain_names),
                        fontsize=8)

        for panel_idx, layer_idx in enumerate(layers_to_show):
            ax = axes[1 + panel_idx]
            layer_probs = probs[:, layer_idx, :]  # (seq_len, E)

            x = np.arange(num_experts)
            width = 0.8 / len(domain_names)
            for d_idx, d_name in enumerate(domain_names):
                d_mask = domain_mask == d_idx
                if d_mask.sum() == 0:
                    continue
                avg_probs = layer_probs[d_mask].mean(dim=0).numpy()
                ax.bar(x + d_idx * width, avg_probs, width,
                       label=d_name, color=domain_colors[d_name], alpha=0.8)

            ax.set_xlabel("Expert")
            ax.set_ylabel("Avg Routing Prob")
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xticks(x + width * (len(domain_names) - 1) / 2)
            ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
            if panel_idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(viz_dir / f"viz_sample_{sample_idx}.png", dpi=120)
        plt.close(fig)

    return indices


def plot_avg_routing_across_samples(data, out_dir, indices, seed=42):
    """One plot per domain: average routing probability per expert across selected viz samples, for every layer."""
    viz_probs = data["viz_routing_probs"]       # (V, max_len, L, E)
    viz_masks = data["viz_domain_masks"]         # (V, max_len)
    viz_attn = data["viz_attention_masks"]       # (V, max_len)
    domain_names = data["meta"]["domain_names"]
    num_experts = data["meta"]["num_experts"]
    num_layers = data["meta"]["num_layers"]
    num_domains = len(domain_names)

    avg_probs = torch.zeros(num_layers, num_domains, num_experts)
    domain_counts = torch.zeros(num_layers, num_domains)

    for sample_idx in indices:
        attn = viz_attn[sample_idx]
        seq_len = attn.sum().item()
        if seq_len == 0:
            continue
        domain_mask = viz_masks[sample_idx, :seq_len]
        probs = viz_probs[sample_idx, :seq_len]  # (seq_len, L, E)

        for d_idx in range(num_domains):
            d_mask = domain_mask == d_idx
            count = d_mask.sum().item()
            if count == 0:
                continue
            for l_idx in range(num_layers):
                avg_probs[l_idx, d_idx] += probs[d_mask, l_idx, :].mean(dim=0)
                domain_counts[l_idx, d_idx] += 1

    domain_counts = domain_counts.clamp(min=1)
    avg_probs /= domain_counts.unsqueeze(-1)

    domain_colors = {"chat": "#2196F3", "code": "#4CAF50", "math": "#FF9800"}
    n_cols = 4
    n_rows = (num_layers + n_cols - 1) // n_cols
    x = np.arange(num_experts)

    for d_idx, d_name in enumerate(domain_names):
        color = domain_colors.get(d_name, None)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows))
        axes = np.array(axes).flatten()

        for l_idx in range(num_layers):
            ax = axes[l_idx]
            ax.bar(x, avg_probs[l_idx, d_idx].numpy(), 0.8, color=color, alpha=0.8)
            ax.set_title(f"Layer {l_idx}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"E{i}" for i in range(num_experts)], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

        for idx in range(num_layers, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Avg Routing Prob — {d_name} (across {len(indices)} samples)", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / f"avg_routing_{d_name}_all_layers.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute ECI metrics from extracted routing data",
    )
    parser.add_argument("--input", required=True,
                        help="Path to eci_routing_data.pt")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: sibling eci_results/)")
    parser.add_argument("--num-viz-plots", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    meta = data["meta"]
    print(f"  {meta['num_samples']} samples, {meta['num_layers']} layers, "
          f"{meta['num_experts']} experts, domains={meta['domain_names']}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.input).parent / "eci_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    eci = compute_global_eci(data["eci_data"], data["domain_token_counts"])
    print(f"\nGlobal ECI shape: {eci.shape}  (layers, domains, experts)")

    for d, name in enumerate(meta["domain_names"]):
        top = eci[:, d, :].mean(dim=0).topk(min(4, meta["num_experts"]))
        top_str = ", ".join(
            f"E{i}({v:.4f})" for i, v in
            zip(top.indices.tolist(), top.values.tolist())
        )
        print(f"  {name} top experts (avg across layers): {top_str}")

    entropy = compute_entropy_by_domain(eci, meta["domain_names"], out_dir)

    jaccard, jaccard_pair_names = compute_jaccard_similarity(
        eci, meta["domain_names"], out_dir,
    )
    gini = compute_gini_coefficient(eci, meta["domain_names"], out_dir)

    torch.save({
        "global_eci": eci,
        "entropy_by_domain": entropy,
        "jaccard_similarity": jaccard,
        "jaccard_pair_names": jaccard_pair_names,
        "gini_coefficient": gini,
        "domain_names": meta["domain_names"],
        "num_layers": meta["num_layers"],
        "num_experts": meta["num_experts"],
        "per_sample_eci": data["eci_data"],
        "domain_token_counts": data["domain_token_counts"],
    }, out_dir / "eci_metrics.pt")
    print(f"\nSaved eci_metrics.pt to {out_dir}")

    print("Generating plots ...")
    plot_eci_heatmaps(eci, meta["domain_names"], out_dir)
    plot_top_experts(eci, meta["domain_names"], out_dir)
    viz_indices = plot_viz_samples(data, out_dir, num_plots=args.num_viz_plots, seed=args.seed)
    plot_avg_routing_across_samples(data, out_dir, viz_indices, seed=args.seed)

    print(f"All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
