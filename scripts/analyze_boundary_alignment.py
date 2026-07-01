"""Analyze alignment of predicted segmentation boundaries with ground truth domain transitions.

Loads a temporal MoE checkpoint, runs inference on the test set, and compares
the learned boundaries (from the ChunkingRouter's termination function) against
ground truth domain transitions in the multi-domain conversation data.

Produces:
  1. Per-sample visualizations (domain strip + pt heatmap + pt curves)
  2. IoU, Precision, Recall, F1 per layer (averaged across samples)
  3. Predicted boundary count distribution per layer
"""

import argparse
import os
import random
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

from scripts.extract_routing_vectors import (
    load_moe_model,
    build_domain_char_spans,
    tokens_to_domain_mask,
    messages_to_text,
    DOMAINS,
    DOMAIN_TO_IDX,
)

DOMAIN_COLORS = {"chat": "#2196F3", "code": "#4CAF50", "math": "#FF9800"}


def get_ground_truth_boundaries(domain_mask):
    """Token positions where the domain index transitions (excluding pos 0).

    Skips over -1 (unmapped) tokens so that a gap between domains still
    produces a boundary at the first token of the new domain.
    """
    valid_idx = (domain_mask >= 0).nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() < 2:
        return torch.tensor([], dtype=torch.long)
    valid_domains = domain_mask[valid_idx]
    changed = valid_domains[1:] != valid_domains[:-1]
    return valid_idx[1:][changed]


def compute_boundary_metrics(pred_bt, gt_positions, seq_len, tolerance=5):
    """Boundary alignment metrics for one sample + one layer.

    Returns dict with iou, precision, recall, f1.
    """
    # exclude position 0 (always forced boundary)
    pred_positions = pred_bt[1:seq_len].nonzero(as_tuple=False).squeeze(-1) + 1

    if pred_positions.dim() == 0:
        pred_positions = pred_positions.unsqueeze(0)
    if gt_positions.dim() == 0:
        gt_positions = gt_positions.unsqueeze(0)

    # window masks
    gt_window = torch.zeros(seq_len, dtype=torch.bool)
    for pos in gt_positions:
        lo = max(0, pos.item() - tolerance)
        hi = min(seq_len, pos.item() + tolerance + 1)
        gt_window[lo:hi] = True

    pred_window = torch.zeros(seq_len, dtype=torch.bool)
    for pos in pred_positions:
        lo = max(0, pos.item() - tolerance)
        hi = min(seq_len, pos.item() + tolerance + 1)
        pred_window[lo:hi] = True

    intersection = (gt_window & pred_window).sum().float()
    union = (gt_window | pred_window).sum().float()
    iou = (intersection / union).item() if union > 0 else 0.0

    if len(pred_positions) > 0:
        precision = gt_window[pred_positions].float().mean().item()
    else:
        precision = 0.0

    if len(gt_positions) > 0:
        recall = pred_window[gt_positions].float().mean().item()
    else:
        recall = 1.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def collect_boundary_data(model, tokenizer, dataset_rows, max_len, batch_size,
                          device):
    """Run inference and collect predicted + ground truth boundaries."""
    num_layers = len(model._moe_layers)
    num_samples = len(dataset_rows)

    texts = [messages_to_text(row["messages"]) for row in dataset_rows]
    all_messages = [row["messages"] for row in dataset_rows]

    all_pred_bt = []
    all_pred_pt = []
    all_domain_masks = []
    all_attention_masks = []
    all_gt_boundaries = []

    for start in tqdm(range(0, num_samples, batch_size), desc="Collecting boundaries"):
        end = min(start + batch_size, num_samples)
        batch_texts = texts[start:end]
        batch_messages = all_messages[start:end]
        B_actual = end - start

        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping").tolist()
        enc = enc.to(device)

        # ground truth domain masks
        batch_domain_masks = []
        batch_gt_boundaries = []
        for b in range(B_actual):
            spans = build_domain_char_spans(batch_messages[b])
            dm = tokens_to_domain_mask(offset_mapping[b], spans, max_len)
            batch_domain_masks.append(dm)
            batch_gt_boundaries.append(get_ground_truth_boundaries(dm))

        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        attn_mask = enc["attention_mask"].cpu()

        # collect per-layer boundary data
        sample_bt = torch.zeros(B_actual, num_layers, max_len, dtype=torch.bool)
        sample_pt = torch.zeros(B_actual, num_layers, max_len)

        for layer_idx in range(num_layers):
            moe_layer = model._moe_layers[layer_idx]
            bt = moe_layer._last_bt[:B_actual].cpu()
            pt = moe_layer._last_pt[:B_actual].float().cpu()
            sample_bt[:, layer_idx, :] = bt
            sample_pt[:, layer_idx, :] = pt

        all_pred_bt.append(sample_bt)
        all_pred_pt.append(sample_pt)
        all_domain_masks.append(torch.stack(batch_domain_masks))
        all_attention_masks.append(attn_mask)
        all_gt_boundaries.extend(batch_gt_boundaries)

    return {
        "num_layers": num_layers,
        "num_samples": num_samples,
        "max_len": max_len,
        "pred_bt": torch.cat(all_pred_bt, dim=0),
        "pred_pt": torch.cat(all_pred_pt, dim=0),
        "domain_masks": torch.cat(all_domain_masks, dim=0),
        "attention_masks": torch.cat(all_attention_masks, dim=0),
        "gt_boundaries": all_gt_boundaries,
    }


def compute_metrics_per_layer(boundary_data, tolerance=5):
    """Average boundary alignment metrics per layer across all samples."""
    num_layers = boundary_data["num_layers"]
    num_samples = boundary_data["num_samples"]

    metrics = {
        l: {"iou": [], "precision": [], "recall": [], "f1": []}
        for l in range(num_layers)
    }

    for s in range(num_samples):
        seq_len = boundary_data["attention_masks"][s].sum().item()
        gt_pos = boundary_data["gt_boundaries"][s]
        if len(gt_pos) == 0:
            continue
        for l in range(num_layers):
            pred_bt = boundary_data["pred_bt"][s, l]
            m = compute_boundary_metrics(pred_bt, gt_pos, seq_len, tolerance)
            for k, v in m.items():
                metrics[l][k].append(v)

    avg_metrics = {}
    for l in range(num_layers):
        avg_metrics[l] = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in metrics[l].items()
        }
    return avg_metrics


def _pick_layers(num_layers, count=3):
    if num_layers <= count:
        return list(range(num_layers))
    step = (num_layers - 1) / (count - 1)
    return [int(round(i * step)) for i in range(count)]


def plot_sample_boundary_alignment(sample_idx, domain_mask, pred_bt, pred_pt,
                                   gt_boundaries, seq_len, num_layers, out_dir):
    """Multi-panel visualization for one sample."""
    layers_to_show = _pick_layers(num_layers)
    num_panels = 1 + 1 + len(layers_to_show)
    height_ratios = [0.4, 2.0] + [1.0] * len(layers_to_show)

    fig, axes = plt.subplots(
        num_panels, 1,
        figsize=(14, 2 + 2.0 + 2.5 * len(layers_to_show)),
        gridspec_kw={"height_ratios": height_ratios},
    )

    gt_pos_list = gt_boundaries.tolist() if len(gt_boundaries) > 0 else []

    # panel 1: domain strip
    ax_strip = axes[0]
    strip = np.full((1, seq_len, 3), 0.85)
    for t in range(seq_len):
        d = domain_mask[t].item()
        if d >= 0:
            strip[0, t] = matplotlib.colors.to_rgb(DOMAIN_COLORS[DOMAINS[d]])
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
    for bp in gt_pos_list:
        ax_strip.axvline(x=bp, color="red", alpha=0.8, linewidth=1.5, linestyle="--")
    ax_strip.set_yticks([])
    ax_strip.set_title(f"Sample {sample_idx} — Domain Segmentation (seq_len={seq_len})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=DOMAIN_COLORS[d]) for d in DOMAINS
    ]
    ax_strip.legend(handles, DOMAINS, loc="upper right", ncol=len(DOMAINS), fontsize=8)

    # panel 2: pt heatmap (layers x tokens)
    ax_hm = axes[1]
    pt_img = pred_pt[:, :seq_len].numpy()
    im = ax_hm.imshow(
        pt_img, aspect="auto", cmap="inferno", interpolation="nearest",
        vmin=0, vmax=1,
    )
    for bp in gt_pos_list:
        ax_hm.axvline(x=bp, color="red", alpha=0.8, linewidth=1.2, linestyle="--")
    ax_hm.set_ylabel("Layer")
    ax_hm.set_title("Termination probability p_t per layer")
    plt.colorbar(im, ax=ax_hm, shrink=0.6, label="p_t")

    # panels 3+: pt curves for representative layers
    for panel_idx, layer_idx in enumerate(layers_to_show):
        ax = axes[2 + panel_idx]
        pt_vals = pred_pt[layer_idx, :seq_len].numpy()
        bt_vals = pred_bt[layer_idx, :seq_len]

        ax.plot(range(seq_len), pt_vals, color="black", linewidth=0.6, alpha=0.8)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

        # predicted boundaries
        pred_pos = bt_vals[1:].nonzero(as_tuple=False).squeeze(-1) + 1
        if pred_pos.dim() == 0:
            pred_pos = pred_pos.unsqueeze(0)
        for bp in pred_pos.tolist():
            ax.axvline(x=bp, color="#4CAF50", alpha=0.3, linewidth=0.6)

        # GT boundaries
        for bp in gt_pos_list:
            ax.axvline(x=bp, color="red", alpha=0.8, linewidth=1.2, linestyle="--")

        n_pred = len(pred_pos)
        ax.set_ylabel("p_t")
        ax.set_title(f"Layer {layer_idx} ({n_pred} predicted boundaries)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if panel_idx == len(layers_to_show) - 1:
            ax.set_xlabel("Token position")

    plt.tight_layout()
    viz_dir = out_dir / "viz_samples"
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(viz_dir / f"boundary_sample_{sample_idx}.png", dpi=120)
    plt.close(fig)


def plot_metrics_by_layer(avg_metrics, tolerance, out_dir):
    """Line chart of IoU, Precision, Recall, F1 per layer."""
    num_layers = len(avg_metrics)
    layers = list(range(num_layers))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, [avg_metrics[l]["iou"] for l in layers],
            marker="o", markersize=3, label="IoU", color="#E91E63")
    ax.plot(layers, [avg_metrics[l]["precision"] for l in layers],
            marker="s", markersize=3, label="Precision", color="#2196F3")
    ax.plot(layers, [avg_metrics[l]["recall"] for l in layers],
            marker="^", markersize=3, label="Recall", color="#4CAF50")
    ax.plot(layers, [avg_metrics[l]["f1"] for l in layers],
            marker="d", markersize=3, label="F1", color="#FF9800")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title(f"Boundary Alignment Metrics by Layer (tolerance=±{tolerance})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "boundary_metrics_by_layer.png", dpi=150)
    plt.close(fig)


def plot_boundary_count_by_layer(boundary_data, out_dir):
    """Bar chart of avg predicted boundary count per layer."""
    num_layers = boundary_data["num_layers"]
    num_samples = boundary_data["num_samples"]

    avg_counts = []
    for l in range(num_layers):
        counts = []
        for s in range(num_samples):
            seq_len = boundary_data["attention_masks"][s].sum().item()
            # exclude position 0
            counts.append(
                boundary_data["pred_bt"][s, l, 1:seq_len].sum().item()
            )
        avg_counts.append(sum(counts) / len(counts) if counts else 0)

    # average GT boundary count
    gt_counts = [len(boundary_data["gt_boundaries"][s]) for s in range(num_samples)]
    avg_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_layers), avg_counts, color="#7E57C2", alpha=0.8)
    ax.axhline(y=avg_gt, color="red", linestyle="--", linewidth=1.5,
               label=f"GT boundaries (avg={avg_gt:.1f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg predicted boundaries per sample")
    ax.set_title("Predicted Boundary Count by Layer (excl. pos 0)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "boundary_count_by_layer.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze alignment of predicted segmentation boundaries "
                    "with ground truth domain transitions",
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to temporal MoE checkpoint directory")
    parser.add_argument("--dataset-name", default="ddidacus/nemotron-moe-exam",
                        help="HF Hub dataset name")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Total test samples to evaluate")
    parser.add_argument("--num-viz-samples", type=int, default=10,
                        help="Number of per-sample visualizations to generate")
    parser.add_argument("--tolerance", type=int, default=5,
                        help="Boundary matching tolerance (±k tokens)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <checkpoint-dir>/boundary_analysis/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else checkpoint_dir / "boundary_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_dir} ...")
    model, tokenizer, moe_type = load_moe_model(str(checkpoint_dir), args.device)
    print(f"MoE type: {moe_type}")
    num_layers = len(model._moe_layers)
    print(f"Layers: {num_layers}")

    if moe_type != "temporal":
        print("WARNING: model is not temporal MoE — _last_bt may not exist")

    print(f"Loading dataset {args.dataset_name} ...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset_name)["test"]
    num_samples = min(args.num_samples, len(ds))
    dataset_rows = list(ds.select(range(num_samples)))
    print(f"Using {num_samples} test samples")

    print("Running inference ...")
    boundary_data = collect_boundary_data(
        model, tokenizer, dataset_rows, args.max_len,
        args.batch_size, args.device,
    )

    # metrics
    print(f"\nComputing metrics (tolerance=±{args.tolerance}) ...")
    avg_metrics = compute_metrics_per_layer(boundary_data, args.tolerance)

    print(f"\n{'Layer':>6}  {'IoU':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
    print("-" * 38)
    all_iou, all_f1 = [], []
    for l in range(num_layers):
        m = avg_metrics[l]
        print(f"{l:>6d}  {m['iou']:>6.4f}  {m['precision']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}")
        all_iou.append(m["iou"])
        all_f1.append(m["f1"])

    print(f"\nGlobal avg IoU:  {sum(all_iou)/len(all_iou):.4f}")
    print(f"Global avg F1:   {sum(all_f1)/len(all_f1):.4f}")

    # save metrics
    torch.save({
        "avg_metrics": avg_metrics,
        "tolerance": args.tolerance,
        "num_samples": num_samples,
        "num_layers": num_layers,
        "checkpoint_dir": str(checkpoint_dir),
    }, out_dir / "boundary_analysis_results.pt")

    # plots
    print("\nGenerating plots ...")
    plot_metrics_by_layer(avg_metrics, args.tolerance, out_dir)
    plot_boundary_count_by_layer(boundary_data, out_dir)

    # per-sample visualizations
    num_viz = min(args.num_viz_samples, num_samples)
    print(f"Generating {num_viz} per-sample visualizations ...")
    for s in tqdm(range(num_viz), desc="Viz samples"):
        seq_len = boundary_data["attention_masks"][s].sum().item()
        plot_sample_boundary_alignment(
            sample_idx=s,
            domain_mask=boundary_data["domain_masks"][s, :seq_len],
            pred_bt=boundary_data["pred_bt"][s],
            pred_pt=boundary_data["pred_pt"][s],
            gt_boundaries=boundary_data["gt_boundaries"][s],
            seq_len=seq_len,
            num_layers=num_layers,
            out_dir=out_dir,
        )

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
