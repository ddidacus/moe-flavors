"""Analyze expert specialization from a trained MoE checkpoint.

Runs inference on held-out samples (10k default, ~2500 per split) and produces:
  1. Per-split top-4 experts by global activation count
  2. Expert co-activation graph (MoEfication-style)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin


TASK_SPLITS = ["stem", "chat", "math", "code"]


def messages_to_text(messages):
    return "\n".join(m["content"] for m in messages)


def load_moe_model(checkpoint_dir, device="cuda"):
    checkpoint_dir = Path(checkpoint_dir)
    meta = json.loads((checkpoint_dir / "moe_meta.json").read_text())

    print(f"Base model: {meta['base_model']}")
    print(f"MoE type: {meta['moe_type']}, config: {meta['moe_config']}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        meta["base_model"], torch_dtype=torch.bfloat16,
    )

    if meta["moe_type"] == "temporal":
        moe_config = TemporalMoEConfig(**meta["moe_config"])
        TemporalMoEMixin.apply(model, moe_config)
    else:
        moe_config = VanillaMoEConfig(**meta["moe_config"])
        VanillaMoEMixin.apply(model, moe_config)

    state_dict = {}
    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    bin_files = sorted(checkpoint_dir.glob("*.bin"))
    if safetensors_files:
        from safetensors.torch import load_file
        for f in safetensors_files:
            state_dict.update(load_file(f, device="cpu"))
    elif bin_files:
        for f in bin_files:
            state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No model weights in {checkpoint_dir}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, tokenizer


def load_split_samples(tokenizer, split_name, num_samples, max_sample_len,
                       offset=0):
    from datasets import load_dataset

    load_count = num_samples * 3
    selector = f"{split_name}[{offset}:{offset + load_count}]"
    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                      split=selector, streaming=False)

    filtered = []
    for row in tqdm(ds, desc=f"Filtering {split_name}", leave=False):
        text = messages_to_text(row["messages"])
        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if tok_len <= max_sample_len:
            filtered.append(text)
        if len(filtered) >= num_samples:
            break

    print(f"  {split_name}: {len(filtered)} samples "
          f"(target {num_samples}, offset {offset})")
    return filtered


@torch.no_grad()
def collect_activations(model, tokenizer, texts, max_sample_len,
                        batch_size, device):
    num_experts = model._moe_layers[0].config.num_experts
    expert_counts = torch.zeros(num_experts)
    coact = torch.zeros(num_experts, num_experts)
    total_tokens = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_sample_len,
                        padding=True, return_tensors="pt").to(device)
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        mask = enc["attention_mask"].bool().cpu()

        for layer in model._moe_layers:
            idx = layer._last_top_k_indices.cpu()
            valid = idx[mask].long()
            N, K = valid.shape
            total_tokens += N

            expert_counts += torch.bincount(
                valid.reshape(-1), minlength=num_experts,
            ).float()

            for i in range(K):
                for j in range(i + 1, K):
                    ei, ej = valid[:, i], valid[:, j]
                    coact.index_put_(
                        (ei, ej), torch.ones(N), accumulate=True,
                    )
                    coact.index_put_(
                        (ej, ei), torch.ones(N), accumulate=True,
                    )

    return expert_counts, coact, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert specialization from MoE checkpoint",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10_000,
                        help="Total samples across all splits")
    parser.add_argument("--max-sample-len", type=int, default=1024)
    parser.add_argument("--data-offset", type=int, default=50_000,
                        help="Offset into each Nemotron split "
                             "(to avoid overlap with training data)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=TASK_SPLITS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model, tokenizer = load_moe_model(args.checkpoint_dir, args.device)
    num_experts = model._moe_layers[0].config.num_experts
    per_split = args.num_samples // len(args.splits)

    print(f"\nCollecting activations: {args.num_samples} samples "
          f"({per_split} per split, offset={args.data_offset})")

    split_counts = {}
    global_coact = torch.zeros(num_experts, num_experts)
    global_tokens = 0

    for sn in args.splits:
        texts = load_split_samples(
            tokenizer, sn, per_split, args.max_sample_len, args.data_offset,
        )
        counts, coact, tokens = collect_activations(
            model, tokenizer, texts, args.max_sample_len,
            args.batch_size, args.device,
        )
        split_counts[sn] = counts
        global_coact += coact
        global_tokens += tokens

    if global_tokens > 0:
        global_coact /= global_tokens

    # ── Results ──

    print("\n" + "=" * 60)
    print("ANALYSIS 1: Top-4 experts per split by global activation count")
    print("=" * 60)
    for sn in args.splits:
        top4 = split_counts[sn].topk(4)
        print(f"\n  {sn}:")
        for rank, (idx, cnt) in enumerate(
            zip(top4.indices.tolist(), top4.values.tolist())
        ):
            print(f"    #{rank + 1}: Expert {idx} ({cnt:.0f} activations)")

    print("\n" + "=" * 60)
    print("ANALYSIS 2: Top-3 co-activated experts per expert")
    print("=" * 60)
    for e in range(num_experts):
        row = global_coact[e].clone()
        row[e] = 0
        top3 = row.topk(3)
        top3_str = ", ".join(
            f"E{idx}({val:.4f})" for idx, val in
            zip(top3.indices.tolist(), top3.values.tolist())
        )
        print(f"  Expert {e:2d}: {top3_str}")

    # ── Plots ──

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(args.output_dir or args.checkpoint_dir) / "specialization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: grouped bar chart of expert activation counts per split
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(num_experts)
    width = 0.8 / len(args.splits)
    colors = plt.cm.Set2(np.linspace(0, 1, len(args.splits)))

    for i, sn in enumerate(args.splits):
        ax.bar(x + i * width, split_counts[sn].numpy(), width,
               label=sn, color=colors[i])

    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Activation Count")
    ax.set_title(f"Expert Activation Counts per Split "
                 f"({args.num_samples} samples)")
    ax.set_xticks(x + width * (len(args.splits) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(num_experts)])
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "expert_counts_per_split.png", dpi=150)
    plt.close(fig)

    # Plot 2: co-activation heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(global_coact.numpy(), cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    ax.set_title(f"Expert Co-activation ({args.num_samples} samples)")
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_experts))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_dir / "coactivation_heatmap.png", dpi=150)
    plt.close(fig)

    # Plot 3: per-expert top-3 co-activated neighbor bar charts
    n_cols = 4
    n_rows = (num_experts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    for e in range(num_experts):
        row = global_coact[e].clone()
        row[e] = 0
        top3 = row.topk(3)
        axes[e].bar(
            [f"E{i}" for i in top3.indices.tolist()],
            top3.values.tolist(),
            color=plt.cm.Set2(np.linspace(0, 1, 3)),
        )
        axes[e].set_title(f"Expert {e}", fontsize=9)
        axes[e].tick_params(labelsize=7)
    for e in range(num_experts, len(axes)):
        axes[e].set_visible(False)
    plt.suptitle("Top-3 Co-activated Neighbors per Expert")
    plt.tight_layout()
    fig.savefig(out_dir / "coactivation_top3.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
