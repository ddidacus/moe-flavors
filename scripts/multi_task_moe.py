import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin


TASK_SPLITS = ["stem", "chat", "math", "code"]
SYSTEM_PROMPT = "You are given problems across 4 disciplines. Please solve them.\n\n"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def messages_to_text(messages):
    return "\n".join(m["content"] for m in messages)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# ── Dataset ─────────────────────────────────────────────────


def load_and_filter_splits(tokenizer, splits, num_samples, max_sample_len,
                           test_frac=0.1, seed=42):
    from datasets import load_dataset

    per_split = num_samples // len(splits)
    split_data = {}

    for split_name in splits:
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                          split=f"{split_name}[:{per_split}]", streaming=False)
        filtered = []
        for row in tqdm(ds, desc=f"Filtering {split_name}", leave=False):
            text = messages_to_text(row["messages"])
            tok_len = len(tokenizer.encode(text, add_special_tokens=False))
            if tok_len <= max_sample_len:
                filtered.append(text)

        rng = random.Random(seed)
        rng.shuffle(filtered)
        n_test = max(int(len(filtered) * test_frac), 4)
        split_data[split_name] = {
            "train": filtered[n_test:],
            "test": filtered[:n_test],
        }
        print(f"  {split_name}: {len(filtered)}/{len(ds)} passed filter "
              f"(<= {max_sample_len} tok), train={len(filtered) - n_test}, test={n_test}")

    return split_data


class MultiTaskDataset(Dataset):
    def __init__(self, split_data, splits, tokenizer, seq_len, num_samples):
        self.split_pools = {s: split_data[s]["train"] for s in splits}
        self.splits = list(splits)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.length = num_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        order = list(self.splits)
        random.shuffle(order)

        parts = [SYSTEM_PROMPT]
        for split_name in order:
            pool = self.split_pools[split_name]
            parts.append(f"### Task: {split_name}\n{random.choice(pool)}\n\n")

        text = "".join(parts)
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.seq_len + 1,
            padding="max_length", return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids[:-1], input_ids[1:]


# ── Analysis ────────────────────────────────────────────────


def build_coactivation_matrix(all_top_k_indices, num_experts):
    coact = torch.zeros(num_experts, num_experts)
    total_tokens = 0

    for indices in all_top_k_indices:
        flat = indices.reshape(-1, indices.shape[-1]).long()
        N, K = flat.shape
        total_tokens += N
        for i in range(K):
            for j in range(i + 1, K):
                ei, ej = flat[:, i], flat[:, j]
                coact.index_put_((ei, ej), torch.ones(N), accumulate=True)
                coact.index_put_((ej, ei), torch.ones(N), accumulate=True)

    if total_tokens > 0:
        coact /= total_tokens
    return coact


@torch.no_grad()
def eval_expert_specialization(model, tokenizer, split_data, splits,
                                max_sample_len, step, num_per_split=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    device = next(model.parameters()).device
    num_experts = model._moe_layers[0].config.num_experts

    split_expert_counts = {}
    all_top_k_indices = []

    for split_name in splits:
        test_texts = split_data[split_name]["test"][:num_per_split]
        counts = torch.zeros(num_experts)

        for text in test_texts:
            enc = tokenizer(text, truncation=True, max_length=max_sample_len,
                            padding=False, return_tensors="pt").to(device)
            model(input_ids=enc["input_ids"])

            for layer in model._moe_layers:
                idx = layer._last_top_k_indices.cpu()
                all_top_k_indices.append(idx)
                counts += torch.bincount(
                    idx.reshape(-1).long(), minlength=num_experts,
                ).float()

        split_expert_counts[split_name] = counts

    # Plot 1: grouped bar chart of expert activations per split
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(num_experts)
    width = 0.8 / len(splits)
    colors = plt.cm.Set2(np.linspace(0, 1, len(splits)))

    for i, sn in enumerate(splits):
        ax.bar(x + i * width, split_expert_counts[sn].numpy(), width,
               label=sn, color=colors[i])

    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Activation Count")
    ax.set_title(f"Expert Activation Counts per Split (step {step})")
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(num_experts)])
    ax.legend()
    plt.tight_layout()
    wandb.log({"eval/expert_counts_per_split": wandb.Image(fig)}, step=step)
    plt.close(fig)

    print(f"\n--- Expert specialization (step {step}) ---")
    for sn in splits:
        top4 = split_expert_counts[sn].topk(min(4, num_experts))
        top4_str = ", ".join(
            f"E{idx}({cnt:.0f})" for idx, cnt in
            zip(top4.indices.tolist(), top4.values.tolist())
        )
        print(f"  {sn} top-4: {top4_str}")

    # Plot 2: co-activation heatmap
    coact = build_coactivation_matrix(all_top_k_indices, num_experts)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(coact.numpy(), cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    ax.set_title(f"Expert Co-activation (step {step})")
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_experts))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    wandb.log({"eval/coactivation_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)

    print("  Co-activation top-3:")
    for e in range(num_experts):
        row = coact[e].clone()
        row[e] = 0
        top3 = row.topk(min(3, num_experts - 1))
        top3_str = ", ".join(
            f"E{idx}({val:.4f})" for idx, val in
            zip(top3.indices.tolist(), top3.values.tolist())
        )
        print(f"    Expert {e:2d}: {top3_str}")


# ── Checkpointing & metrics ────────────────────────────────


def save_checkpoint(model, tokenizer, save_dir, step, args, moe_config):
    save_path = Path(save_dir) / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    meta = {
        "base_model": args.model,
        "moe_type": args.moe_type,
        "moe_config": asdict(moe_config),
        "step": step,
    }
    (save_path / "moe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[checkpoint] Saved to {save_path}")


@torch.no_grad()
def compute_switch_rate(model):
    rates = []
    for layer in model._moe_layers:
        indices = layer._last_top_k_indices
        sorted_idx = indices.sort(dim=-1).values
        changed = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).any(dim=-1)
        rates.append(changed.float().mean().item())
    return sum(rates) / len(rates)


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-task MoE Training")
    parser.add_argument("--moe-type", choices=["vanilla", "temporal"],
                        default="temporal")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ratio-loss-N", type=int, nargs="+", default=[3])
    parser.add_argument("--ratio-loss-alpha", type=float, default=0.3)
    parser.add_argument("--entropy-threshold", type=float, default=0.1)
    parser.add_argument("--entropy-alpha", type=float, default=1.0)
    parser.add_argument("--entropy-warmup-steps", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-sample-len", type=int, default=None,
                        help="Max token length per individual sample "
                             "(default: seq_len // num_splits - 32)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-splits", nargs="+", default=TASK_SPLITS,
                        choices=["code", "math", "stem", "chat",
                                 "multilingual_ja", "multilingual_de",
                                 "multilingual_it", "multilingual_es",
                                 "multilingual_fr"])
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-multi-task")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    if args.max_sample_len is None:
        args.max_sample_len = args.seq_len // len(args.dataset_splits) - 32

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

    wandb_kwargs = {}
    if args.wandb_run_name:
        wandb_kwargs["name"] = args.wandb_run_name
    accelerator.init_trackers(
        args.wandb_project, config=vars(args),
        init_kwargs={"wandb": wandb_kwargs},
    )
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    # ── Load model ──

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    dense_params = count_params(model)
    accelerator.print(f"Dense model parameters: {dense_params:,}")

    # ── Dense -> MoE ──

    if args.moe_type == "temporal":
        moe_config = TemporalMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            ratio_loss_N=args.ratio_loss_N,
            ratio_loss_alpha=args.ratio_loss_alpha,
            entropy_threshold=args.entropy_threshold,
            entropy_alpha=args.entropy_alpha,
        )
        TemporalMoEMixin.apply(model, moe_config)
    else:
        moe_config = VanillaMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
        )
        VanillaMoEMixin.apply(model, moe_config)
    moe_params = count_params(model)
    accelerator.print(f"MoE model parameters:   {moe_params:,}  "
                      f"({moe_params / dense_params:.1f}x dense)")

    # ── Load & filter dataset ──

    accelerator.print(f"Loading & filtering Nemotron SFT "
                      f"({args.dataset_splits}, {args.num_samples} samples, "
                      f"max_sample_len={args.max_sample_len}) ...")
    split_data = load_and_filter_splits(
        tokenizer, args.dataset_splits, args.num_samples,
        args.max_sample_len, seed=args.seed,
    )

    train_dataset = MultiTaskDataset(
        split_data, args.dataset_splits, tokenizer,
        args.seq_len, args.num_samples,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    # ── Prepare training ──

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        wandb.log({"dense_params": dense_params, "moe_params": moe_params}, step=0)

    model.train()
    raw_model = _unwrap(model)
    num_moe_layers = len(raw_model._moe_layers)
    ratio_loss_alpha = (raw_model._moe_layers[0]._ratio_loss_alpha
                        if args.moe_type == "temporal" else None)
    entropy_alpha_target = (args.entropy_alpha
                            if args.moe_type == "temporal" else 0.0)

    # ── Training loop ──

    step = 0
    for epoch in range(100):
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False,
                    disable=not accelerator.is_main_process)
        for x, y in pbar:
            if args.moe_type == "temporal":
                ea = (entropy_alpha_target
                      if step >= args.entropy_warmup_steps else 0.0)
                for m in raw_model._moe_layers:
                    m._entropy_alpha = ea

            outputs = model(input_ids=x, labels=y)
            aux_loss = raw_model.get_moe_loss()

            if args.moe_type == "temporal":
                total_loss = outputs.loss
                lm_loss = (total_loss - aux_loss).detach()
            else:
                lm_loss = outputs.loss
                total_loss = lm_loss + aux_loss
                lm_loss = lm_loss.detach()
            aux_loss = aux_loss.detach()

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if accelerator.is_main_process:
                pbar.set_postfix(lm=f"{lm_loss.item():.4f}",
                                 aux=f"{aux_loss.item():.4f}", step=step)
                if step % args.log_every == 0:
                    metrics = {
                        "train/lm_loss": lm_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/epoch": epoch,
                        "train/switch_rate": compute_switch_rate(raw_model),
                    }
                    if ratio_loss_alpha is not None:
                        metrics["train/ratio_loss_per_layer_unscaled"] = (
                            aux_loss.item() / (num_moe_layers * ratio_loss_alpha)
                        )
                        avg_G = sum(
                            m._last_G.item() for m in raw_model._moe_layers
                        ) / num_moe_layers
                        avg_F = sum(
                            m._last_F.item() for m in raw_model._moe_layers
                        ) / num_moe_layers
                        avg_ent = sum(
                            m._last_pt_entropy.item()
                            for m in raw_model._moe_layers
                        ) / num_moe_layers
                        metrics["train/G_value"] = avg_G
                        metrics["train/F_value"] = avg_F
                        metrics["train/pt_entropy"] = avg_ent
                    wandb.log(metrics, step=step)

                if step % args.eval_every == 0:
                    raw_model.eval()
                    eval_expert_specialization(
                        raw_model, tokenizer, split_data,
                        args.dataset_splits, args.max_sample_len, step,
                    )
                    raw_model.train()

                if (args.save_dir and args.save_every > 0
                        and step % args.save_every == 0):
                    save_checkpoint(raw_model, tokenizer, args.save_dir,
                                    step, args, moe_config)

            if step >= args.num_steps:
                break
        if step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if accelerator.is_main_process and args.save_dir:
        save_checkpoint(raw_model, tokenizer, args.save_dir,
                        step, args, moe_config)

    if accelerator.is_main_process:
        raw_model.eval()
        eval_expert_specialization(
            raw_model, tokenizer, split_data,
            args.dataset_splits, args.max_sample_len, step,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
