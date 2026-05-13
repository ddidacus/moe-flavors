import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin


# ====================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_nemotron_loader(tokenizer, seq_len: int, batch_size: int, num_samples: int,
                        splits: list[str] = ("code",)):
    from datasets import load_dataset, concatenate_datasets
    from torch.utils.data import DataLoader, Dataset

    per_split = num_samples // len(splits)
    all_ds = []
    for split in splits:
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                          split=f"{split}[:{per_split}]", streaming=False)
        all_ds.append(ds)
    ds = concatenate_datasets(all_ds).shuffle(seed=42)

    def messages_to_text(messages):
        return "\n".join(m["content"] for m in messages)

    class TokenizedDataset(Dataset):
        def __init__(self, rows, tokenizer, seq_len):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            text = messages_to_text(self.rows[idx]["messages"])
            tokens = self.tokenizer(
                text, truncation=True,
                max_length=self.seq_len + 1, padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0) # seq_len+1
            return input_ids[:-1], input_ids[1:]       # x, y

    dataset = TokenizedDataset(ds, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

@torch.no_grad()
def eval_boundaries(model, tokenizer, x, step, layer_idx=0, num_samples=16):
    """Run a full forward pass on num_samples sequences, read cached router state for metrics."""
    model.eval()
    moe_layer = model._moe_layers[layer_idx]
    router = moe_layer.router
    B = x.shape[0]
    n = min(num_samples, B)
    sample_indices = random.sample(range(B), n)
    viz_local = 0

    sub_x = x[sample_indices]
    model(input_ids=sub_x)

    pt_all = moe_layer._last_pt
    bt_all = moe_layer._last_bt
    top_k_indices_all = moe_layer._last_top_k_indices

    all_G, all_F, all_entropy, all_num_segments = [], [], [], []
    for i in range(n):
        pt_list = pt_all[i].cpu().tolist()
        bt_list = bt_all[i].cpu().tolist()
        all_G.append(sum(pt_list) / len(pt_list))
        all_F.append(sum(bt_list) / len(bt_list))
        pt_clamped = [max(1e-7, min(1 - 1e-7, p)) for p in pt_list]
        all_entropy.append(
            sum(-p * math.log(p) - (1 - p) * math.log(1 - p) for p in pt_clamped) / len(pt_clamped)
        )
        all_num_segments.append(sum(1 for j, b in enumerate(bt_list) if b or j == 0))

    viz_pt = pt_all[viz_local].cpu().tolist()
    viz_bt = bt_all[viz_local].cpu().tolist()
    viz_expert_ids = top_k_indices_all[viz_local].cpu().tolist()
    viz_input_ids = sub_x[viz_local]

    # --- visualization for one sample ---
    tokens = [tokenizer.decode(tid) for tid in viz_input_ids.cpu().tolist()]

    segments = []
    current_segment = []
    for i, tok in enumerate(tokens):
        if viz_bt[i] and current_segment:
            segments.append("".join(current_segment))
            current_segment = []
        current_segment.append(tok)
    if current_segment:
        segments.append("".join(current_segment))
    annotated = " | ".join(segments)

    table = wandb.Table(columns=["pos", "token", "boundary", "p_t", "experts"])
    for i, tok in enumerate(tokens):
        table.add_data(i, tok, int(viz_bt[i]), round(viz_pt[i], 4), str(viz_expert_ids[i]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    num_experts = router.num_experts
    top_k = router.top_k
    L = len(tokens)

    fig, (ax_exp, ax_pt) = plt.subplots(2, 1, figsize=(max(14, L * 0.08), 5),
                                         sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    colors = plt.cm.Set1(np.linspace(0, 1, num_experts))
    for k_idx in range(top_k):
        experts_k = [viz_expert_ids[i][k_idx] for i in range(L)]
        ax_exp.scatter(range(L), [k_idx] * L, c=[colors[e] for e in experts_k],
                       s=12, marker="s")

    boundary_positions = [i for i in range(L) if viz_bt[i]]
    for bp in boundary_positions:
        ax_exp.axvline(x=bp, color="red", alpha=0.4, linewidth=0.8, linestyle="--")
        ax_pt.axvline(x=bp, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

    ax_exp.set_yticks(range(top_k))
    ax_exp.set_yticklabels([f"top-{k+1}" for k in range(top_k)])
    ax_exp.set_ylabel("expert slot")
    ax_exp.set_title(f"Expert assignments over tokens (step {step}, layer {layer_idx})")

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[e], label=f"expert {e}") for e in range(num_experts)]
    ax_exp.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=num_experts)

    ax_pt.plot(range(L), viz_pt, color="black", linewidth=0.8)
    ax_pt.axhline(y=router._boundary_threshold, color="red", alpha=0.5, linewidth=0.8, linestyle=":")
    ax_pt.set_ylabel("p_t")
    ax_pt.set_xlabel("token position")
    ax_pt.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # --- log averaged metrics ---
    avg_G = sum(all_G) / len(all_G)
    avg_F = sum(all_F) / len(all_F)
    avg_entropy = sum(all_entropy) / len(all_entropy)
    avg_segments = sum(all_num_segments) / len(all_num_segments)

    wandb.log({
        "eval/segmented_text": wandb.Html(f"<pre>{annotated}</pre>"),
        "eval/boundary_table": table,
        "eval/expert_boundaries": wandb.Image(fig),
        "eval/num_segments": avg_segments,
        "eval/G_value": avg_G,
        "eval/F_value": avg_F,
        "eval/pt_entropy": avg_entropy,
    }, step=step)
    plt.close(fig)

    print(f"\n--- eval (step {step}, layer {layer_idx}, n={n}) ---")
    print(annotated[:500])
    print(f"avg segments: {avg_segments:.1f}, G_value: {avg_G:.3f}, F_value: {avg_F:.3f}, pt_entropy: {avg_entropy:.3f}")

    model.train()

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


def _unwrap(model):
    """Get the underlying model from DDP/accelerate wrappers."""
    return model.module if hasattr(model, "module") else model

# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="MoE Mixin PoC")
    parser.add_argument("--moe-type", choices=["vanilla", "temporal"], default="temporal",
                        help="MoE variant: vanilla (standard top-k) or temporal (chunking)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ratio-loss-N", type=int, default=3,
                        help="N parameter for temporal ratio loss (target segment length)")
    parser.add_argument("--ratio-loss-alpha", type=float, default=0.3,
                        help="Weight for temporal ratio loss")
    parser.add_argument("--entropy-threshold", type=float, default=0.1,
                        help="Floor for p_t entropy penalty (0 disables)")
    parser.add_argument("--entropy-alpha", type=float, default=1.0,
                        help="Weight for entropy penalty")
    parser.add_argument("--entropy-warmup-steps", type=int, default=0,
                        help="Steps to keep entropy alpha at zero before enabling")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-splits", nargs="+", default=["code"],
                        choices=["code", "math", "stem", "chat",
                                 "multilingual_ja", "multilingual_de",
                                 "multilingual_it", "multilingual_es",
                                 "multilingual_fr"],
                        help="Nemotron SFT splits to use (multiple allowed)")
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-chunking-poc")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save model checkpoints (default: no saving)")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

    # logging with wandb

    accelerator.init_trackers(args.wandb_project, config=vars(args))
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    # loading dense LLM model

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    dense_params = count_params(model)
    accelerator.print(f"Dense model parameters: {dense_params:,}")

    # dense --> MoE

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

    # load dataset
    accelerator.print(f"Loading Nemotron SFT ({args.dataset_splits}, {args.num_samples} docs) ...")
    loader = get_nemotron_loader(tokenizer, args.seq_len, args.batch_size, args.num_samples,
                                 splits=args.dataset_splits)

    # prepare for distributed training

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        wandb.log({"dense_params": dense_params, "moe_params": moe_params}, step=0)

    # train loop

    model.train()
    raw_model = _unwrap(model)
    num_moe_layers = len(raw_model._moe_layers)
    if args.moe_type == "temporal":
        ratio_loss_alpha = raw_model._moe_layers[0]._ratio_loss_alpha
    else:
        ratio_loss_alpha = None

    entropy_alpha_target = args.entropy_alpha if args.moe_type == "temporal" else 0.0

    step = 0
    for epoch in range(100): # enough epochs to hit num_steps
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False,
                    disable=not accelerator.is_main_process)
        for x, y in pbar:
            if args.moe_type == "temporal":
                ea = entropy_alpha_target if step >= args.entropy_warmup_steps else 0.0
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
                pbar.set_postfix(lm=f"{lm_loss.item():.4f}", aux=f"{aux_loss.item():.4f}",
                                 step=step)
                if step % args.log_every == 0:
                    metrics = {
                        "train/lm_loss": lm_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/epoch": epoch,
                    }
                    if ratio_loss_alpha is not None:
                        metrics["train/ratio_loss_per_layer_unscaled"] = (
                            aux_loss.item() / (num_moe_layers * ratio_loss_alpha)
                        )
                        avg_G = sum(m._last_G.item() for m in raw_model._moe_layers) / num_moe_layers
                        avg_F = sum(m._last_F.item() for m in raw_model._moe_layers) / num_moe_layers
                        avg_pt_entropy = sum(m._last_pt_entropy.item() for m in raw_model._moe_layers) / num_moe_layers
                        metrics["train/G_value"] = avg_G
                        metrics["train/F_value"] = avg_F
                        metrics["train/pt_entropy"] = avg_pt_entropy
                    wandb.log(metrics, step=step)

                if step % args.eval_every == 0 and args.moe_type == "temporal":
                    eval_boundaries(raw_model, tokenizer, x, step)

                if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(raw_model, tokenizer, args.save_dir, step, args, moe_config)

            if step >= args.num_steps:
                break
        if step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if accelerator.is_main_process and args.save_dir:
        save_checkpoint(raw_model, tokenizer, args.save_dir, step, args, moe_config)

    # final inspection
    if accelerator.is_main_process:
        raw_model.eval()
        prompt = "The mixture of experts architecture"
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            out = raw_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Prompt:     {prompt!r}")
        print(f"Generation: {gen_text!r}")
        wandb.log({"final/generation": wandb.Html(f"<pre>{gen_text}</pre>")}, step=step)

    accelerator.end_training()


if __name__ == "__main__":
    main()
