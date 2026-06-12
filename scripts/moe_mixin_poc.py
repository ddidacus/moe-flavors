import argparse
import json
import math
import random
import signal
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

# Stub symbols removed in transformers >=5.x but still imported by transformers_stream_generator
# (used only for constrained beam search / streaming, not needed for training)
import transformers
import transformers.generation.utils as _gen_utils
for _name in ("DisjunctiveConstraint", "BeamSearchScorer", "PhrasalConstraint",
              "ConstrainedBeamSearchScorer"):
    if not hasattr(transformers, _name):
        setattr(transformers, _name, type(_name, (), {}))
if not hasattr(_gen_utils, "SampleOutput"):
    _gen_utils.SampleOutput = type("SampleOutput", (), {})

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.chunk_moe import MoEConfig as ChunkMoEConfig, MoEMixin as ChunkMoEMixin
from src.deepseek_moe import MoEConfig as DeepSeekMoEConfig, MoEMixin as DeepSeekMoEMixin
from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin


# ====================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_nemotron_loaders(tokenizer, seq_len: int, batch_size: int, dataset_name: str):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset, Subset

    ds = load_dataset(dataset_name)

    class TokenizedDataset(Dataset):
        def __init__(self, rows, tokenizer, seq_len):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            text = "\n".join(m["content"] for m in self.rows[idx]["messages"])
            tokens = self.tokenizer(
                text, truncation=True,
                max_length=self.seq_len + 1, padding="max_length",
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            return input_ids[:-1], input_ids[1:]

    train_loader = DataLoader(TokenizedDataset(ds["train"], tokenizer, seq_len),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TokenizedDataset(ds["test"], tokenizer, seq_len),
                             batch_size=max(1, batch_size), shuffle=True, drop_last=True)
    return train_loader, test_loader

def _pick_eval_layers(num_layers):
    """Return 3 equally spaced layer indices: first, middle, last."""
    if num_layers <= 3:
        return list(range(num_layers))
    mid = num_layers // 2
    return [0, mid, num_layers - 1]


@torch.no_grad()
def eval_boundaries(model, tokenizer, sub_x, step, layer_idx=0, num_samples=16):
    """Read cached router state for one layer and log visualizations."""
    moe_layer = model._moe_layers[layer_idx]
    router = moe_layer.router
    n = min(num_samples, sub_x.shape[0])
    viz_local = 0

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

    pfx = f"eval/layer_{layer_idx}"
    wandb.log({
        f"{pfx}/segmented_text": wandb.Html(f"<pre>{annotated}</pre>"),
        f"{pfx}/boundary_table": table,
        f"{pfx}/expert_boundaries": wandb.Image(fig),
        f"{pfx}/num_segments": avg_segments,
        f"{pfx}/G_value": avg_G,
        f"{pfx}/F_value": avg_F,
        f"{pfx}/pt_entropy": avg_entropy,
    }, step=step)
    plt.close(fig)

    print(f"\n--- eval (step {step}, layer {layer_idx}, n={n}) ---")
    print(annotated[:500])
    print(f"avg segments: {avg_segments:.1f}, G_value: {avg_G:.3f}, F_value: {avg_F:.3f}, pt_entropy: {avg_entropy:.3f}")


@torch.no_grad()
def eval_expert_assignments(model, tokenizer, sub_x, step, layer_idx=0):
    """Visualize expert assignments for vanilla MoE (no boundaries)."""
    moe_layer = model._moe_layers[layer_idx]
    top_k_indices_all = moe_layer._last_top_k_indices
    viz_expert_ids = top_k_indices_all[0].cpu().tolist()
    viz_input_ids = sub_x[0]

    tokens = [tokenizer.decode(tid) for tid in viz_input_ids.cpu().tolist()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    num_experts = moe_layer.config.num_experts
    top_k = moe_layer.config.top_k
    L = len(tokens)

    fig, ax = plt.subplots(figsize=(max(14, L * 0.08), 2.5))
    colors = plt.cm.Set1(np.linspace(0, 1, num_experts))
    for k_idx in range(top_k):
        experts_k = [viz_expert_ids[i][k_idx] for i in range(L)]
        ax.scatter(range(L), [k_idx] * L, c=[colors[e] for e in experts_k],
                   s=12, marker="s")

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"top-{k+1}" for k in range(top_k)])
    ax.set_ylabel("expert slot")
    ax.set_xlabel("token position")
    ax.set_title(f"Expert assignments over tokens (step {step}, layer {layer_idx})")
    legend_patches = [Patch(facecolor=colors[e], label=f"expert {e}") for e in range(num_experts)]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=num_experts)
    plt.tight_layout()

    pfx = f"eval/layer_{layer_idx}"
    wandb.log({f"{pfx}/expert_assignments": wandb.Image(fig)}, step=step)
    plt.close(fig)


def save_checkpoint(model, tokenizer, save_dir, step, epoch, args, moe_config,
                    accelerator):
    save_path = Path(save_dir) / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        meta = {
            "base_model": args.model,
            "moe_type": args.moe_type,
            "moe_config": asdict(moe_config),
            "step": step,
            "epoch": epoch,
        }
        (save_path / "moe_meta.json").write_text(json.dumps(meta, indent=2))
    accelerator.save_state(str(save_path / "accelerator_state"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (save_path / ".complete").write_text("")
    accelerator.print(f"[checkpoint] Saved to {save_path}")


def find_latest_checkpoint(save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    ckpts = sorted(save_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for ckpt in reversed(ckpts):
        if (ckpt / ".complete").exists() and (ckpt / "accelerator_state").exists():
            return ckpt
    return None


@torch.no_grad()
def eval_perplexity(model, test_loader, step, n_batches=4):
    device = next(model.parameters()).device
    batches = list(test_loader)
    subset = random.sample(batches, min(n_batches, len(batches)))
    total_loss, total_tokens = 0.0, 0
    for x, y in subset:
        x, y = x.to(device), y.to(device)
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    wandb.log({"eval/perplexity": ppl, "eval/lm_loss": avg_loss}, step=step)
    print(f"eval perplexity (step {step}): {ppl:.2f}")


@torch.no_grad()
def evaluate(model, test_loader, step):
    model.eval()
    eval_perplexity(model, test_loader, step)
    model.train()


@torch.no_grad()
def compute_switch_rate(model):
    """Fraction of adjacent token pairs whose sorted top-k expert sets differ."""
    rates = []
    for layer in model._moe_layers:
        indices = layer._last_top_k_indices                # (B, L, top_k)
        sorted_idx = indices.sort(dim=-1).values
        changed = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).any(dim=-1)  # (B, L-1)
        rates.append(changed.float().mean().item())
    return sum(rates) / len(rates)


def _unwrap(model):
    """Get the underlying model from DDP/accelerate wrappers."""
    return model.module if hasattr(model, "module") else model

# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="MoE Mixin PoC")
    parser.add_argument("--moe-type", choices=["vanilla", "temporal", "deepseek", "chunk"], default="temporal",
                        help="MoE variant: vanilla (plain top-k), deepseek (shared + routed), temporal (chunking), or chunk (CLS-token router)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-dim", type=int, default=None,
                        help="Per-expert intermediate dim (default: hidden_size // num_experts)")
    parser.add_argument("--num-copies", type=int, default=0,
                        help="FFN copies for segmentation upcycling (0 = random-init experts)")
    parser.add_argument("--num-shared-experts", type=int, default=2,
                        help="Number of always-active shared experts (deepseek mode)")
    parser.add_argument("--aux-loss-coeff", type=float, default=0.01,
                        help="Expert-level balance loss coefficient (deepseek mode)")
    parser.add_argument("--ratio-loss-N", type=int, nargs="+", default=[3],
                        help="N parameter(s) for temporal ratio loss — one per layer, or a single value broadcast to all")
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--dataset", type=str, default="ddidacus/nemotron-moe-exam",
                        help="HuggingFace Hub dataset name")
    parser.add_argument("--num-steps", type=int, default=0,
                        help="Stop after N optimizer steps (0 = run all epochs)")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (default: 0)")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of total steps used for LR warmup (default: 0)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "constant_with_warmup"],
                        help="LR scheduler type (default: cosine)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-chunking-poc")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save model checkpoints (default: no saving)")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint dir to resume from, or 'auto' to find latest in --save-dir")
    args = parser.parse_args()

    if args.num_copies > 0 and not args.expert_dim:
        parser.error("--expert-dim is required when --num-copies > 0")

    torch.manual_seed(args.seed)

    # resolve resume checkpoint
    resume_ckpt = None
    if args.resume_from == "auto" and args.save_dir:
        resume_ckpt = find_latest_checkpoint(args.save_dir)
    elif args.resume_from and args.resume_from != "auto":
        resume_ckpt = Path(args.resume_from)

    if args.save_dir and (Path(args.save_dir) / "COMPLETED").exists():
        print(f"Training already completed (found {args.save_dir}/COMPLETED). Exiting.")
        return

    if resume_ckpt:
        print(f"Will resume from checkpoint: {resume_ckpt}")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # logging with wandb

    wandb_kwargs = {}
    if args.wandb_run_name:
        wandb_kwargs["name"] = args.wandb_run_name
    if resume_ckpt:
        wandb_kwargs["resume"] = "allow"
    accelerator.init_trackers(
        args.wandb_project, config=vars(args),
        init_kwargs={"wandb": wandb_kwargs},
    )
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    # loading dense LLM model

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
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
            expert_dim=args.expert_dim,
            num_copies=args.num_copies,
        )
        TemporalMoEMixin.apply(model, moe_config)
    elif args.moe_type == "chunk":
        moe_config = ChunkMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            ratio_loss_N=args.ratio_loss_N,
            ratio_loss_alpha=args.ratio_loss_alpha,
            entropy_threshold=args.entropy_threshold,
            entropy_alpha=args.entropy_alpha,
            expert_dim=args.expert_dim,
            num_copies=args.num_copies,
        )
        ChunkMoEMixin.apply(model, moe_config)
    elif args.moe_type == "deepseek":
        moe_config = DeepSeekMoEConfig(
            num_experts=args.num_experts,
            num_shared_experts=args.num_shared_experts,
            top_k=args.top_k,
            aux_loss_coeff=args.aux_loss_coeff,
            expert_dim=args.expert_dim,
            num_copies=args.num_copies,
        )
        DeepSeekMoEMixin.apply(model, moe_config)
    else:
        moe_config = VanillaMoEConfig(
            num_experts=args.num_experts,
            top_k=args.top_k,
            expert_dim=args.expert_dim,
            num_copies=args.num_copies,
        )
        VanillaMoEMixin.apply(model, moe_config)
    moe_params = count_params(model)
    accelerator.print(f"MoE model parameters:   {moe_params:,}  "
                      f"({moe_params / dense_params:.1f}x dense)")

    # load dataset
    accelerator.print(f"Loading dataset {args.dataset} ...")
    loader, test_loader = get_nemotron_loaders(tokenizer, args.seq_len, args.batch_size,
                                               args.dataset)

    # prepare for distributed training

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = len(loader) * args.num_epochs // args.gradient_accumulation_steps
    if args.num_steps:
        total_steps = min(total_steps, args.num_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    accelerator.print(f"LR scheduler: {args.lr_scheduler}, warmup={warmup_steps}/{total_steps} steps, "
                      f"weight_decay={args.weight_decay}")

    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler)

    # resume from checkpoint
    start_epoch = 0
    start_step = 0
    if resume_ckpt:
        accelerator.print(f"Restoring training state from {resume_ckpt} ...")
        accelerator.load_state(str(resume_ckpt / "accelerator_state"))
        meta = json.loads((resume_ckpt / "moe_meta.json").read_text())
        start_step = meta["step"]
        start_epoch = meta.get("epoch", 0)
        accelerator.print(f"Resumed at step={start_step}, epoch={start_epoch}")

    if accelerator.is_main_process:
        wandb.log({"dense_params": dense_params, "moe_params": moe_params}, step=0)

    # preemption handler — save checkpoint on SIGTERM/SIGUSR1 from SLURM
    _preempt_state = {"step": start_step, "epoch": start_epoch}

    def _preemption_handler(signum, frame):
        s, e = _preempt_state["step"], _preempt_state["epoch"]
        accelerator.print(f"\n[preemption] Signal {signum} received at step {s}, saving checkpoint...")
        if args.save_dir:
            save_checkpoint(model, tokenizer, args.save_dir, s, e, args, moe_config, accelerator)
        accelerator.end_training()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _preemption_handler)
    signal.signal(signal.SIGTERM, _preemption_handler)

    # train loop

    model.train()
    raw_model = _unwrap(model)
    num_moe_layers = len(raw_model._moe_layers)
    if args.moe_type in ("temporal", "chunk"):
        ratio_loss_alpha = raw_model._moe_layers[0]._ratio_loss_alpha
    else:
        ratio_loss_alpha = None

    entropy_alpha_target = args.entropy_alpha if args.moe_type in ("temporal", "chunk") else 0.0

    step = start_step
    batches_per_step = args.gradient_accumulation_steps

    # epochs
    for epoch in range(start_epoch, args.num_epochs):

        # resuming skipping
        if epoch == start_epoch and start_step > 0:
            skip_batches = start_step * batches_per_step
            accelerator.print(f"Skipping {skip_batches} batches to resume epoch {epoch} ...")
            active_loader = accelerator.skip_first_batches(loader, skip_batches)
        else:
            active_loader = loader
        pbar = tqdm(active_loader, desc=f"epoch {epoch}", leave=False,
                    disable=not accelerator.is_main_process)

        # iterate dataloader
        for x, y in pbar:
            with accelerator.accumulate(model):
                if args.moe_type in ("temporal", "chunk"):
                    ea = entropy_alpha_target if step >= args.entropy_warmup_steps else 0.0
                    for m in raw_model._moe_layers:
                        m._entropy_alpha = ea

                outputs = model(input_ids=x, labels=y)
                aux_loss = raw_model.get_moe_loss()

                if args.moe_type in ("temporal", "chunk"):
                    total_loss = outputs.loss
                    lm_loss = (total_loss - aux_loss).detach()
                else:
                    lm_loss = outputs.loss
                    total_loss = lm_loss + aux_loss
                    lm_loss = lm_loss.detach()
                aux_loss = aux_loss.detach()

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            step += 1
            _preempt_state["step"] = step
            _preempt_state["epoch"] = epoch

            # logging
            if accelerator.is_main_process:

                pbar.set_postfix(lm=f"{lm_loss.item():.4f}", aux=f"{aux_loss.item():.4f}",
                                 step=step)

                if step % args.log_every == 0:
                    metrics = {
                        "train/lm_loss": lm_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/switch_rate": compute_switch_rate(raw_model),
                    }
                    if ratio_loss_alpha:
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

            # save checkpoint (before eval so OOM in eval doesn't lose progress)
            if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.save_dir, step, epoch, args,
                                moe_config, accelerator)

            # evaluation plots
            if accelerator.is_main_process and step % args.eval_every == 0:
                raw_model.eval()

                # perplexity
                evaluate(raw_model, test_loader, step)

                # plot boundaries for random batch
                n_eval = min(16, x.shape[0])
                sub_x = x[random.sample(range(x.shape[0]), n_eval)]
                raw_model(input_ids=sub_x)
                for li in _pick_eval_layers(num_moe_layers):
                    if args.moe_type in ("temporal", "chunk"):
                        eval_boundaries(raw_model, tokenizer, sub_x, step,
                                        layer_idx=li, num_samples=n_eval)
                    else:
                        eval_expert_assignments(raw_model, tokenizer, sub_x,
                                                step, layer_idx=li)
                raw_model.train()

            if args.num_steps and step >= args.num_steps:
                break
        if args.num_steps and step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if args.save_dir:
        save_checkpoint(model, tokenizer, args.save_dir, step, epoch, args,
                        moe_config, accelerator)
        if accelerator.is_main_process:
            (Path(args.save_dir) / "COMPLETED").write_text(f"step={step}\n")

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
