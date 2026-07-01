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

import transformers
import transformers.generation.utils as _gen_utils
for _name in ("DisjunctiveConstraint", "BeamSearchScorer", "PhrasalConstraint",
              "ConstrainedBeamSearchScorer"):
    if not hasattr(transformers, _name):
        setattr(transformers, _name, type(_name, (), {}))
if not hasattr(_gen_utils, "SampleOutput"):
    _gen_utils.SampleOutput = type("SampleOutput", (), {})

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from src.vanilla_moe import MoEConfig, MoEMixin


# ── domain boundary helpers (from extract_routing_vectors.py) ──

DOMAINS = ["chat", "code", "math"]


def build_domain_char_spans(messages):
    spans = []
    char_offset = 0
    i = 0
    while i < len(messages):
        content = messages[i]["content"]
        msg_start = char_offset
        char_offset += len(content) + 1
        domain_found = None
        for domain in DOMAINS:
            prefix = f"Here is a question from the {domain} domain."
            if content.startswith(prefix):
                domain_found = domain
                break
        if domain_found is not None and i + 1 < len(messages):
            assistant_content = messages[i + 1]["content"]
            span_end = char_offset + len(assistant_content)
            char_offset += len(assistant_content) + 1
            spans.append((DOMAINS.index(domain_found), msg_start, span_end))
            i += 2
        else:
            i += 1
    return spans


def tokens_to_domain_mask(offset_mapping, domain_spans, seq_len):
    mask = torch.full((seq_len,), -1, dtype=torch.long)
    for tok_idx in range(seq_len):
        tok_start, tok_end = offset_mapping[tok_idx]
        if tok_start == tok_end:
            continue
        tok_mid = (tok_start + tok_end) / 2
        for domain_idx, span_start, span_end in domain_spans:
            if span_start <= tok_mid < span_end:
                mask[tok_idx] = domain_idx
                break
    return mask


# ── utilities ──

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def compute_switch_rate(model):
    rates = []
    for layer in model._moe_layers:
        indices = layer._last_top_k_indices
        sorted_idx = indices.sort(dim=-1).values
        changed = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).any(dim=-1)
        rates.append(changed.float().mean().item())
    return sum(rates) / len(rates)


# ── dataset ──

def get_supervised_loaders(tokenizer, seq_len, batch_size, dataset_name):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    ds = load_dataset(dataset_name)

    class SupervisedTokenizedDataset(Dataset):
        def __init__(self, rows, tokenizer, seq_len):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            messages = self.rows[idx]["messages"]
            text = "\n".join(m["content"] for m in messages)

            tokens = self.tokenizer(
                text, truncation=True,
                max_length=self.seq_len + 1, padding="max_length",
                return_tensors="pt", return_offsets_mapping=True,
            )
            input_ids = tokens["input_ids"].squeeze(0)
            offset_mapping = tokens["offset_mapping"].squeeze(0).tolist()

            domain_spans = build_domain_char_spans(messages)
            domain_mask = tokens_to_domain_mask(
                offset_mapping, domain_spans, self.seq_len + 1,
            )

            # boundary_mask[i] = 1 where domain changes between input token i and i+1
            # input tokens are positions 0..seq_len-1, so transitions are 0..seq_len-2
            dm = domain_mask[:self.seq_len]
            valid_pair = (dm[:-1] >= 0) & (dm[1:] >= 0)
            changed = dm[:-1] != dm[1:]
            boundary_mask = (valid_pair & changed).float()

            return input_ids[:-1], input_ids[1:], boundary_mask

    train_loader = DataLoader(
        SupervisedTokenizedDataset(ds["train"], tokenizer, seq_len),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        SupervisedTokenizedDataset(ds["test"], tokenizer, seq_len),
        batch_size=max(1, batch_size), shuffle=True, drop_last=True,
    )
    return train_loader, test_loader


# ── eval ──

@torch.no_grad()
def eval_perplexity(model, test_loader, step, accelerator, n_batches=4):
    device = next(model.parameters()).device
    batches = list(test_loader)
    subset = random.sample(batches, min(n_batches, len(batches)))
    total_loss, total_tokens = 0.0, 0
    for batch in subset:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    if accelerator.is_main_process:
        wandb.log({"eval/perplexity": ppl, "eval/lm_loss": avg_loss}, step=step)
        print(f"eval perplexity (step {step}): {ppl:.2f}")


@torch.no_grad()
def evaluate(model, test_loader, step, accelerator):
    model.eval()
    eval_perplexity(model, test_loader, step, accelerator)
    model.train()


@torch.no_grad()
def eval_cosine_boundaries(model, tokenizer, sub_x, boundary_targets, step,
                           tau, last_moe_output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    last_layer = model._moe_layers[-1]
    top_k_indices = last_layer._last_top_k_indices

    h = last_moe_output["h"]
    h_norm = F.normalize(h.float(), dim=-1)
    delta = 1.0 - (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1)

    viz_idx = 0
    viz_expert_ids = top_k_indices[viz_idx].cpu().tolist()
    viz_delta = delta[viz_idx].cpu().tolist()
    viz_boundaries = boundary_targets[viz_idx].cpu()
    gt_positions = viz_boundaries.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(gt_positions, int):
        gt_positions = [gt_positions]
    tokens = [tokenizer.decode(tid) for tid in sub_x[viz_idx].cpu().tolist()]

    L = len(tokens)
    num_experts = last_layer.config.num_experts
    top_k = last_layer.config.top_k

    fig, (ax_exp, ax_delta) = plt.subplots(
        2, 1, figsize=(max(14, L * 0.08), 5),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    n_colors = min(num_experts, 20)
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    for k_idx in range(top_k):
        experts_k = [viz_expert_ids[i][k_idx] for i in range(L)]
        ax_exp.scatter(
            range(L), [k_idx] * L,
            c=[colors[e % n_colors] for e in experts_k],
            s=12, marker="s",
        )

    for pos in gt_positions:
        ax_exp.axvline(x=pos, color="red", alpha=0.4, linewidth=0.8, linestyle="--")
        ax_delta.axvline(x=pos, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

    ax_exp.set_yticks(range(top_k))
    ax_exp.set_yticklabels([f"top-{k+1}" for k in range(top_k)])
    ax_exp.set_ylabel("expert slot")
    ax_exp.set_title(
        f"Expert assignments + cosine boundaries (step {step}, last layer)"
    )

    ax_delta.plot(range(L - 1), viz_delta, color="black", linewidth=0.8)
    ax_delta.axhline(y=tau, color="red", alpha=0.5, linewidth=0.8, linestyle=":")
    ax_delta.set_ylabel("δ_i (cosine dissim)")
    ax_delta.set_xlabel("token position")
    delta_max = max(viz_delta) if viz_delta else 1.0
    ax_delta.set_ylim(-0.05, delta_max * 1.1 + 0.05)

    plt.tight_layout()

    wandb.log({
        "eval/last_layer/expert_cosine_boundaries": wandb.Image(fig),
        "eval/last_layer/avg_delta": delta.mean().item(),
        "eval/last_layer/switch_rate": (delta > tau).float().mean().item(),
    }, step=step)
    plt.close(fig)

    print(f"\n--- eval cosine boundaries (step {step}, last layer) ---")
    print(f"avg delta: {delta.mean().item():.4f}, "
          f"switch_rate(tau={tau}): {(delta > tau).float().mean().item():.4f}, "
          f"GT boundaries: {len(gt_positions)}")


# ── checkpointing ──

def save_checkpoint(model, tokenizer, save_dir, step, epoch, args, moe_config,
                    accelerator, keep_last=1):
    save_path = Path(save_dir) / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        meta = {
            "base_model": args.model,
            "moe_type": "vanilla_supervised",
            "moe_config": asdict(moe_config),
            "step": step,
            "epoch": epoch,
        }
        if wandb.run is not None:
            meta["wandb_run_id"] = wandb.run.id
        (save_path / "moe_meta.json").write_text(json.dumps(meta, indent=2))
    accelerator.wait_for_everyone()
    accelerator.save_state(str(save_path / "accelerator_state"))
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)
        (save_path / ".complete").write_text("")
    accelerator.wait_for_everyone()
    accelerator.print(f"[checkpoint] Saved to {save_path}")
    if accelerator.is_main_process:
        _rotate_checkpoints(Path(save_dir), keep_last, accelerator)


def _rotate_checkpoints(save_dir, keep_last, accelerator):
    import shutil
    ckpts = sorted(
        [p for p in save_dir.glob("step_*") if (p / ".complete").exists()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    while len(ckpts) > keep_last:
        old = ckpts.pop(0)
        shutil.rmtree(old)
        accelerator.print(f"[checkpoint] Rotated out {old}")


def find_latest_checkpoint(save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    ckpts = sorted(save_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for ckpt in reversed(ckpts):
        if (ckpt / ".complete").exists() and (ckpt / "accelerator_state").exists():
            return ckpt
    return None


# ── main ──

def main():
    parser = argparse.ArgumentParser(
        description="Vanilla MoE + supervised Tversky/switch regularization"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-dim", type=int, default=None)
    parser.add_argument("--num-copies", type=int, default=0)
    parser.add_argument("--aux-loss-coeff", type=float, default=0.01)
    # supervised regularization
    parser.add_argument("--semantic-alpha", type=float, default=0.1,
                        help="Weight for Tversky semantic consistency loss")
    parser.add_argument("--switch-lambda", type=float, default=0.01,
                        help="Weight for switch/cache loss")
    parser.add_argument("--tversky-alpha", type=float, default=0.3,
                        help="Tversky false-positive penalty")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                        help="Tversky false-negative penalty")
    parser.add_argument("--switch-tau", type=float, default=0.3,
                        help="Cosine dissimilarity threshold for switch detection")
    parser.add_argument("--switch-temperature", type=float, default=0.1,
                        help="Sigmoid temperature for differentiable switch indicator")
    # training
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ddidacus/nemotron-moe-exam")
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-chunking-poc")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    if args.num_copies > 0 and not args.expert_dim:
        parser.error("--expert-dim is required when --num-copies > 0")

    torch.manual_seed(args.seed)

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

    wandb_kwargs = {}
    if args.wandb_run_name:
        wandb_kwargs["name"] = args.wandb_run_name
    if resume_ckpt:
        try:
            meta = json.loads((resume_ckpt / "moe_meta.json").read_text())
            if "wandb_run_id" in meta:
                wandb_kwargs["id"] = meta["wandb_run_id"]
                wandb_kwargs["resume"] = "must"
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    accelerator.init_trackers(
        args.wandb_project, config=vars(args),
        init_kwargs={"wandb": wandb_kwargs},
    )
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    # load model

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    dense_params = count_params(model)
    accelerator.print(f"Dense model parameters: {dense_params:,}")

    # dense -> vanilla MoE

    moe_config = MoEConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        aux_loss_coeff=args.aux_loss_coeff,
        expert_dim=args.expert_dim,
        num_copies=args.num_copies,
    )
    MoEMixin.apply(model, moe_config)
    moe_params = count_params(model)
    accelerator.print(f"MoE model parameters:   {moe_params:,}  "
                      f"({moe_params / dense_params:.1f}x dense)")

    # register forward hook on last MoE layer to capture output
    _last_moe_output = {}

    def _capture_last_moe(_module, _input, output):
        _last_moe_output["h"] = output

    model._moe_layers[-1].register_forward_hook(_capture_last_moe)

    # load dataset

    accelerator.print(f"Loading dataset {args.dataset} ...")
    loader, test_loader = get_supervised_loaders(
        tokenizer, args.seq_len, args.batch_size, args.dataset,
    )

    # optimizer + scheduler

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

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
    accelerator.print(
        f"LR scheduler: {args.lr_scheduler}, warmup={warmup_steps}/{total_steps} steps"
    )

    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler,
    )

    # resume

    start_epoch = 0
    start_step = 0
    if resume_ckpt:
        try:
            accelerator.print(f"Restoring training state from {resume_ckpt} ...")
            accelerator.load_state(str(resume_ckpt / "accelerator_state"))
            meta = json.loads((resume_ckpt / "moe_meta.json").read_text())
            start_step = meta["step"]
            start_epoch = meta.get("epoch", 0)
            accelerator.print(f"Resumed at step={start_step}, epoch={start_epoch}")
        except (RuntimeError, FileNotFoundError) as e:
            accelerator.print(f"[WARNING] Failed to load checkpoint {resume_ckpt}: {e}")
            accelerator.print("Starting training from scratch.")
            start_step = 0
            start_epoch = 0

    if accelerator.is_main_process:
        wandb.log({"dense_params": dense_params, "moe_params": moe_params}, step=0)

    # preemption handler

    _preempt_state = {"step": start_step, "epoch": start_epoch}

    def _preemption_handler(signum, frame):
        s, e = _preempt_state["step"], _preempt_state["epoch"]
        accelerator.print(
            f"\n[preemption] Signal {signum} received at step {s}, saving checkpoint..."
        )
        if args.save_dir:
            save_checkpoint(
                model, tokenizer, args.save_dir, s, e, args, moe_config, accelerator,
            )
        accelerator.end_training()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _preemption_handler)
    signal.signal(signal.SIGTERM, _preemption_handler)

    # ── training loop ──

    model.train()
    raw_model = _unwrap(model)
    num_moe_layers = len(raw_model._moe_layers)
    step = start_step
    batches_per_step = args.gradient_accumulation_steps

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == start_epoch and start_step > 0:
            skip_batches = start_step * batches_per_step
            accelerator.print(f"Skipping {skip_batches} batches to resume epoch {epoch} ...")
            active_loader = accelerator.skip_first_batches(loader, skip_batches)
        else:
            active_loader = loader

        pbar = tqdm(
            active_loader, desc=f"epoch {epoch}", leave=False,
            disable=not accelerator.is_main_process,
        )

        for x, y, boundary_targets in pbar:
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y)
                lm_loss = outputs.loss
                aux_loss = raw_model.get_moe_loss()

                # supervised regularization on last MoE layer
                h = _last_moe_output["h"]
                h_norm = F.normalize(h.float(), dim=-1)
                delta = 1.0 - (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1)

                valid = boundary_targets.sum(dim=-1) > 0
                if valid.any():
                    d = delta[valid]
                    y_gt = boundary_targets[valid].to(d.dtype)
                    tp = (d * y_gt).sum()
                    fp = (d * (1.0 - y_gt)).sum()
                    fn = ((1.0 - d) * y_gt).sum()
                    tversky = tp / (tp + args.tversky_alpha * fp + args.tversky_beta * fn + 1e-8)
                    semantic_loss = 1.0 - tversky
                else:
                    semantic_loss = torch.tensor(0.0, device=x.device)

                switch_loss = torch.sigmoid(
                    (delta - args.switch_tau) / args.switch_temperature
                ).mean()

                total_loss = (
                    lm_loss
                    + aux_loss
                    + args.semantic_alpha * semantic_loss
                    + args.switch_lambda * switch_loss
                )

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            step += 1
            _preempt_state["step"] = step
            _preempt_state["epoch"] = epoch

            if accelerator.is_main_process:
                pbar.set_postfix(
                    lm=f"{lm_loss.item():.4f}",
                    sem=f"{semantic_loss.item():.4f}",
                    sw=f"{switch_loss.item():.4f}",
                    step=step,
                )

                if step % args.log_every == 0:
                    metrics = {
                        "train/lm_loss": lm_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/semantic_loss": semantic_loss.item()
                            if torch.is_tensor(semantic_loss) else semantic_loss,
                        "train/switch_loss": switch_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/avg_delta": delta.mean().item(),
                        "train/switch_rate": (delta > args.switch_tau).float().mean().item(),
                        "train/switch_rate_expert": compute_switch_rate(raw_model),
                    }
                    wandb.log(metrics, step=step)

            # checkpoint
            if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    model, tokenizer, args.save_dir, step, epoch, args,
                    moe_config, accelerator,
                )

            # eval
            if step % args.eval_every == 0:
                raw_model.eval()

                evaluate(model, test_loader, step, accelerator)

                n_eval = min(16, x.shape[0])
                sample_idx = random.sample(range(x.shape[0]), n_eval)
                sub_x = x[sample_idx]
                sub_bt = boundary_targets[sample_idx]
                model(input_ids=sub_x)

                if accelerator.is_main_process:
                    eval_cosine_boundaries(
                        raw_model, tokenizer, sub_x, sub_bt, step,
                        args.switch_tau, _last_moe_output,
                    )

                raw_model.train()

            if args.num_steps and step >= args.num_steps:
                break
        if args.num_steps and step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if args.save_dir:
        save_checkpoint(
            model, tokenizer, args.save_dir, step, epoch, args,
            moe_config, accelerator,
        )
        if accelerator.is_main_process:
            (Path(args.save_dir) / "COMPLETED").write_text(f"step={step}\n")

    try:
        raw_model.eval()
        prompt = "The mixture of experts architecture"
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        with torch.no_grad():
            out = raw_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        if accelerator.is_main_process:
            gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"Prompt:     {prompt!r}")
            print(f"Generation: {gen_text!r}")
            wandb.log(
                {"final/generation": wandb.Html(f"<pre>{gen_text}</pre>")}, step=step,
            )
    except RuntimeError as e:
        accelerator.print(f"[WARNING] Generation failed: {e}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
