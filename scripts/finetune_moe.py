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
from accelerate import Accelerator

import transformers
import transformers.generation.utils as _gen_utils
for _name in ("DisjunctiveConstraint", "BeamSearchScorer", "PhrasalConstraint",
              "ConstrainedBeamSearchScorer"):
    if not hasattr(transformers, _name):
        setattr(transformers, _name, type(_name, (), {}))
if not hasattr(_gen_utils, "SampleOutput"):
    _gen_utils.SampleOutput = type("SampleOutput", (), {})

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_nemotron_loaders(tokenizer, seq_len, batch_size, dataset_name):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

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
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            return input_ids[:-1], input_ids[1:]

    train_loader = DataLoader(
        TokenizedDataset(ds["train"], tokenizer, seq_len),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TokenizedDataset(ds["test"], tokenizer, seq_len),
        batch_size=max(1, batch_size), shuffle=True, drop_last=True,
    )
    return train_loader, test_loader


def save_checkpoint(model, tokenizer, save_dir, step, epoch, args,
                    temporal_config, accelerator, optimizer=None,
                    lr_scheduler=None, keep_last=1):
    save_path = Path(save_dir) / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        meta = {
            "base_model": args.model,
            "temporal": args.temporal,
            "step": step,
            "epoch": epoch,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }
        if temporal_config is not None:
            meta["temporal_config"] = asdict(temporal_config)
        if wandb.run is not None:
            meta["wandb_run_id"] = wandb.run.id
        (save_path / "meta.json").write_text(json.dumps(meta, indent=2))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        raw.save_pretrained(save_path / "adapter")
        # Save temporal params separately if present
        temporal_state = {k: v.cpu() for k, v in raw.state_dict().items()
                         if "term_proj" in k or "_log_N" in k}
        if temporal_state:
            torch.save(temporal_state, save_path / "temporal_params.pt")
        torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
        torch.save(lr_scheduler.state_dict(), save_path / "scheduler.pt")
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
        has_new = (ckpt / ".complete").exists() and (ckpt / "adapter").exists()
        has_old = (ckpt / ".complete").exists() and (ckpt / "accelerator_state").exists()
        if has_new or has_old:
            return ckpt
    return None


@torch.no_grad()
def evaluate(model, test_loader, step, accelerator, n_batches=4, has_temporal=False):
    batches = list(test_loader)
    subset = random.sample(batches, min(n_batches, len(batches)))
    total_loss, total_tokens = 0.0, 0
    for x, y in subset:
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    if accelerator.is_main_process:
        metrics = {"eval/perplexity": ppl, "eval/lm_loss": avg_loss}
        if has_temporal:
            raw = accelerator.unwrap_model(model)
            base = raw.base_model.model if hasattr(raw, 'base_model') else raw
            if hasattr(base, '_moe_layers'):
                for i, layer in enumerate(base._moe_layers):
                    metrics[f"eval/layer_{i}/G_value"] = layer._last_G.item()
                    metrics[f"eval/layer_{i}/F_value"] = layer._last_F.item()
                    metrics[f"eval/layer_{i}/learned_N"] = layer.learned_N
        wandb.log(metrics, step=step)
        print(f"eval perplexity (step {step}): {ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained MoE LLM with LoRA, optionally adding temporal boundary routing"
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="HF model id")
    parser.add_argument("--dataset", type=str, default="ddidacus/nemotron-moe-exam")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=0,
                        help="Stop after N optimizer steps (0 = run all epochs)")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-chunking-poc")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint dir or 'auto' to find latest in --save-dir")

    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    lora_group.add_argument("--lora-dropout", type=float, default=0.05)
    lora_group.add_argument("--lora-target-modules", nargs="+",
                            default=["q_proj", "k_proj", "v_proj", "o_proj"],
                            help="Modules to apply LoRA to")

    temporal_group = parser.add_argument_group("Temporal MoE (optional)")
    temporal_group.add_argument("--temporal", action="store_true",
                                help="Add temporal boundary routing on top of existing MoE experts")
    temporal_group.add_argument("--ratio-loss-N", type=int, nargs="+", default=[8],
                                help="Target segment length N per MoE layer (single value = same for all)")
    temporal_group.add_argument("--ratio-loss-alpha", type=float, default=0.03)
    temporal_group.add_argument("--entropy-threshold", type=float, default=0.1)
    temporal_group.add_argument("--entropy-alpha", type=float, default=0.05)
    temporal_group.add_argument("--entropy-warmup-steps", type=int, default=500)

    args = parser.parse_args()
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

    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    wandb_kwargs = {}
    if args.wandb_run_name:
        wandb_kwargs["name"] = args.wandb_run_name
    if resume_ckpt:
        try:
            meta = json.loads((resume_ckpt / "meta.json").read_text())
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

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    temporal_config = None
    if args.temporal:
        temporal_config = TemporalWrapConfig(
            ratio_loss_N=args.ratio_loss_N,
            ratio_loss_alpha=args.ratio_loss_alpha,
            entropy_threshold=args.entropy_threshold,
            entropy_alpha=args.entropy_alpha,
        )
        TemporalWrapMixin.apply(model, temporal_config)
        accelerator.print("[temporal] Boundary routing applied to MoE layers")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    if args.temporal:
        for name, param in model.named_parameters():
            if "term_proj" in name or "_log_N" in name:
                param.requires_grad = True

    model.gradient_checkpointing_enable()

    total, trainable = count_params(model)
    accelerator.print(f"Total parameters: {total:,}")
    accelerator.print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

    accelerator.print(f"Loading dataset {args.dataset} ...")
    loader, test_loader = get_nemotron_loaders(
        tokenizer, args.seq_len, args.batch_size, args.dataset,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
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

    start_epoch = 0
    start_step = 0
    if resume_ckpt:
        try:
            meta = json.loads((resume_ckpt / "meta.json").read_text())
            start_step = meta["step"]
            start_epoch = meta.get("epoch", 0)
            if (resume_ckpt / "adapter").exists():
                accelerator.print(f"Restoring adapter weights from {resume_ckpt} ...")
                raw = accelerator.unwrap_model(model)
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                sf_path = resume_ckpt / "adapter" / "adapter_model.safetensors"
                if sf_path.exists():
                    adapter_state = load_file(str(sf_path))
                else:
                    adapter_state = torch.load(
                        resume_ckpt / "adapter" / "adapter_model.bin",
                        map_location="cpu", weights_only=True)
                set_peft_model_state_dict(raw, adapter_state)
                if (resume_ckpt / "temporal_params.pt").exists():
                    temporal_state = torch.load(
                        resume_ckpt / "temporal_params.pt", map_location="cpu",
                        weights_only=True)
                    raw.load_state_dict(temporal_state, strict=False)
                    accelerator.print(f"  Loaded {len(temporal_state)} temporal params")
                if (resume_ckpt / "optimizer.pt").exists():
                    opt_state = torch.load(
                        resume_ckpt / "optimizer.pt", map_location="cpu",
                        weights_only=True)
                    optimizer.load_state_dict(opt_state)
                if (resume_ckpt / "scheduler.pt").exists():
                    sched_state = torch.load(
                        resume_ckpt / "scheduler.pt", map_location="cpu",
                        weights_only=True)
                    lr_scheduler.load_state_dict(sched_state)
            else:
                accelerator.print(f"Restoring from legacy accelerator state ...")
                accelerator.load_state(str(resume_ckpt / "accelerator_state"))
            accelerator.print(f"Resumed at step={start_step}, epoch={start_epoch}")
        except (RuntimeError, FileNotFoundError) as e:
            accelerator.print(f"[WARNING] Failed to load checkpoint {resume_ckpt}: {e}")
            accelerator.print("Starting training from scratch.")
            start_step = 0
            start_epoch = 0

    if accelerator.is_main_process:
        wandb.log({"total_params": total, "trainable_params": trainable}, step=0)

    entropy_alpha_target = args.entropy_alpha if args.temporal else 0.0
    entropy_warmup = args.entropy_warmup_steps

    _preempt_state = {"step": start_step, "epoch": start_epoch}

    def _preemption_handler(signum, frame):
        s, e = _preempt_state["step"], _preempt_state["epoch"]
        accelerator.print(
            f"\n[preemption] Signal {signum} received at step {s}, saving checkpoint..."
        )
        if args.save_dir:
            save_checkpoint(model, tokenizer, args.save_dir, s, e, args,
                            temporal_config, accelerator,
                            optimizer=optimizer, lr_scheduler=lr_scheduler)
        accelerator.end_training()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _preemption_handler)
    signal.signal(signal.SIGTERM, _preemption_handler)

    model.train()
    raw_model = accelerator.unwrap_model(model)
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

        for x, y in pbar:
            with accelerator.accumulate(model):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss

                if args.temporal:
                    base = raw_model.base_model.model if hasattr(raw_model, 'base_model') else raw_model
                    moe_loss = base.get_moe_loss()
                    loss = loss + moe_loss

                accelerator.backward(loss)
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

            if args.temporal and entropy_warmup > 0:
                frac = min(1.0, step / entropy_warmup)
                base = raw_model.base_model.model if hasattr(raw_model, 'base_model') else raw_model
                for m in base._moe_layers:
                    m._entropy_alpha = entropy_alpha_target * frac

            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

                if step % args.log_every == 0:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    }
                    if args.temporal:
                        base = raw_model.base_model.model if hasattr(raw_model, 'base_model') else raw_model
                        if hasattr(base, '_moe_layers'):
                            log_dict["train/moe_loss"] = moe_loss.item()
                            for i, m in enumerate(base._moe_layers):
                                log_dict[f"train/layer_{i}/G_value"] = m._last_G.item()
                                log_dict[f"train/layer_{i}/F_value"] = m._last_F.item()
                                log_dict[f"train/layer_{i}/learned_N"] = m.learned_N
                    wandb.log(log_dict, step=step)

            if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.save_dir, step, epoch, args,
                                temporal_config, accelerator,
                                optimizer=optimizer, lr_scheduler=lr_scheduler)

            if step % args.eval_every == 0:
                raw_model.eval()
                evaluate(model, test_loader, step, accelerator,
                         has_temporal=args.temporal)
                raw_model.train()

            if args.num_steps and step >= args.num_steps:
                break
        if args.num_steps and step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if args.save_dir:
        save_checkpoint(model, tokenizer, args.save_dir, step, epoch, args,
                        temporal_config, accelerator,
                        optimizer=optimizer, lr_scheduler=lr_scheduler)
        if accelerator.is_main_process:
            (Path(args.save_dir) / "COMPLETED").write_text(f"step={step}\n")

    accelerator.end_training()


if __name__ == "__main__":
    main()
