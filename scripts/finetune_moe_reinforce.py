"""REINFORCE-with-baseline finetuning of an MoE LLM (OLMoE) with two rewards:

  r_t = alpha * r_t^BC + (1 - alpha) * r_t^cache

  * r_t^BC     = log p_teacher(a_t|x,a_<t) - log p_student(a_t|x,a_<t)
                 (-r_t^BC is an unbiased estimator of KL(student || teacher)).
                 Rollout tokens are sampled from the mixture
                 p_mix = (1-tau) p_student + tau p_teacher to prevent hacking,
                 corrected with importance weights w_t = p_student/p_mix.
  * r_t^cache  = 1[e_t in C] / T, where C is a fixed-size LRU working set
                 of experts simulated at one router (default: middle layer)
                 and e_t ~ G(x_t) is the expert sampled from that router
                 (positive reward for cache hits).

Prompts x come from nvidia/Nemotron-Post-Training-Dataset-v2; continuations
are generated on-policy. The teacher defaults to the frozen base model
(LoRA adapters disabled), or a separate HF model via --teacher-model.
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

import torch
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

from src.cache_reinforce import (cache_emulation_rewards, kd_reward_terms,
                                 rewards_to_go)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_nemotron_prompt_loaders(tokenizer, prompt_len, batch_size, dataset_name,
                                split, max_samples, seed, test_fraction=0.02):
    """Nemotron rows -> prompt-only loaders (system+user turns, left-padded)."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
    prompts = []
    for sp in splits:
        ds = load_dataset(dataset_name, split=sp, streaming=True)
        n = 0
        for row in ds:
            parts = []
            for m in row["messages"]:
                if m["role"] == "assistant":
                    break
                if m["content"].strip():
                    parts.append(m["content"])
            text = "\n".join(parts).strip()
            if text:
                prompts.append(text)
                n += 1
            if n >= per_split:
                break

    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_test = max(batch_size, int(len(prompts) * test_fraction))
    test_prompts, train_prompts = prompts[:n_test], prompts[n_test:]

    def collate(batch):
        enc = tokenizer(batch, truncation=True, max_length=prompt_len,
                        padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    train_loader = DataLoader(train_prompts, batch_size=batch_size,
                              shuffle=True, drop_last=True, collate_fn=collate)
    test_loader = DataLoader(test_prompts, batch_size=batch_size,
                             shuffle=False, drop_last=True, collate_fn=collate)
    return train_loader, test_loader


def make_teacher_forward(raw_model, external_teacher):
    """Teacher logits fn. Default: same weights with LoRA adapters disabled."""
    if external_teacher is not None:
        def teacher_forward(input_ids, attention_mask, position_ids=None,
                            past_key_values=None, use_cache=False):
            with torch.no_grad():
                return external_teacher(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache)
    else:
        def teacher_forward(input_ids, attention_mask, position_ids=None,
                            past_key_values=None, use_cache=False):
            with torch.no_grad(), raw_model.disable_adapter():
                return raw_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache)
    return teacher_forward


@torch.no_grad()
def mixture_rollout(raw_model, teacher_forward, input_ids, attention_mask,
                    gen_len, tau, eos_id, pad_id):
    """Sample continuations a_t ~ p_mix = (1-tau) p_student + tau p_teacher.

    Sampling is hierarchical, which is exactly equivalent to sampling from
    p_mix: per step flip b_t ~ Bernoulli(tau) (shared across the batch), then
    sample from the teacher if b_t=1 else from the student. The teacher is
    therefore only run when its branch fires, catching up its KV cache on all
    tokens generated since its last call in one chunked forward — the decode
    loop is dominated by 1 student call/step instead of 2 sequential calls.
    (The KD reward and IS weights never use rollout logits; they are
    recomputed in the teacher-forced grad pass.)

    Returns the full (prompt + generation) ids/mask and a bool mask over
    generated action tokens (post-EOS padding excluded).
    """
    was_training = raw_model.training
    raw_model.eval()
    device = input_ids.device
    B = input_ids.size(0)

    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    s_out = raw_model(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=True)
    s_cache = s_out.past_key_values
    logits_s = s_out.logits[:, -1]
    t_cache, logits_t = None, None  # teacher prefill deferred to first use

    next_pos = attention_mask.sum(-1, keepdim=True)  # left-padded prompts
    teacher_pos = next_pos.clone()   # next position the teacher has NOT seen
    pending = []                     # tokens generated since last teacher call
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    gen_tokens, gen_active = [], []

    for _ in range(gen_len):
        if torch.rand(()).item() < tau:
            if t_cache is None:
                t_out = teacher_forward(input_ids, attention_mask[:, :input_ids.size(1)],
                                        position_ids=position_ids, use_cache=True)
                t_cache, logits_t = t_out.past_key_values, t_out.logits[:, -1]
            if pending:
                chunk = torch.stack(pending, dim=1)
                k = chunk.size(1)
                chunk_pos = teacher_pos + torch.arange(k, device=device).unsqueeze(0)
                t_out = teacher_forward(chunk, attention_mask,
                                        position_ids=chunk_pos,
                                        past_key_values=t_cache, use_cache=True)
                t_cache, logits_t = t_out.past_key_values, t_out.logits[:, -1]
                teacher_pos = teacher_pos + k
                pending = []
            probs = torch.softmax(logits_t.float(), -1)
        else:
            probs = torch.softmax(logits_s.float(), -1)

        a = torch.multinomial(probs, 1).squeeze(-1)
        a = torch.where(finished, torch.full_like(a, pad_id), a)
        gen_tokens.append(a)
        gen_active.append(~finished)
        finished = finished | (a == eos_id)

        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype,
                                        device=device)], dim=1)
        if finished.all():
            break
        step_ids = a.unsqueeze(-1)
        s_out = raw_model(input_ids=step_ids, attention_mask=attention_mask,
                          position_ids=next_pos, past_key_values=s_cache,
                          use_cache=True)
        s_cache, logits_s = s_out.past_key_values, s_out.logits[:, -1]
        pending.append(a)
        next_pos = next_pos + 1

    full_ids = torch.cat([input_ids, torch.stack(gen_tokens, dim=1)], dim=1)
    action_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    action_mask[:, input_ids.size(1):] = torch.stack(gen_active, dim=1)

    if was_training:
        raw_model.train()
    return full_ids, attention_mask, action_mask


def save_checkpoint(model, tokenizer, save_dir, step, epoch, args, accelerator,
                    optimizer=None, lr_scheduler=None, keep_last=1):
    save_path = Path(save_dir) / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        meta = {
            "base_model": args.model,
            "step": step,
            "epoch": epoch,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "alpha": args.alpha,
            "mix_tau": args.mix_tau,
            "cache_size": args.cache_size,
            "cache_layer": args.cache_layer,
        }
        if wandb.run is not None:
            meta["wandb_run_id"] = wandb.run.id
        (save_path / "meta.json").write_text(json.dumps(meta, indent=2))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        raw.save_pretrained(save_path / "adapter")
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
        if (ckpt / ".complete").exists() and (ckpt / "adapter").exists():
            return ckpt
    return None


def compute_reinforce_loss(model, raw_model, teacher_forward, full_ids,
                           full_mask, action_mask, args, baseline_state):
    """One teacher-forced grad pass over prompt+generation -> REINFORCE loss.

    Everything on the "target grid": logits at position i predict token i+1,
    so tensors below are sliced/aligned to full_ids[:, 1:].
    """
    B, S = full_ids.shape
    out = model(input_ids=full_ids, attention_mask=full_mask,
                output_router_logits=True)
    t_logits = teacher_forward(full_ids, full_mask).logits

    actions = full_ids[:, 1:]
    act_mask = action_mask[:, 1:].float()
    logp_student, r_bc, w = kd_reward_terms(
        out.logits[:, :-1], t_logits[:, :-1], actions, args.mix_tau)

    # Router logits at the cached layer, one row per token of the sequence.
    router_logits = out.router_logits[args.cache_layer].view(B, S, -1)
    r_cache_full, expert_samples, hit_rate = cache_emulation_rewards(
        router_logits.detach(), full_mask.bool(), action_mask,
        cache_size=args.cache_size,
        experts_per_token=args.cache_experts_per_token,
        use_topk=args.cache_topk,
    )
    r_cache = r_cache_full[:, 1:]  # cache reward of token a_t = full_ids[:, t]

    rewards = (args.alpha * r_bc + (1.0 - args.alpha) * r_cache) * act_mask
    returns = rewards_to_go(rewards, gamma=args.gamma)

    # Scalar EMA baseline over per-token returns.
    batch_mean = (returns * act_mask).sum() / act_mask.sum().clamp(min=1)
    if baseline_state["value"] is None:
        baseline_state["value"] = batch_mean.item()
    else:
        m = args.baseline_ema
        baseline_state["value"] = m * baseline_state["value"] \
            + (1 - m) * batch_mean.item()
    advantages = (returns - baseline_state["value"]) * act_mask

    # Policy term: token log-prob, plus router log-prob of the sampled expert
    # so the cache reward reaches the router directly (sampled mode only —
    # top-k selection is deterministic, no REINFORCE term for it).
    policy_logp = logp_student
    if not args.cache_topk:
        router_logp = torch.log_softmax(router_logits.float(), dim=-1) \
            .gather(-1, expert_samples.to(router_logits.device)) \
            .sum(-1)
        policy_logp = policy_logp + router_logp[:, 1:]

    if args.is_clip > 0:
        w = w.clamp(max=args.is_clip)
    loss = -(w * advantages.detach() * policy_logp * act_mask).sum() \
        / act_mask.sum().clamp(min=1)

    stats = {
        "r_bc": (r_bc * act_mask).sum().item() / max(act_mask.sum().item(), 1),
        "r_cache": (r_cache * act_mask).sum().item() / max(act_mask.sum().item(), 1),
        "reward": (rewards.sum() / B).item(),
        "is_weight": (w * act_mask).sum().item() / max(act_mask.sum().item(), 1),
        "cache_hit_rate": hit_rate,
        "baseline": baseline_state["value"],
        "gen_tokens": act_mask.sum().item() / B,
    }
    return loss, stats


@torch.no_grad()
def evaluate(model, raw_model, teacher_forward, test_loader, step, args,
             accelerator, n_batches=4):
    baseline_state = {"value": 0.0}
    agg, n = {}, 0
    for i, (ids, mask) in enumerate(test_loader):
        if i >= n_batches:
            break
        ids, mask = ids.to(accelerator.device), mask.to(accelerator.device)
        full_ids, full_mask, action_mask = mixture_rollout(
            raw_model, teacher_forward, ids, mask, args.gen_len,
            args.mix_tau, args.eos_id, args.pad_id)
        _, stats = compute_reinforce_loss(
            model, raw_model, teacher_forward, full_ids, full_mask,
            action_mask, args, baseline_state)
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    if accelerator.is_main_process and n > 0:
        metrics = {f"eval/{k}": v / n for k, v in agg.items()}
        wandb.log(metrics, step=step)
        print(f"eval (step {step}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE finetuning of an MoE LLM with KD + expert-cache rewards"
    )
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Optional separate teacher (default: frozen base "
                             "via disabled LoRA adapters)")
    parser.add_argument("--dataset", type=str,
                        default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-split", type=str, default="math,code",
                        help="Comma-separated Nemotron splits")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", type=str, default="constant_with_warmup",
                        choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-cache-reinforce")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)

    rl_group = parser.add_argument_group("REINFORCE / rewards")
    rl_group.add_argument("--alpha", type=float, default=0.5,
                          help="Weight of r^BC; (1-alpha) weights r^cache")
    rl_group.add_argument("--mix-tau", type=float, default=0.2,
                          help="Teacher fraction of the sampling mixture p_mix "
                               "(bounds IS weights by 1/(1-tau))")
    rl_group.add_argument("--gamma", type=float, default=1.0)
    rl_group.add_argument("--baseline-ema", type=float, default=0.9,
                          help="EMA momentum of the scalar return baseline")
    rl_group.add_argument("--is-clip", type=float, default=0.0,
                          help="Clip importance weights at this value (0 = off; "
                               "already bounded by 1/(1-tau))")

    cache_group = parser.add_argument_group("Expert LRU cache")
    cache_group.add_argument("--cache-size", type=int, default=16,
                             help="Number of experts in the working set")
    cache_group.add_argument("--cache-layer", type=int, default=-1,
                             help="MoE layer whose router is cache-simulated "
                                  "(-1 = middle layer)")
    cache_group.add_argument("--cache-experts-per-token", type=int, default=1,
                             help="Experts drawn per token for the cache reward")
    cache_group.add_argument("--cache-topk", action="store_true",
                             help="Use router top-k instead of sampling e_t ~ G(x_t)")

    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, default=16)
    lora_group.add_argument("--lora-alpha", type=int, default=16)
    lora_group.add_argument("--lora-dropout", type=float, default=0.0,
                            help="Must be 0 when targeting the fused expert "
                                 "params (peft ParamWrapper forbids dropout)")
    lora_group.add_argument("--lora-target-modules", nargs="+",
                            default=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj",
                                     "gate"],
                            help="Attention + all experts + router. 'gate' "
                                 "matches only the router mlp.gate; 'up_proj'/"
                                 "'down_proj' match the fused OlmoeExperts "
                                 "params (per-expert batched LoRA)")

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
    accelerator.init_trackers(args.wandb_project, config=vars(args),
                              init_kwargs={"wandb": wandb_kwargs})
    if accelerator.is_main_process:
        wandb.config.update({"num_gpus": accelerator.num_processes})

    accelerator.print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation with batched prompts
    args.eos_id = tokenizer.eos_token_id
    args.pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    num_layers = model.config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    accelerator.print(f"[cache] LRU size={args.cache_size} on router of layer "
                      f"{args.cache_layer}/{num_layers}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # needed for grad ckpt with frozen embeddings
    model.gradient_checkpointing_enable()

    external_teacher = None
    if args.teacher_model:
        accelerator.print(f"Loading teacher {args.teacher_model} ...")
        external_teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(accelerator.device)
        external_teacher.eval()
        for p in external_teacher.parameters():
            p.requires_grad = False

    total, trainable = count_params(model)
    accelerator.print(f"Total parameters: {total:,}")
    accelerator.print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

    accelerator.print(f"Loading dataset {args.dataset} [{args.dataset_split}] ...")
    # One loader batch = one optimizer step: the whole batch is rolled out in
    # a single decode loop (decode is latency-bound, so batching it is ~free),
    # then split into `gradient_accumulation_steps` chunks for the grad passes.
    rollout_batch = args.batch_size * args.gradient_accumulation_steps
    loader, test_loader = get_nemotron_prompt_loaders(
        tokenizer, args.prompt_len, rollout_batch, args.dataset,
        args.dataset_split, args.max_samples, args.seed,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.num_epochs
    if args.num_steps:
        total_steps = min(total_steps, args.num_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    accelerator.print(f"LR scheduler: {args.lr_scheduler}, "
                      f"warmup={warmup_steps}/{total_steps} steps")

    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler,
    )
    raw_model = accelerator.unwrap_model(model)
    teacher_forward = make_teacher_forward(raw_model, external_teacher)

    start_epoch, start_step = 0, 0
    if resume_ckpt:
        try:
            meta = json.loads((resume_ckpt / "meta.json").read_text())
            start_step = meta["step"]
            start_epoch = meta.get("epoch", 0)
            accelerator.print(f"Restoring adapter weights from {resume_ckpt} ...")
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sf_path = resume_ckpt / "adapter" / "adapter_model.safetensors"
            if sf_path.exists():
                adapter_state = load_file(str(sf_path))
            else:
                adapter_state = torch.load(
                    resume_ckpt / "adapter" / "adapter_model.bin",
                    map_location="cpu", weights_only=True)
            set_peft_model_state_dict(raw_model, adapter_state)
            if (resume_ckpt / "optimizer.pt").exists():
                optimizer.load_state_dict(torch.load(
                    resume_ckpt / "optimizer.pt", map_location="cpu",
                    weights_only=True))
            if (resume_ckpt / "scheduler.pt").exists():
                lr_scheduler.load_state_dict(torch.load(
                    resume_ckpt / "scheduler.pt", map_location="cpu",
                    weights_only=True))
            accelerator.print(f"Resumed at step={start_step}, epoch={start_epoch}")
        except (RuntimeError, FileNotFoundError) as e:
            accelerator.print(f"[WARNING] Failed to load checkpoint {resume_ckpt}: {e}")
            start_step, start_epoch = 0, 0

    if accelerator.is_main_process:
        wandb.log({"total_params": total, "trainable_params": trainable}, step=0)

    _preempt_state = {"step": start_step, "epoch": start_epoch}

    def _preemption_handler(signum, frame):
        try:
            s, e = _preempt_state["step"], _preempt_state["epoch"]
            accelerator.print(f"\n[preemption] Signal {signum} at step {s}, saving...")
            if args.save_dir:
                save_checkpoint(model, tokenizer, args.save_dir, s, e, args,
                                accelerator, optimizer=optimizer,
                                lr_scheduler=lr_scheduler)
            accelerator.end_training()
        except (BrokenPipeError, KeyboardInterrupt):
            os._exit(130)  # second Ctrl-C / dead pipes: bail out quietly
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _preemption_handler)
    signal.signal(signal.SIGTERM, _preemption_handler)
    signal.signal(signal.SIGINT, _preemption_handler)  # Ctrl-C: same clean path

    model.train()
    step = start_step
    baseline_state = {"value": None}

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == start_epoch and start_step > 0:
            accelerator.print(f"Skipping {start_step} batches to resume epoch {epoch} ...")
            active_loader = accelerator.skip_first_batches(loader, start_step)
        else:
            active_loader = loader

        pbar = tqdm(active_loader, desc=f"epoch {epoch}", leave=False,
                    disable=not accelerator.is_main_process)

        for prompt_ids, prompt_mask in pbar:
            t0 = time.perf_counter()
            full_ids, full_mask, action_mask = mixture_rollout(
                raw_model, teacher_forward, prompt_ids, prompt_mask,
                args.gen_len, args.mix_tau, args.eos_id, args.pad_id)
            rollout_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            losses, chunk_stats = [], []
            for c in range(args.gradient_accumulation_steps):
                sl = slice(c * args.batch_size, (c + 1) * args.batch_size)
                with accelerator.accumulate(model):
                    loss, stats = compute_reinforce_loss(
                        model, raw_model, teacher_forward, full_ids[sl],
                        full_mask[sl], action_mask[sl], args, baseline_state)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
                chunk_stats.append(stats)
            grad_s = time.perf_counter() - t0

            step += 1
            _preempt_state["step"] = step
            _preempt_state["epoch"] = epoch

            mean_loss = sum(losses) / len(losses)
            stats = {k: sum(s[k] for s in chunk_stats) / len(chunk_stats)
                     for k in chunk_stats[0]}

            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{mean_loss:.4f}",
                                 hit=f"{stats['cache_hit_rate']:.2f}", step=step)
                if step % args.log_every == 0:
                    log_dict = {
                        "train/loss": mean_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/rollout_s": rollout_s,
                        "train/grad_s": grad_s,
                    }
                    log_dict.update({f"train/{k}": v for k, v in stats.items()})
                    wandb.log(log_dict, step=step)

            if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.save_dir, step, epoch,
                                args, accelerator, optimizer=optimizer,
                                lr_scheduler=lr_scheduler)

            if step % args.eval_every == 0:
                model.eval()
                evaluate(model, raw_model, teacher_forward, test_loader, step,
                         args, accelerator)
                model.train()

            if args.num_steps and step >= args.num_steps:
                break
        if args.num_steps and step >= args.num_steps:
            break

    accelerator.print(f"\nTraining done ({step} steps).")

    if args.save_dir:
        save_checkpoint(model, tokenizer, args.save_dir, step, epoch, args,
                        accelerator, optimizer=optimizer,
                        lr_scheduler=lr_scheduler)
        if accelerator.is_main_process:
            (Path(args.save_dir) / "COMPLETED").write_text(f"step={step}\n")

    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        # Interrupt landed outside the signal handler (e.g. during teardown);
        # suppress the BrokenPipeError cascade from tqdm/wandb streams.
        try:
            sys.stderr.write("\n[interrupt] exiting.\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(130)
