"""GRPO finetuning (TRL) of an MoE LLM with the expert-cache reward.

Maps the REINFORCE setup (scripts/finetune_moe_reinforce.py) onto TRL's
GRPOTrainer:

  * Advantages: group-relative (G completions per prompt, reward centered
    within the group) — replaces the scalar EMA baseline.
  * Cache reward: sequence-level hit fraction in [0, 1] — the sum over
    generated tokens of 1[e_t in C]/T, i.e. exactly the per-sequence return
    of the REINFORCE cache reward. Computed by a custom reward function that
    runs one no-grad forward with output_router_logits and simulates the LRU
    working set (src.cache_reinforce.cache_emulation_rewards).
  * KD/BC anchor to the frozen base model, two equivalent knobs:
      --alpha  weights a sequence-level r^BC reward (mean per-token
               log p_ref - log p_policy) against the cache reward,
               mirroring r = alpha*r^BC + (1-alpha)*r^cache;
      --beta   TRL's native per-token KL(policy || ref) penalty in the loss
               (ref = adapters disabled). Prefer beta; keep alpha=0.
  * On-policy generation replaces mixture sampling: the anti-hacking role of
    p_mix is covered by GRPO's clipped importance ratio and the beta-KL.

Not ported from the REINFORCE version: the router log-prob REINFORCE term
(the router only receives gradient through the token log-probs here).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from src.cache_reinforce import cache_emulation_rewards
from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin

# peft 0.19 x transformers 5.8 bug: on adapter-checkpoint load, peft's v4->v5
# key conversion calls WeightConverter with a removed kwarg
# ('distributed_operation') and crashes. Our checkpoints are saved by this
# same environment and already use v5 keys, so the conversion is a no-op --
# bypass it unless legacy-format keys are actually present.
import peft.utils.transformers_weight_conversion as _pwc

_orig_convert = _pwc.convert_peft_adapter_state_dict_for_transformers

def _convert_only_if_legacy(model, peft_config, adapter_state_dict, adapter_name):
    legacy = any(".w1." in k or ".w2." in k or ".w3." in k
                 or "block_sparse_moe" in k for k in adapter_state_dict)
    if not legacy:
        return adapter_state_dict
    return _orig_convert(model=model, peft_config=peft_config,
                         adapter_state_dict=adapter_state_dict,
                         adapter_name=adapter_name)

_pwc.convert_peft_adapter_state_dict_for_transformers = _convert_only_if_legacy


def build_prompt_dataset(tokenizer, dataset_name, split, max_samples,
                         prompt_len, seed):
    """Nemotron rows -> Dataset({'prompt'}) with prompts pre-truncated to
    prompt_len tokens (TRL 1.8 has no max_prompt_length)."""
    from datasets import load_dataset

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
    use_chat = tokenizer.chat_template is not None
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
                ids = tokenizer(text, truncation=True,
                                max_length=prompt_len)["input_ids"]
                text = tokenizer.decode(ids)
                # conversational format -> TRL applies the chat template
                prompts.append([{"role": "user", "content": text}]
                               if use_chat else text)
                n += 1
            if n >= per_split:
                break
    return Dataset.from_dict({"prompt": prompts}).shuffle(seed=seed)


class RewardEngine:
    """Computes cache and KD rewards for a batch of GRPO completions.

    Both reward functions share one memoized scoring pass: policy forward
    with output_router_logits (router logits at the cache layer + policy
    log-probs), plus an adapter-disabled ref forward when KD is needed.
    """

    def __init__(self, args, temporal_wrappers=None):
        self.args = args
        self.trainer = None
        # With temporal routing, the cache reward reads each token's
        # *effective held* expert decisions from the wrapper at the cache
        # layer (stored by its forward), not the raw router logits.
        self.temporal_wrappers = temporal_wrappers
        self._memo_key = None
        self._memo = None

    def attach(self, trainer):
        self.trainer = trainer

    # -- shared scoring pass -------------------------------------------------

    @torch.no_grad()
    def _compute(self, prompts, completion_ids):
        args = self.args
        tok = self.trainer.processing_class
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        device = self.trainer.accelerator.device
        pad_id = tok.pad_token_id

        def encode_prompt(p):
            if not isinstance(p, str):  # conversational -> render template
                p = tok.apply_chat_template(p, add_generation_prompt=True,
                                            tokenize=False)
            return tok(p, truncation=True, max_length=args.prompt_len,
                       add_special_tokens=False)["input_ids"]

        prompt_ids = [encode_prompt(p) for p in prompts]
        seqs = [torch.tensor(p + list(c), dtype=torch.long)
                for p, c in zip(prompt_ids, completion_ids)]
        B = len(seqs)
        S = max(len(s) for s in seqs)
        full_ids = torch.full((B, S), pad_id, dtype=torch.long)
        valid = torch.zeros(B, S, dtype=torch.bool)
        action = torch.zeros(B, S, dtype=torch.bool)  # completion positions
        for i, (s, p) in enumerate(zip(seqs, prompt_ids)):
            full_ids[i, :len(s)] = s                  # right padding
            valid[i, :len(s)] = True
            action[i, len(p):len(s)] = True
        full_ids, valid, action = (t.to(device) for t in (full_ids, valid, action))

        was_training = model.training
        model.eval()
        out = model(input_ids=full_ids, attention_mask=valid.long(),
                    output_router_logits=True, use_cache=False)
        if self.temporal_wrappers is not None:
            # per-token decisions actually executed (held across segments)
            held = self.temporal_wrappers[args.cache_layer]._last_top_k_indices
            r_cache_tok, _, hit_rate = cache_emulation_rewards(
                None, valid, action, cache_size=args.cache_size,
                experts=held,
            )
        else:
            router_logits = out.router_logits[args.cache_layer].view(B, S, -1)
            r_cache_tok, _, hit_rate = cache_emulation_rewards(
                router_logits, valid, action, cache_size=args.cache_size,
                experts_per_token=args.cache_experts_per_token,
                use_topk=args.cache_topk,
            )
        cache_rewards = r_cache_tok.sum(-1)  # per-seq hit fraction in [0, 1]

        kd_rewards = torch.zeros(B)
        if args.alpha > 0:
            act_t = action[:, 1:]  # target grid: logits at i predict token i+1
            targets = full_ids[:, 1:]
            lp_pol = F.log_softmax(out.logits[:, :-1].float(), -1) \
                .gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            with model.disable_adapter():
                ref_logits = model(input_ids=full_ids, attention_mask=valid.long(),
                                   use_cache=False).logits
            lp_ref = F.log_softmax(ref_logits[:, :-1].float(), -1) \
                .gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # mean per-token r^BC (length-invariant sequence reward)
            n_act = act_t.sum(-1).clamp(min=1)
            kd_rewards = (((lp_ref - lp_pol) * act_t).sum(-1) / n_act).cpu()
            kd_rewards = kd_rewards * args.kd_scale

        if was_training:
            model.train()
        return {"cache": cache_rewards.cpu().tolist(),
                "kd": kd_rewards.tolist(),
                "hit_rate": hit_rate}

    def _scores(self, prompts, completion_ids):
        key = (id(completion_ids), len(completion_ids),
               tuple(completion_ids[0][:4]) if len(completion_ids[0]) else ())
        if key != self._memo_key:
            self._memo = self._compute(prompts, completion_ids)
            self._memo_key = key
        return self._memo

    # -- reward functions passed to GRPOTrainer ------------------------------

    def cache_reward(self, prompts, completions, completion_ids,
                     log_metric=None, **kwargs):
        scores = self._scores(prompts, completion_ids)
        if log_metric is not None:
            log_metric("cache_hit_rate", scores["hit_rate"])
            if self.temporal_wrappers is not None:
                w = self.temporal_wrappers[self.args.cache_layer]
                log_metric("boundary_rate", float(w._last_F))
        return scores["cache"]

    def kd_reward(self, prompts, completions, completion_ids, **kwargs):
        return self._scores(prompts, completion_ids)["kd"]


def main():
    parser = argparse.ArgumentParser(
        description="GRPO finetuning of an MoE LLM with the expert-cache reward (TRL)"
    )
    parser.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    parser.add_argument("--dataset", type=str,
                        default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-split", type=str, default="math,code")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--completion-len", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=8,
                        help="G: completions per prompt (group size)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-device completions per step; global batch "
                             "must be divisible by --num-generations")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-epochs", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="LoRA RL lr: ~10x the ~1e-6..5e-6 full-FT GRPO range (DeepSeekMath; LoRA-without-regret 10x rule)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-cache-reinforce")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/grpo_olmoe_cache")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in --save-dir")

    rl_group = parser.add_argument_group("Rewards")
    rl_group.add_argument("--alpha", type=float, default=0.0,
                          help="Weight of the r^BC reward; (1-alpha) weights "
                               "the cache reward. Prefer --beta and alpha=0.")
    rl_group.add_argument("--beta", type=float, default=0.04,
                          help="TRL KL(policy||ref) coefficient (ref = "
                               "adapters disabled = frozen base)")
    rl_group.add_argument("--kd-scale", type=float, default=1.0,
                          help="Multiplier on the KD reward (KLReward reward_scale in rl_moe): reward = kd_scale * mean_t(log p_ref - log p_policy). Use to match the cache reward's within-group std (measured ~4x)")
    rl_group.add_argument("--router-aux-loss-coef", type=float, default=0.0,
                          help="MoE load-balancing aux loss coef. Keep 0: "
                               "balancing pushes uniform expert usage, the "
                               "opposite of cache consolidation")

    cache_group = parser.add_argument_group("Expert LRU cache")
    cache_group.add_argument("--cache-size", type=int, default=16)
    cache_group.add_argument("--cache-layer", type=int, default=-1,
                             help="-1 = middle layer")
    cache_group.add_argument("--cache-experts-per-token", type=int, default=1)
    cache_group.add_argument("--cache-topk", action="store_true")

    temporal_group = parser.add_argument_group("Temporal MoE (optional)")
    temporal_group.add_argument("--temporal", action="store_true",
                                help="Wrap MoE blocks with boundary prediction "
                                     "+ hold/switch routing (TemporalWrapMixin)")
    temporal_group.add_argument("--ratio-loss-N", type=int, nargs="+", default=[8],
                                help="Target segment length per MoE layer")
    temporal_group.add_argument("--ratio-loss-alpha", type=float, default=0.03)
    temporal_group.add_argument("--entropy-threshold", type=float, default=0.1)
    temporal_group.add_argument("--entropy-alpha", type=float, default=0.05)

    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-r", type=int, default=16)
    lora_group.add_argument("--lora-alpha", type=int, default=32)
    lora_group.add_argument("--lora-target-modules", nargs="+",
                            default=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj",
                                     "gate", "router"],
                            help="'gate' matches the OLMoE router, 'router' "
                                 "the PhiMoE one; unmatched names are ignored")

    args = parser.parse_args()

    import os
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building prompt dataset {args.dataset} [{args.dataset_split}] ...")
    train_dataset = build_prompt_dataset(
        tokenizer, args.dataset, args.dataset_split, args.max_samples,
        args.prompt_len, args.seed)

    # peft auto-converts classic expert names (gate_proj/up_proj/down_proj) to
    # fused target_parameters for registered archs (olmoe->qwen2_moe pattern),
    # but phimoe is missing from its registry -- target the fused 3D expert
    # params explicitly, doubling r/alpha on gate_up to keep per-branch scale.
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(args.model)
    target_modules = list(args.lora_target_modules)
    lora_kwargs = {}
    if model_cfg.model_type == "phimoe":
        # phimoe's router module returns a tuple, which LoRA's module wrapper
        # can't handle -> adapt its weight as a parameter instead. Same for
        # the fused 3D expert params; double r/alpha on gate_up to keep
        # per-branch scale.
        target_modules = [t for t in target_modules
                          if t not in ("router", "gate")]
        lora_kwargs = dict(
            target_parameters=["experts.gate_up_proj", "experts.down_proj",
                               "router.weight"],
            rank_pattern={r".*\.gate_up_proj": args.lora_r * 2},
            alpha_pattern={r".*\.gate_up_proj": args.lora_alpha * 2},
        )
    # Temporal MoE: load the model ourselves, wrap the MoE blocks, and hand
    # the instance to TRL. The boundary-prediction layers (term_proj1/2) are
    # new non-LoRA params -> trained + checkpointed via peft modules_to_save.
    model_or_id = args.model
    temporal_wrappers = None
    if args.temporal:
        from transformers import AutoModelForCausalLM
        model_or_id = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        # STE boundary: no ratio/entropy regularizers -- the boundary
        # predictor learns from the GRPO objective alone, with gradients
        # flowing through the straight-through hold/switch decision.
        temporal_config = TemporalWrapConfig(ste=True)
        TemporalWrapMixin.apply(model_or_id, temporal_config)
        temporal_wrappers = model_or_id._moe_layers
        lora_kwargs["modules_to_save"] = ["term_proj1", "term_proj2"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # ParamWrapper (fused experts) forbids dropout
        target_modules=target_modules,
        **lora_kwargs,
    )

    engine = RewardEngine(args, temporal_wrappers=temporal_wrappers)
    reward_funcs = [engine.cache_reward]
    reward_weights = [1.0 - args.alpha]
    if args.alpha > 0:
        reward_funcs.append(engine.kd_reward)
        reward_weights.append(args.alpha)

    grpo_config = GRPOConfig(
        output_dir=args.save_dir,
        run_name=args.wandb_run_name,
        report_to=["wandb"],
        seed=args.seed,
        bf16=True,
        # Gradient checkpointing's no-grad first pass would detach the
        # temporal ratio/entropy loss (computed as a forward side effect),
        # zeroing the boundary predictor's gradient -> disable when temporal.
        gradient_checkpointing=not args.temporal,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.num_steps,
        logging_steps=1,
        log_completions=True,  # sample completions -> wandb table (quality check)
        num_completions_to_print=0,
        save_steps=args.save_every,
        save_total_limit=3,  # keep rewind points (reward hacking recovery)
        num_generations=args.num_generations,
        max_completion_length=args.completion_len,
        temperature=args.temperature,
        beta=args.beta,
        router_aux_loss_coef=args.router_aux_loss_coef,
        reward_weights=reward_weights,
        # No trust_remote_code: use transformers' native classes (Phi-tiny's
        # bundled remote code requires flash_attn and bypasses the peft/TRL
        # integration we rely on). Not allowed when passing a model instance.
        model_init_kwargs=None if args.temporal else {"dtype": torch.bfloat16},
    )

    trainer = GRPOTrainer(
        model=model_or_id,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    engine.attach(trainer)

    model = trainer.accelerator.unwrap_model(trainer.model)
    num_layers = model.config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    print(f"[cache] LRU size={args.cache_size} on router of layer "
          f"{args.cache_layer}/{num_layers}")

    resume_ckpt = None
    if args.resume and Path(args.save_dir).is_dir():
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(args.save_dir)
        if resume_ckpt:
            print(f"Resuming from {resume_ckpt}")

    if resume_ckpt:
        # peft 0.19 cannot load target_parameters (ParamWrapper) adapters via
        # set_peft_model_state_dict ('PhimoeExperts' has no attribute
        # 'weight'). Replace PeftModel.load_adapter -- which the HF Trainer
        # calls on resume -- with a direct load_state_dict of the remapped
        # adapter tensors. Optimizer/scheduler/trainer state still load
        # through the normal Trainer path.
        from safetensors.torch import load_file
        peft_model = trainer.model

        def _manual_load_adapter(ckpt_dir, adapter_name="default", **kwargs):
            sd = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
            model_keys = set(peft_model.state_dict().keys())
            remapped = {}
            for k, v in sd.items():
                nk = k.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight") \
                      .replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
                if nk not in model_keys:
                    # modules_to_save entries (e.g. term_proj*): saved without
                    # the wrapper infix -> '...modules_to_save.default.weight'
                    head, _, tail = nk.rpartition(".")
                    cand = f"{head}.modules_to_save.{adapter_name}.{tail}"
                    if cand in model_keys:
                        nk = cand
                remapped[nk] = v
            res = peft_model.load_state_dict(remapped, strict=False)
            if res.unexpected_keys:
                raise RuntimeError(
                    f"adapter resume failed, unexpected keys: {res.unexpected_keys[:5]}")
            missing_lora = [k for k in res.missing_keys if "lora" in k]
            if missing_lora:
                raise RuntimeError(
                    f"adapter resume failed, missing lora keys: {missing_lora[:5]}")
            print(f"[resume] manually loaded {len(remapped)} adapter tensors")

        peft_model.load_adapter = _manual_load_adapter

    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    main()
