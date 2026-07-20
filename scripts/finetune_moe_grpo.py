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
from transformers import AutoTokenizer, TrainerCallback
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


EVAL_POOL_PER_SPLIT = 1000  # first rows of each split reserved for eval


def build_prompt_dataset(tokenizer, dataset_name, split, max_samples,
                         prompt_len, seed, skip_first=EVAL_POOL_PER_SPLIT):
    """Nemotron rows -> Dataset({'prompt'}) with prompts pre-truncated to
    prompt_len tokens (TRL 1.8 has no max_prompt_length). The first
    `skip_first` rows of each split are reserved for the perplexity eval."""
    from datasets import load_dataset

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
    use_chat = tokenizer.chat_template is not None
    prompts = []
    for sp in splits:
        ds = load_dataset(dataset_name, split=sp, streaming=True)
        n = 0
        for r_idx, row in enumerate(ds):
            if r_idx < skip_first:
                continue
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


def build_eval_sequences(tokenizer, dataset_name, split, n_total, max_len,
                         seed, pool_per_split=EVAL_POOL_PER_SPLIT):
    """Held-out full conversations (chat template incl. reference answer),
    truncated to max_len tokens, sampled from the reserved eval pool."""
    import random
    from datasets import load_dataset

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = n_total // len(splits)
    rng = random.Random(seed)
    use_chat = tokenizer.chat_template is not None
    eval_ids = []
    for sp in splits:
        ds = load_dataset(dataset_name, split=sp, streaming=True)
        pool = []
        for row in ds:
            msgs = [m for m in row["messages"] if m["content"].strip()]
            if any(m["role"] == "assistant" for m in msgs):
                pool.append(msgs)
            if len(pool) >= pool_per_split:
                break
        for msgs in rng.sample(pool, min(per_split, len(pool))):
            if use_chat:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            else:
                text = "\n".join(m["content"] for m in msgs)
            ids = tokenizer(text, truncation=True, max_length=max_len,
                            add_special_tokens=False)["input_ids"]
            eval_ids.append(ids)
    return eval_ids


class PerplexityCallback(TrainerCallback):
    """Every `every` optimizer steps, computes teacher-forced perplexity of
    the policy on the held-out sequences (rank 0 only) and logs eval/ppl.
    The frozen base's perplexity is logged once as eval/ppl_base."""

    def __init__(self, eval_ids, pad_id, every=25, batch_size=16):
        self.eval_ids = eval_ids
        self.pad_id = pad_id
        self.every = every
        self.bs = batch_size
        self.trainer = None
        self._base_ppl = None

    def attach(self, trainer):
        self.trainer = trainer

    @torch.no_grad()
    def _ppl(self, model):
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        total_nll, total_tok = 0.0, 0
        for i in range(0, len(self.eval_ids), self.bs):
            chunk = self.eval_ids[i:i + self.bs]
            L = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, x in enumerate(chunk):
                ids[j, :len(x)] = torch.tensor(x, dtype=torch.long)
                mask[j, :len(x)] = 1
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask,
                           use_cache=False).logits
            lp = F.log_softmax(logits[:, :-1].float(), -1) \
                .gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].bool()
            total_nll += -lp[m].sum().item()
            total_tok += int(m.sum())
        if was_training:
            model.train()
        import math
        return math.exp(total_nll / max(total_tok, 1))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0:
            return
        tr = self.trainer
        if tr is None:
            return
        # ALL ranks run the (identical) eval so no rank lags the others into
        # an NCCL collective timeout; only rank 0 logs. Redundant compute,
        # but rank-divergent multi-minute work deadlocks DDP.
        model = tr.accelerator.unwrap_model(tr.model)
        logs = {"eval/ppl": self._ppl(model)}
        if self._base_ppl is None:
            with model.disable_adapter():
                self._base_ppl = self._ppl(model)
        logs["eval/ppl_base"] = self._base_ppl
        if tr.accelerator.is_main_process:
            # Log straight to wandb: trainer.log() would trigger TRL's
            # completions-table upload mid-step (rank-0 stall -> NCCL
            # timeout), and wandb.log(step=...) with a backdated step is
            # silently dropped. No explicit step; global_step goes along as
            # a field for the x-axis.
            import wandb
            if wandb.run is not None:
                wandb.log({**logs, "train/global_step": state.global_step})
            print(f"[eval] step {state.global_step}: ppl {logs['eval/ppl']:.3f} "
                  f"(base {self._base_ppl:.3f})", flush=True)


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
                use_topk=args.cache_topk, soft=args.soft_cache,
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
    parser.add_argument("--eval-ppl-seqs", type=int, default=256,
                        help="Held-out sequences for the perplexity eval")
    parser.add_argument("--eval-ppl-every", type=int, default=25,
                        help="Optimizer steps between perplexity evals")

    rl_group = parser.add_argument_group("Rewards")
    rl_group.add_argument("--alpha", type=float, default=0.0,
                          help="Weight of the r^BC reward; (1-alpha) weights "
                               "the cache reward. Prefer --beta and alpha=0.")
    rl_group.add_argument("--beta", type=float, default=0.04,
                          help="TRL KL(policy||ref) coefficient (ref = "
                               "adapters disabled = frozen base)")
    rl_group.add_argument("--multi-objective-aggregation", type=str,
                          default="normalize_then_sum",
                          choices=["sum_then_normalize", "normalize_then_sum"],
                          help="normalize_then_sum z-scores each reward per "
                               "group before weighting -> weights are true "
                               "influence ratios (kd-scale becomes a no-op)")
    rl_group.add_argument("--loss-type", type=str, default="dapo",
                          help="TRL GRPO loss variant (dapo = TRL 1.8 default)")
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
    cache_group.add_argument("--soft-cache", action="store_true",
                             help="Dense routing at the cache layer (K = all "
                                  "experts, full-softmax weights) + soft cache "
                                  "reward: router probability mass on the "
                                  "cached experts. LRU still touched by the "
                                  "top-k (--cache-experts-per-token) experts")

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
    # Serialize dataset streaming across ranks: concurrent access to the
    # shared HF cache can fail with "[Errno 16] Device or resource busy".
    # Additionally cache the built prompts on disk so concurrent sweep jobs
    # don't all re-stream from HF (observed 504s killing sibling jobs).
    import hashlib, pickle
    key = hashlib.md5(str((args.dataset, args.dataset_split, args.max_samples,
                           args.prompt_len, args.completion_len,
                           args.eval_ppl_seqs, args.seed, args.model)).encode()
                      ).hexdigest()[:12]
    cache_file = Path("data") / f"prompt_cache_{key}.pkl"
    from accelerate import PartialState
    with PartialState().main_process_first():
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                prompts_list, eval_ids = pickle.load(f)
            train_dataset = Dataset.from_dict({"prompt": prompts_list})
            print(f"[data] loaded cached prompts from {cache_file}")
        else:
            train_dataset = build_prompt_dataset(
                tokenizer, args.dataset, args.dataset_split, args.max_samples,
                args.prompt_len, args.seed)
            eval_ids = build_eval_sequences(
                tokenizer, args.dataset, args.dataset_split, args.eval_ppl_seqs,
                args.prompt_len + args.completion_len, args.seed)
            if PartialState().is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump((list(train_dataset["prompt"]), eval_ids), f)
                print(f"[data] cached prompts to {cache_file}")
    print(f"[eval] {len(eval_ids)} held-out sequences "
          f"(<= {args.prompt_len + args.completion_len} tokens each)")

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
        ddp_timeout=3600,  # headroom for the synchronized perplexity eval
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
        loss_type=args.loss_type,
        multi_objective_aggregation=args.multi_objective_aggregation,
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

    ppl_cb = PerplexityCallback(eval_ids, tokenizer.pad_token_id,
                                every=args.eval_ppl_every)
    ppl_cb.attach(trainer)
    trainer.add_callback(ppl_cb)

    model = trainer.accelerator.unwrap_model(trainer.model)
    num_layers = model.config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    print(f"[cache] LRU size={args.cache_size} on router of layer "
          f"{args.cache_layer}/{num_layers}")

    if args.soft_cache:
        # Dense routing at the cache layer: every expert is active with its
        # full-softmax weight (sparsemixer only supports iterative top-1/2).
        # Patched on the instance post-peft, so it applies to rollouts, the
        # scoring pass, and the adapter-disabled ref/teacher forwards alike.
        # self.weight is the LoRA-merged router weight inside ParamWrapper's
        # forward, and full softmax keeps routing differentiable end-to-end.
        if model.config.model_type != "phimoe":
            raise ValueError("--soft-cache is only wired up for phimoe")
        if args.temporal:
            raise ValueError("--soft-cache is incompatible with --temporal")
        import types

        def _dense_router_forward(self, hidden_states):
            router_logits = F.linear(hidden_states, self.weight, self.bias)
            routing_weights = torch.softmax(router_logits.float(), dim=-1) \
                .to(hidden_states.dtype)
            selected = torch.arange(router_logits.shape[-1],
                                    device=router_logits.device) \
                .expand(router_logits.shape[0], -1)
            return router_logits, routing_weights, selected

        patched = []
        for name, mod in model.named_modules():
            if (type(mod).__name__ == "PhimoeTopKRouter"
                    and f"layers.{args.cache_layer}.mlp" in name):
                mod.forward = types.MethodType(_dense_router_forward, mod)
                patched.append(name)
        if len(patched) != 1:
            raise RuntimeError(
                f"expected exactly 1 router at layer {args.cache_layer}, "
                f"patched {patched}")
        print(f"[cache] dense routing (K=all experts) patched on {patched[0]}; "
              f"soft cache reward = router prob mass on cached experts")

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
