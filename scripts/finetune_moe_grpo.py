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
  * SFT NLL loss (--sft-coef), added directly to the loss, NOT mixed in as a
    reward: total_loss = policy_loss (GRPO/DAPO) + sft_coef * NLL(target),
    where NLL(target) is the standard teacher-forced cross-entropy of the
    policy on the dataset's own ground-truth completion (GRPOTrainerWithSFT).
    This replaces the former KD/BC reward (log p_ref - log p_policy on the
    target), which only anchored p_policy(target) to the frozen base model's
    *original* value; it never pushed the likelihood of the target sequence
    up. Directly minimizing NLL(target) does.
  * TRL's native --beta: per-token KL(policy || ref) penalty in the loss
    (ref = adapters disabled), independent of the SFT term above.
  * On-policy generation replaces mixture sampling: the anti-hacking role of
    p_mix is covered by GRPO's clipped importance ratio and the beta-KL.

Not ported from the REINFORCE version: the router log-prob REINFORCE term
(the router only receives gradient through the token log-probs here).
"""

import argparse
import signal
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
                         prompt_len, completion_len, seed,
                         skip_first=EVAL_POOL_PER_SPLIT):
    """Nemotron rows -> Dataset({'prompt', 'target_ids'}).

    'prompt' is pre-truncated to prompt_len tokens (TRL 1.8 has no
    max_prompt_length). 'target_ids' is the dataset's own ground-truth
    assistant response, tokenized and truncated to completion_len tokens --
    used only for the SFT NLL loss term (GRPOTrainerWithSFT), never for
    reward computation or generation. The first `skip_first` rows of each
    split are reserved for the perplexity eval.

    Uses reservoir sampling (Algorithm R) within each split so the result is
    a genuine random sample, not a fixed prefix -- capped at a scan window
    (see MAX_SCAN_PER_SPLIT) so very large splits (e.g. "chat") don't have to
    be streamed to completion just to draw a few hundred rows from them.
    Tokenization happens AFTER the reservoir is finalized, not during the
    scan: doing it eagerly per scanned row cost up to MAX_SCAN_PER_SPLIT x
    num_splits BPE calls for a result that only keeps per_split x num_splits
    of them -- with 9 splits x 50k that's ~450k wasted tokenizer calls,
    which single-handedly blew well past accelerate's 600s multi-GPU
    rendezvous timeout (rank 0 stuck tokenizing while other ranks waited and
    gave up).
    """
    from datasets import load_dataset
    import random

    MAX_SCAN_PER_SPLIT = 50_000  # bounds streaming cost on large splits
    rng = random.Random(seed)
    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
    use_chat = tokenizer.chat_template is not None
    prompts, targets = [], []
    for sp in splits:
        ds = load_dataset(dataset_name, split=sp, streaming=True)
        reservoir = []  # raw (text, target_text) strings, untokenized
        seen = 0
        scanned = 0
        for r_idx, row in enumerate(ds):
            if r_idx < skip_first:
                continue
            if scanned >= max(MAX_SCAN_PER_SPLIT, per_split):
                break
            scanned += 1
            parts = []
            target_text = None
            for m in row["messages"]:
                if m["role"] == "assistant":
                    target_text = m["content"]
                    break
                if m["content"].strip():
                    parts.append(m["content"])
            text = "\n".join(parts).strip()
            if not (text and target_text and target_text.strip()):
                continue
            item = (text, target_text.strip())
            if len(reservoir) < per_split:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen)
                if j < per_split:
                    reservoir[j] = item
            seen += 1
        for text, target_text in reservoir:
            ids = tokenizer(text, truncation=True,
                            max_length=prompt_len)["input_ids"]
            text = tokenizer.decode(ids)
            # conversational format -> TRL applies the chat template
            prompt_repr = [{"role": "user", "content": text}] if use_chat else text
            target_ids = tokenizer(target_text, truncation=True,
                                   max_length=completion_len,
                                   add_special_tokens=False)["input_ids"]
            prompts.append(prompt_repr)
            targets.append(target_ids)
    return Dataset.from_dict({"prompt": prompts, "target_ids": targets}) \
        .shuffle(seed=seed)


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


class PreemptionCallback(TrainerCallback):
    """SLURM preemption/requeue: SIGUSR1 (sent --signal seconds before the
    time limit, or on preemption of a --requeue job) and SIGTERM just set a
    flag; the Trainer's own callback loop -- not the signal handler itself,
    which must stay async-signal-safe -- checkpoints (optimizer/scheduler/
    RNG state included, so --resume restores exactly) and stops training at
    the next step boundary. Pairs with --resume + a fixed --save-dir: SLURM
    requeues the same job, which reruns this script and picks the latest
    checkpoint back up via get_last_checkpoint()."""

    def __init__(self):
        self._triggered = False
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        print(f"[preemption] signal {signum} received -- checkpointing and "
              f"stopping at the next step boundary", flush=True)
        self._triggered = True

    def on_step_end(self, args, state, control, **kwargs):
        if self._triggered:
            control.should_save = True
            control.should_training_stop = True
        return control


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

        if was_training:
            model.train()
        return {"cache": cache_rewards.cpu().tolist(), "hit_rate": hit_rate}

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


class GRPOTrainerWithSFT(GRPOTrainer):
    """GRPOTrainer + a standard teacher-forced NLL loss on the dataset's own
    ground-truth completion ('target_ids', see build_prompt_dataset), added
    directly to the total loss:

        total_loss = rl_coef * policy_loss (GRPO/DAPO) + sft_coef * NLL(target)

    This replaces the KD/BC reward (log p_ref - log p_policy on the target,
    mixed into the group-relative reward): that term only anchored
    p_policy(target) to the frozen base model's *original* likelihood, it
    never increased it. Directly minimizing NLL(target) does, independent of
    the cache reward's advantage signal. rl_coef lets the policy loss be
    scaled relative to the SFT term (e.g. up-weighted if SFT dominates the
    gradient and drives KL/entropy drift without cache-reward gains).
    """

    def __init__(self, *args, sft_coef=0.0, rl_coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_coef = sft_coef
        self.rl_coef = rl_coef

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.sft_coef <= 0:
            return output
        pad_id = self.processing_class.pad_token_id
        target_ids_list = [x["target_ids"] for x in inputs]
        L = max((len(t) for t in target_ids_list), default=1) or 1
        B = len(target_ids_list)
        target_ids = torch.full((B, L), pad_id, dtype=torch.long)
        target_mask = torch.zeros((B, L), dtype=torch.long)
        for i, t in enumerate(target_ids_list):
            if t:
                target_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
                target_mask[i, :len(t)] = 1
        output["target_ids"] = target_ids
        output["target_mask"] = target_mask
        return output

    def _sft_nll_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        device = prompt_ids.device
        target_ids = inputs["target_ids"].to(device)
        target_mask = inputs["target_mask"].to(device)
        # prompt_ids: left-padded (TRL generation convention); target_ids:
        # right-padded. Concatenating them mirrors exactly how GRPOTrainer
        # itself scores prompt+completion, so the same attention-mask-derived
        # position handling applies.
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        full_mask = torch.cat([prompt_mask, target_mask], dim=1)
        logits = model(input_ids=full_ids, attention_mask=full_mask,
                      use_cache=False).logits
        P = prompt_ids.size(1)
        pred_logits = logits[:, P - 1:-1]  # predict target_ids[:, t] from position P-1+t
        logp = F.log_softmax(pred_logits.float(), dim=-1) \
            .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        mask = target_mask.float()
        return -(logp * mask).sum() / mask.sum().clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        total = self.rl_coef * loss
        if self.sft_coef > 0 and "target_ids" in inputs:
            sft_loss = self._sft_nll_loss(model, inputs)
            mode = "train" if model.training else "eval"
            self._metrics[mode]["sft_nll"].append(sft_loss.item())
            total = total + self.sft_coef * sft_loss
        return total


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
    rl_group.add_argument("--rl-coef", type=float, default=1.0,
                          help="Weight of the GRPO/DAPO policy loss: "
                               "total_loss = rl_coef * policy_loss + "
                               "sft_coef * NLL(target). Raise this (or lower "
                               "--sft-coef) if the SFT term dominates the "
                               "gradient and the policy drifts (rising KL/"
                               "entropy) without cache-reward gains.")
    rl_group.add_argument("--sft-coef", type=float, default=1.0,
                          help="Weight of the SFT NLL loss, added directly to "
                               "the policy loss (NOT mixed into the reward): "
                               "total_loss = rl_coef * policy_loss + sft_coef "
                               "* NLL(target), target = the dataset's own "
                               "ground-truth completion. 0 disables it. "
                               "Replaces the old KD/BC reward.")
    rl_group.add_argument("--beta", type=float, default=0.04,
                          help="TRL KL(policy||ref) coefficient (ref = "
                               "adapters disabled = frozen base)")
    rl_group.add_argument("--multi-objective-aggregation", type=str,
                          default="normalize_then_sum",
                          choices=["sum_then_normalize", "normalize_then_sum"],
                          help="normalize_then_sum z-scores each reward per "
                               "group before weighting -> weights are true "
                               "influence ratios")
    rl_group.add_argument("--loss-type", type=str, default="dapo",
                          help="TRL GRPO loss variant (dapo = TRL 1.8 default)")
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
                           args.eval_ppl_seqs, args.seed, args.model,
                           "v2_target_ids")).encode()  # bump on cache schema changes
                      ).hexdigest()[:12]
    cache_file = Path("data") / f"prompt_cache_{key}.pkl"
    from accelerate import PartialState
    with PartialState().main_process_first():
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                prompts_list, targets_list, eval_ids = pickle.load(f)
            train_dataset = Dataset.from_dict(
                {"prompt": prompts_list, "target_ids": targets_list})
            print(f"[data] loaded cached prompts from {cache_file}")
        else:
            train_dataset = build_prompt_dataset(
                tokenizer, args.dataset, args.dataset_split, args.max_samples,
                args.prompt_len, args.completion_len, args.seed)
            eval_ids = build_eval_sequences(
                tokenizer, args.dataset, args.dataset_split, args.eval_ppl_seqs,
                args.prompt_len + args.completion_len, args.seed)
            if PartialState().is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump((list(train_dataset["prompt"]),
                                list(train_dataset["target_ids"]), eval_ids), f)
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
    use_grad_checkpointing = not (args.temporal and not temporal_config.ste) \
        if args.temporal else True

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
    reward_weights = [1.0]

    grpo_config = GRPOConfig(
        output_dir=args.save_dir,
        run_name=args.wandb_run_name,
        report_to=["wandb"],
        seed=args.seed,
        ddp_timeout=3600,  # headroom for the synchronized perplexity eval
        bf16=True,
        # Gradient checkpointing's no-grad first pass would detach the
        # temporal ratio/entropy loss (computed as a forward side effect,
        # only present when not ste) -> only disable for the non-STE
        # boundary variant. The STE path is checkpointing-safe; leaving
        # checkpointing off unconditionally OOMs regardless of batch size,
        # since the fused-expert LoRA weight (W + delta_weight) is
        # materialized per MoE layer and stays resident across all layers
        # in one forward pass without it.
        gradient_checkpointing=use_grad_checkpointing,
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

    trainer = GRPOTrainerWithSFT(
        model=model_or_id,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        sft_coef=args.sft_coef,
        rl_coef=args.rl_coef,
    )
    engine.attach(trainer)

    ppl_cb = PerplexityCallback(eval_ids, tokenizer.pad_token_id,
                                every=args.eval_ppl_every)
    ppl_cb.attach(trainer)
    trainer.add_callback(ppl_cb)
    trainer.add_callback(PreemptionCallback())

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
