"""GRPO finetuning (TRL) of an MoE LLM, cache size conditioned via a PROMPT
PREFIX rather than an out-of-band embedding.

Variant of scripts/train/finetune_moe_grpo.py:

  * No SFT term: pure GRPO/DAPO policy loss (no GRPOTrainerWithSFT, no
    --sft-coef/--rl-coef, no target_ids column).
  * TRL's native --beta (per-token KL(policy || ref), ref = adapters
    disabled) defaults to 0.04, same value finetune_moe_grpo.py uses.
  * Cache-size conditioning: every filtered training prompt is expanded into
    len(--prompt-cache-sizes) prompts (default cache sizes 2, 4, 8), each
    with a literal "[CACHE_SIZE=X]" prefix prepended to its first turn. The
    model has to learn to read the conditioning signal out of its own
    context. At reward time the prefix is parsed back out of each prompt in
    the batch (a batch is a shuffled mix of cache-size conditions) to pick
    which LRU capacity to simulate for THAT sequence -- the cache size is a
    per-example property of cache_emulation_rewards (src.cache_reinforce),
    not a per-batch/per-step one like --conditioned-cache-sizes in
    finetune_moe_grpo.py.

Not ported from finetune_moe_grpo.py: --temporal (boundary-prediction MoE
routing), --conditioned-cache-sizes (embedding-based size conditioning --
superseded here by the prompt-prefix mechanism), --sft-coef/--rl-coef,
--multi-objective-aggregation (only one reward function is registered).
"""

import argparse
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F

from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from src.cache_reinforce import cache_emulation_rewards


def _checkpoint_is_intact(ckpt_dir):
    """Best-effort integrity check for a Trainer checkpoint: verifies the
    safetensors adapter file's header parses and optimizer.pt/scheduler.pt
    (plain torch pickle/zip) fully deserialize. Catches the truncated-write
    corruption a disk-quota-exceeded save leaves behind (safetensors
    'HeaderTooLarge'/short-read, torch.load 'unexpected pos' zip errors)
    without doing a real load into the model."""
    ckpt_dir = Path(ckpt_dir)
    try:
        from safetensors import safe_open
        with safe_open(str(ckpt_dir / "adapter_model.safetensors"),
                        framework="pt") as f:
            list(f.keys())
        for fname in ("optimizer.pt", "scheduler.pt"):
            fpath = ckpt_dir / fname
            if fpath.is_file():
                torch.load(str(fpath), map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        print(f"[resume] checkpoint {ckpt_dir} failed integrity check "
              f"({type(e).__name__}: {e}) -- skipping")
        return False


def _find_valid_checkpoint(save_dir):
    """Like transformers.trainer_utils.get_last_checkpoint, but falls back
    to progressively older checkpoints in --save-dir when the newest one(s)
    are corrupted (e.g. a save interrupted mid-write by a disk quota hit)."""
    import re as _re
    ckpt_re = _re.compile(r"^checkpoint-(\d+)$")
    candidates = sorted(
        (p for p in Path(save_dir).iterdir()
         if p.is_dir() and ckpt_re.match(p.name)),
        key=lambda p: int(ckpt_re.match(p.name).group(1)), reverse=True)
    for ckpt in candidates:
        if _checkpoint_is_intact(ckpt):
            return str(ckpt)
    return None


def _load_adapter_tensors(peft_model, ckpt_dir, adapter_name="default"):
    """Load LoRA (+ modules_to_save) tensors from ckpt_dir's
    adapter_model.safetensors directly via load_state_dict, bypassing
    peft's set_peft_model_state_dict (broken for target_parameters/
    ParamWrapper adapters in peft 0.19 -- 'PhimoeExperts' has no attribute
    'weight'). Shared by --resume (own save-dir, full trainer state) and
    --init-adapter (a different checkpoint, weights only)."""
    from safetensors.torch import load_file
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
            f"adapter load failed, unexpected keys: {res.unexpected_keys[:5]}")
    missing_lora = [k for k in res.missing_keys if "lora" in k]
    if missing_lora:
        raise RuntimeError(
            f"adapter load failed, missing lora keys: {missing_lora[:5]}")
    return len(remapped)


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

CACHE_PREFIX_TEMPLATE = "[CACHE_SIZE={size}]"
_CACHE_PREFIX_RE = re.compile(r"^\[CACHE_SIZE=(\d+)\]")


def _prepend_cache_prefix(prompt, size):
    """Prepend the literal '[CACHE_SIZE=X]' conditioning prefix to a prompt
    -- either a plain string, or the first (user) turn of a conversational
    prompt ([{"role": ..., "content": ...}, ...])."""
    prefix = CACHE_PREFIX_TEMPLATE.format(size=size)
    if isinstance(prompt, str):
        return f"{prefix} {prompt}"
    out = list(prompt)
    out[0] = {**out[0], "content": f"{prefix} {out[0]['content']}"}
    return out


def parse_cache_size(prompt, default):
    """Parse the cache size back out of a (possibly prefixed) prompt. Falls
    back to `default` if no prefix is present (shouldn't happen for prompts
    coming out of build_prompt_conditioned_dataset / build_eval_prompts_
    conditioned, both of which always inject one)."""
    text = prompt if isinstance(prompt, str) else prompt[0].get("content", "")
    m = _CACHE_PREFIX_RE.match(text)
    return int(m.group(1)) if m else default


def build_prompt_conditioned_dataset(tokenizer, dataset_name, split, max_samples,
                                     prompt_len, cache_sizes, seed,
                                     skip_first=EVAL_POOL_PER_SPLIT):
    """Nemotron rows -> Dataset({'prompt'}), each filtered base prompt
    expanded into len(cache_sizes) rows, one per '[CACHE_SIZE=X]' prefix.

    The base prompt is filtered (not truncated) to `prompt_len - reserve`
    tokens, where `reserve` is the largest prefix's tokenized length plus a
    joining-space token -- so prepending any prefix keeps the whole prompt
    within `prompt_len` (RewardEngine._compute's own truncation is a safety
    net for the rare BPE-boundary edge case). See src.nemotron_data.
    sample_filtered_prompts for the scan/filter logic, shared with the
    other training scripts in this directory.
    """
    from src.nemotron_data import sample_filtered_prompts

    reserve = max(
        len(tokenizer(CACHE_PREFIX_TEMPLATE.format(size=s),
                     add_special_tokens=False)["input_ids"])
        for s in cache_sizes) + 1

    use_chat = tokenizer.chat_template is not None
    prompts = []
    for ids, _ in sample_filtered_prompts(
            tokenizer, dataset_name, split, max_samples,
            prompt_len - reserve, seed, skip_first):
        text = tokenizer.decode(ids)
        base = [{"role": "user", "content": text}] if use_chat else text
        for size in cache_sizes:
            prompts.append(_prepend_cache_prefix(base, size))
    return Dataset.from_dict({"prompt": prompts}).shuffle(seed=seed)


def build_eval_sequences(tokenizer, dataset_name, split, n_total, max_len,
                         seed, pool_per_split=EVAL_POOL_PER_SPLIT):
    """Held-out full conversations (chat template incl. reference answer),
    truncated to max_len tokens, sampled from the reserved eval pool. Not
    cache-size conditioned -- used only for the plain perplexity eval."""
    import random
    from src.nemotron_data import load_split_stream

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = n_total // len(splits)
    rng = random.Random(seed)
    use_chat = tokenizer.chat_template is not None
    eval_ids = []
    for sp in splits:
        ds = load_split_stream(dataset_name, sp)
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


def build_eval_prompts_conditioned(tokenizer, dataset_name, split, n_total,
                                   max_len, cache_sizes, seed,
                                   pool_per_split=EVAL_POOL_PER_SPLIT):
    """Held-out PROMPTS only (chat template + add_generation_prompt=True, no
    reference answer), each prefixed with a '[CACHE_SIZE=X]' condition and
    truncated to max_len tokens, from the same reserved eval pool as
    build_eval_sequences -- for the on-policy per-size cache-hit-rate eval.
    n_total is split evenly across cache_sizes so the eval budget matches
    finetune_moe_grpo.py's --eval-hitrate-seqs. Returns (prompt_ids,
    cache_sizes_per_prompt) -- parallel lists, so the eval callback doesn't
    need to re-parse token ids back to text."""
    import random
    from src.nemotron_data import load_split_stream

    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_size_total = max(n_total // len(cache_sizes), 1)
    per_split = max(per_size_total // len(splits), 1)
    rng = random.Random(seed)
    use_chat = tokenizer.chat_template is not None
    prompt_ids, prompt_sizes = [], []
    for sp in splits:
        ds = load_split_stream(dataset_name, sp)
        pool = []
        for row in ds:
            msgs = [m for m in row["messages"] if m["content"].strip()]
            if any(m["role"] == "assistant" for m in msgs):
                pool.append(msgs)
            if len(pool) >= pool_per_split:
                break
        for msgs in rng.sample(pool, min(per_split, len(pool))):
            user_msgs = []
            for m in msgs:
                if m["role"] == "assistant":
                    break
                user_msgs.append(m)
            if use_chat:
                text = tokenizer.apply_chat_template(
                    user_msgs, add_generation_prompt=True, tokenize=False)
            else:
                text = "\n".join(m["content"] for m in user_msgs)
            for size in cache_sizes:
                prefixed = f"{CACHE_PREFIX_TEMPLATE.format(size=size)} {text}"
                ids = tokenizer(prefixed, truncation=True, max_length=max_len,
                                add_special_tokens=False)["input_ids"]
                prompt_ids.append(ids)
                prompt_sizes.append(size)
    return prompt_ids, prompt_sizes


class CompletionsPruneCallback(TrainerCallback):
    """GRPOConfig(log_completions=True) writes a new completions_NNNNN.parquet
    to <output_dir>/completions/ on every logging step, with no rotation
    (unlike model checkpoints' save_total_limit) -- left alone this fills
    the disk over a long run. Keep only the `limit` most recent files."""

    def __init__(self, limit=3):
        self.limit = limit

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        comp_dir = Path(args.output_dir) / "completions"
        if not comp_dir.is_dir():
            return
        files = sorted(comp_dir.glob("completions_*.parquet"))
        for f in files[:-self.limit] if self.limit > 0 else []:
            f.unlink(missing_ok=True)


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


class CacheHitRateEvalCallback(TrainerCallback):
    """Every `every` optimizer steps, generates on-policy completions for a
    small held-out prompt set (rank 0 only) and logs the average LRU
    cache-hit rate at `cache_layer`, scored on the generated tokens only --
    per-example conditioned: `eval_cache_sizes[i]` is the LRU capacity
    simulated for `eval_prompt_ids[i]` (parsed from its own '[CACHE_SIZE=X]'
    prefix at eval-prompt build time). Logs both the aggregate hit rate and
    a per-size breakdown (eval/cache_hit_rate_size{X})."""

    def __init__(self, eval_prompt_ids, eval_cache_sizes, tokenizer, cache_layer,
                experts_per_token, use_topk, gen_len, every=25, batch_size=16):
        self.eval_prompt_ids = eval_prompt_ids
        self.eval_cache_sizes = eval_cache_sizes
        self.tokenizer = tokenizer
        self.cache_layer = cache_layer
        self.experts_per_token = experts_per_token
        self.use_topk = use_topk
        self.gen_len = gen_len
        self.every = every
        self.bs = batch_size
        self.trainer = None

    def attach(self, trainer):
        self.trainer = trainer

    @torch.no_grad()
    def _hit_rate(self, model):
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        per_seq_hit_rates = []
        per_seq_sizes = []
        for i in range(0, len(self.eval_prompt_ids), self.bs):
            chunk = self.eval_prompt_ids[i:i + self.bs]
            chunk_sizes = self.eval_cache_sizes[i:i + self.bs]
            B = len(chunk)
            P = max(len(p) for p in chunk)
            prompt_batch = torch.full((B, P), pad_id, dtype=torch.long)
            prompt_mask = torch.zeros((B, P), dtype=torch.long)
            for j, p in enumerate(chunk):
                prompt_batch[j, P - len(p):] = torch.tensor(p, dtype=torch.long)
                prompt_mask[j, P - len(p):] = 1
            prompt_batch, prompt_mask = prompt_batch.to(device), prompt_mask.to(device)

            gen = model.generate(
                input_ids=prompt_batch, attention_mask=prompt_mask,
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=self.gen_len, pad_token_id=pad_id,
            )
            completion_ids = gen[:, P:]

            S = P + completion_ids.shape[1]
            full_ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
            valid = torch.zeros((B, S), dtype=torch.bool, device=device)
            action = torch.zeros((B, S), dtype=torch.bool, device=device)
            full_ids[:, :P] = prompt_batch
            valid[:, :P] = prompt_mask.bool()
            full_ids[:, P:] = completion_ids
            comp_valid = completion_ids != pad_id
            # eos may legitimately appear as a real token; only pad tail is invalid
            comp_len = comp_valid.float().flip(-1).cumsum(-1).flip(-1).bool() | comp_valid
            valid[:, P:] = comp_len if comp_len.any() else comp_valid
            action[:, P:] = valid[:, P:]

            out = model(input_ids=full_ids, attention_mask=valid.long(),
                        output_router_logits=True, use_cache=False)
            router_logits = out.router_logits[self.cache_layer].view(B, S, -1)
            r_cache_tok, _, _ = cache_emulation_rewards(
                router_logits, valid, action, cache_size=chunk_sizes,
                experts_per_token=self.experts_per_token, use_topk=self.use_topk,
            )
            per_seq_hit_rates.extend(r_cache_tok.sum(-1).cpu().tolist())
            per_seq_sizes.extend(chunk_sizes)
        if was_training:
            model.train()
        result = {}
        if per_seq_hit_rates:
            result["eval/cache_hit_rate"] = sum(per_seq_hit_rates) / len(per_seq_hit_rates)
            for size in sorted(set(per_seq_sizes)):
                vals = [r for r, s in zip(per_seq_hit_rates, per_seq_sizes) if s == size]
                result[f"eval/cache_hit_rate_size{size}"] = sum(vals) / len(vals)
        return result

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0:
            return
        tr = self.trainer
        if tr is None:
            return
        # ALL ranks run the (identical) eval so no rank lags the others into
        # an NCCL collective timeout; only rank 0 logs -- same pattern as
        # PerplexityCallback.
        model = tr.accelerator.unwrap_model(tr.model)
        logs = self._hit_rate(model)
        if tr.accelerator.is_main_process and logs:
            import wandb
            if wandb.run is not None:
                wandb.log({**logs, "train/global_step": state.global_step})
            print(f"[eval] step {state.global_step}: " +
                 ", ".join(f"{k.split('/', 1)[1]} {v:.4f}" for k, v in logs.items()),
                 flush=True)


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
    """Computes the cache reward for a batch of GRPO completions, with a
    PER-EXAMPLE LRU capacity parsed from each prompt's own '[CACHE_SIZE=X]'
    prefix (see parse_cache_size) -- unlike finetune_moe_grpo.py's single
    args.cache_size (or its --conditioned-cache-sizes per-STEP override), a
    batch here is a shuffled mix of cache-size conditions and each sequence
    is scored against its own condition."""

    def __init__(self, args):
        self.args = args
        self.trainer = None
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
        router_logits = out.router_logits[args.cache_layer].view(B, S, -1)
        cache_sizes = [parse_cache_size(p, args.cache_size) for p in prompts]
        r_cache_tok, _, hit_rate = cache_emulation_rewards(
            router_logits, valid, action, cache_size=cache_sizes,
            experts_per_token=args.cache_experts_per_token,
            use_topk=args.cache_topk, soft=args.soft_cache,
        )
        cache_rewards = r_cache_tok.sum(-1)  # per-seq hit fraction in [0, 1]

        if was_training:
            model.train()
        return {"cache": cache_rewards.cpu().tolist(), "hit_rate": hit_rate,
               "cache_sizes": cache_sizes}

    def _scores(self, prompts, completion_ids):
        key = (id(completion_ids), len(completion_ids),
               tuple(completion_ids[0][:4]) if len(completion_ids[0]) else ())
        if key != self._memo_key:
            self._memo = self._compute(prompts, completion_ids)
            self._memo_key = key
        return self._memo

    # -- reward function passed to GRPOTrainer -------------------------------

    def cache_reward(self, prompts, completions, completion_ids,
                     log_metric=None, **kwargs):
        scores = self._scores(prompts, completion_ids)
        if log_metric is not None:
            log_metric("cache_hit_rate", scores["hit_rate"])
            cache_rewards, cache_sizes = scores["cache"], scores["cache_sizes"]
            for size in sorted(set(cache_sizes)):
                vals = [r for r, s in zip(cache_rewards, cache_sizes) if s == size]
                log_metric(f"cache_hit_rate_size{size}", sum(vals) / len(vals))
        return scores["cache"]


def main():
    parser = argparse.ArgumentParser(
        description="GRPO finetuning of an MoE LLM with a PROMPT-CONDITIONED "
                    "expert-cache reward (TRL)"
    )
    parser.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    parser.add_argument("--dataset", type=str,
                        default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-split", type=str, default="math,code")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Base (unprefixed) prompts sampled; the final "
                             "dataset has max_samples * len(--prompt-cache-"
                             "sizes) rows")
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
    parser.add_argument("--save-dir", type=str,
                        default="checkpoints/grpo_prompt_conditioned")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Rotating checkpoints kept in --save-dir")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Steps between metric/completions-sample logs. "
                             "GRPOConfig(log_completions=True) writes a new, "
                             "never-rotated completions_NNNNN.parquet per "
                             "logging step -- keep this well above 1 or it "
                             "silently fills the disk over a long run.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in --save-dir")
    parser.add_argument("--init-adapter", type=str, default=None,
                        help="Initialize the LoRA adapter's weights from a "
                             "different checkpoint's adapter_model.safetensors "
                             "(e.g. a completed SFT run) before RL training "
                             "starts -- weights only, no optimizer/step state. "
                             "Ignored when --resume finds a checkpoint already "
                             "in --save-dir (that run continues instead).")
    parser.add_argument("--eval-ppl-seqs", type=int, default=256,
                        help="Held-out sequences for the perplexity eval")
    parser.add_argument("--eval-ppl-every", type=int, default=25,
                        help="Optimizer steps between perplexity evals")
    parser.add_argument("--eval-hitrate-seqs", type=int, default=32,
                        help="Held-out prompts for the on-policy cache-hit-"
                             "rate eval, split evenly across --prompt-cache-"
                             "sizes (generates a completion per prompt, "
                             "same mechanics as eval/eval_router.py)")
    parser.add_argument("--eval-hitrate-every", type=int, default=25,
                        help="Optimizer steps between cache-hit-rate evals")

    rl_group = parser.add_argument_group("Rewards")
    rl_group.add_argument("--beta", type=float, default=0.04,
                          help="TRL KL(policy||ref) coefficient (ref = "
                               "adapters disabled = frozen base)")
    rl_group.add_argument("--loss-type", type=str, default="dapo",
                          help="TRL GRPO loss variant (dapo = TRL 1.8 default)")
    rl_group.add_argument("--router-aux-loss-coef", type=float, default=0.0,
                          help="MoE load-balancing aux loss coef. Keep 0: "
                               "balancing pushes uniform expert usage, the "
                               "opposite of cache consolidation")

    cache_group = parser.add_argument_group("Expert LRU cache")
    cache_group.add_argument("--cache-size", type=int, default=4,
                             help="Fallback LRU capacity, used only if a "
                                  "prompt is somehow missing its '[CACHE_SIZE"
                                  "=X]' prefix (shouldn't happen)")
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
    cache_group.add_argument("--prompt-cache-sizes", type=str, default="2,4,8",
                             help="Comma list of cache sizes to condition on "
                                  "via a '[CACHE_SIZE=X]' prompt prefix: every "
                                  "filtered base prompt is expanded into one "
                                  "row per size, shuffled together so each "
                                  "training batch mixes conditions. The LRU "
                                  "simulated for the reward is parsed back "
                                  "out of each sequence's own prompt (per-"
                                  "example, not per-step).")

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

    args.prompt_cache_sizes = [
        int(x) for x in args.prompt_cache_sizes.split(",") if x.strip()]
    if not args.prompt_cache_sizes:
        raise ValueError("--prompt-cache-sizes must list at least one size")

    import os
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building prompt-conditioned dataset {args.dataset} "
          f"[{args.dataset_split}], cache sizes {args.prompt_cache_sizes} ...")
    # Serialize dataset streaming across ranks: concurrent access to the
    # shared HF cache can fail with "[Errno 16] Device or resource busy".
    # Additionally cache the built prompts on disk so concurrent sweep jobs
    # don't all re-stream from HF (observed 504s killing sibling jobs).
    import hashlib, pickle
    key = hashlib.md5(str((args.dataset, args.dataset_split, args.max_samples,
                           args.prompt_len, args.prompt_cache_sizes,
                           args.eval_ppl_seqs, args.eval_hitrate_seqs,
                           args.seed, args.model,
                           "v1_prompt_conditioned")).encode()  # bump on cache schema changes
                      ).hexdigest()[:12]
    cache_file = Path("data") / f"prompt_cache_{key}.pkl"
    from accelerate import PartialState
    with PartialState().main_process_first():
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                prompts_list, eval_ids, eval_prompt_ids, eval_prompt_sizes = pickle.load(f)
            train_dataset = Dataset.from_dict({"prompt": prompts_list})
            print(f"[data] loaded cached prompts from {cache_file}")
        else:
            train_dataset = build_prompt_conditioned_dataset(
                tokenizer, args.dataset, args.dataset_split, args.max_samples,
                args.prompt_len, args.prompt_cache_sizes, args.seed)
            eval_ids = build_eval_sequences(
                tokenizer, args.dataset, args.dataset_split, args.eval_ppl_seqs,
                args.prompt_len + args.completion_len, args.seed)
            eval_prompt_ids, eval_prompt_sizes = build_eval_prompts_conditioned(
                tokenizer, args.dataset, args.dataset_split, args.eval_hitrate_seqs,
                args.prompt_len, args.prompt_cache_sizes, args.seed)
            if PartialState().is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write (per-process temp file + os.replace): two
                # entirely separate training runs sharing the same dataset
                # config can hash to the same cache_file and race on writing
                # it. os.replace is atomic on POSIX, so a concurrent reader
                # always sees either the old file or one fully-written new
                # one -- but the temp filename itself must be unique per
                # process (pid-suffixed), or two racing writers can share it
                # and one's os.replace consumes the file out from under the
                # other, raising FileNotFoundError.
                tmp_file = cache_file.with_suffix(f".pkl.tmp.{os.getpid()}")
                with open(tmp_file, "wb") as f:
                    pickle.dump((list(train_dataset["prompt"]), eval_ids,
                                eval_prompt_ids, eval_prompt_sizes), f)
                os.replace(tmp_file, cache_file)
                print(f"[data] cached prompts to {cache_file}")
    print(f"[data] {len(train_dataset)} training prompts "
          f"({len(train_dataset) // len(args.prompt_cache_sizes)} base x "
          f"{len(args.prompt_cache_sizes)} cache sizes)")
    print(f"[eval] {len(eval_ids)} held-out sequences "
          f"(<= {args.prompt_len + args.completion_len} tokens each), "
          f"{len(eval_prompt_ids)} held-out prompts for cache-hit-rate eval")

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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # ParamWrapper (fused experts) forbids dropout
        target_modules=target_modules,
        **lora_kwargs,
    )

    engine = RewardEngine(args)
    reward_funcs = [engine.cache_reward]
    reward_weights = [1.0]

    grpo_config = GRPOConfig(
        output_dir=args.save_dir,
        run_name=args.wandb_run_name,
        report_to=["wandb"],
        seed=args.seed,
        ddp_timeout=3600,  # headroom for the synchronized perplexity eval
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.num_steps,
        logging_steps=args.logging_steps,
        log_completions=True,  # sample completions -> wandb table (quality check)
        num_completions_to_print=0,
        save_steps=args.save_every,
        save_total_limit=args.save_total_limit,  # keep rewind points (reward hacking recovery)
        num_generations=args.num_generations,
        max_completion_length=args.completion_len,
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,
        router_aux_loss_coef=args.router_aux_loss_coef,
        reward_weights=reward_weights,
        # No trust_remote_code: use transformers' native classes (Phi-tiny's
        # bundled remote code requires flash_attn and bypasses the peft/TRL
        # integration we rely on).
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager"},
    )

    trainer = GRPOTrainer(
        model=args.model,
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
    trainer.add_callback(PreemptionCallback())
    trainer.add_callback(CompletionsPruneCallback(limit=args.save_total_limit))

    model = trainer.accelerator.unwrap_model(trainer.model)
    num_layers = model.config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    print(f"[cache] prompt-conditioned LRU sizes={args.prompt_cache_sizes} "
          f"on router of layer {args.cache_layer}/{num_layers}")

    if args.soft_cache:
        # Dense routing at the cache layer: every expert is active with its
        # full-softmax weight (sparsemixer only supports iterative top-1/2).
        # Patched on the instance post-peft, so it applies to rollouts, the
        # scoring pass, and the adapter-disabled ref/teacher forwards alike.
        # self.weight is the LoRA-merged router weight inside ParamWrapper's
        # forward, and full softmax keeps routing differentiable end-to-end.
        if model.config.model_type != "phimoe":
            raise ValueError("--soft-cache is only wired up for phimoe")
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

    hitrate_cb = CacheHitRateEvalCallback(
        eval_prompt_ids, eval_prompt_sizes, tokenizer, args.cache_layer,
        args.cache_experts_per_token, args.cache_topk, args.completion_len,
        every=args.eval_hitrate_every)
    hitrate_cb.attach(trainer)
    trainer.add_callback(hitrate_cb)

    resume_ckpt = None
    if args.resume and Path(args.save_dir).is_dir():
        resume_ckpt = _find_valid_checkpoint(args.save_dir)
        if resume_ckpt:
            print(f"Resuming from {resume_ckpt}")

    if resume_ckpt:
        # peft 0.19 cannot load target_parameters (ParamWrapper) adapters via
        # set_peft_model_state_dict ('PhimoeExperts' has no attribute
        # 'weight'). Replace PeftModel.load_adapter -- which the HF Trainer
        # calls on resume -- with a direct load_state_dict of the remapped
        # adapter tensors. Optimizer/scheduler/trainer state still load
        # through the normal Trainer path.
        peft_model = trainer.model

        def _manual_load_adapter(ckpt_dir, adapter_name="default", **kwargs):
            n = _load_adapter_tensors(peft_model, ckpt_dir, adapter_name)
            print(f"[resume] manually loaded {n} adapter tensors")

        peft_model.load_adapter = _manual_load_adapter
    elif args.init_adapter:
        n = _load_adapter_tensors(trainer.model, args.init_adapter)
        print(f"[init] loaded {n} adapter tensors from {args.init_adapter}")

    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    main()
