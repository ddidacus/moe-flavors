"""Minimal PyTorch reimplementation of the Option-Critic MoE controller from
Shen & Henderson, "Temporally Extended Mixture-of-Experts Models" (2026)
(https://github.com/princeton-polaris-lab/rl_moe), as a baseline to compare
against our own cache-hit-reward / temporal-mixin GRPO runs.

The paper trains a per-layer controller (termination head + Plackett-Luce
expert-selection head + value head) via Option-Critic with deliberation
costs, so the model switches its ACTIVE expert subset rarely instead of
every token, while a self-distillation reward (reverse KL between a frozen
dense teacher and the expert-masked student) keeps behavior close to the
unrestricted model. Their reference implementation is ~4300 lines, tied to
gpt-oss + custom transformers patches + DeepSpeed, with a full generation
rollout loop, GAE, and a Q-head for Q_U(s, option). This is a deliberately
minimal, single-file version:

  - ONE controller at --cache-layer (not every MoE layer), matching our own
    single-layer convention (finetune_moe_grpo.py, eval_soft_cache.py) so
    it's directly comparable to the other runs.
  - Teacher-forced, not on-policy rollout: reward_type=kl (self-distillation)
    needs no generation, only one forward pass over the dataset's own
    prompt+completion. This drops the correctness-reward / RL-rollout
    machinery entirely (out of scope for a baseline comparison).
  - Closed-form reward instead of a literal second downstream forward: with
    student := teacher's router distribution renormalized onto the current
    option (the natural "masked and renormalized" student), the reverse KL
    collapses to a one-liner --
        KL(student || teacher) = sum_{e in option} student_e * log(student_e/teacher_e)
                                = -log Z,  Z = sum_{e in option} teacher_e
    (substitute student_e = teacher_e / Z and the sum over the option
    telescopes to exactly -log Z). So reward_t = log(Z_t): maximize the
    teacher's probability mass retained on the chosen option -- the same
    "probability mass on the working set" quantity our own soft-cache
    reward already uses, just in log-space here for the proper KL. This
    avoids re-running every downstream layer twice per candidate option.
  - Plain one-step TD advantage + a V-head (Bacon et al. 2017 Option-Critic)
    instead of the reference's separate Q_U(s,option) head + GAE.
  - The termination gradient is PATHWISE (direct autodiff through beta =
    sigmoid(switch_logit), not REINFORCE): loss_term = -beta*(A+eta), which
    is exactly d/dtheta_beta of the option-critic objective (Bacon et al.
    Eq., deliberation cost eta only in the termination gradient -- Harb et
    al.'s lambda=0 regime, so it never distorts the reward signal itself).
    Selection is REINFORCE (Plackett-Luce log-prob), since sampling a new
    discrete expert SET isn't differentiable.
  - LM/LoRA weights are trained by the SAME SFT NLL loss as our other runs
    (finetune_moe_grpo.py / finetune_moe_sft.py), not the paper's own
    LM-policy-gradient-on-KL -- keeps this baseline on equal footing:
    same dataset, same prompt/completion length, same LoRA r/alpha as our
    other experiments (the paper's own LoRA hparams are NOT used here).

Non-LoRA hyperparameters below are the paper's own argparse defaults/README
example (--controller-expert-embed-dim 32, --activation-controller-mlp-hidden
512, --deliberation-cost 0.02, --value-coef 0.1, --gamma 0.99,
switch_init_bias -3.0). --controller-allowed-experts (k, the option size)
defaults to our own --cache-size (4) instead of their gpt-oss-scale default
(16), for direct comparability with our cache-hit-reward runs' cache size.
"""

import argparse
import math
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# peft 0.19 x transformers 5.8 bug -- see finetune_moe_grpo.py for the same
# workaround, kept identical across every script in this pipeline.
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


EVAL_POOL_PER_SPLIT = 1000  # matches finetune_moe_grpo.py -- same held-out rows


# ============================================================================
# Controller (minimal reimplementation of GptOssActivationController)
# ============================================================================

class ExpertSelectionController(nn.Module):
    """Termination + Plackett-Luce selection + value heads, reading the
    cache layer's own hidden states -- architecture mirrors the reference
    repo's GptOssActivationController (rl_moe/transformers_patches/models/
    gpt_oss/modeling_gpt_oss.py) exactly, minus its separate Q_U(s,option)
    head (we use a plain V-head + one-step TD advantage instead)."""

    def __init__(self, hidden_size, num_experts, embed_dim=32, mlp_hidden=512,
                switch_init_bias=-3.0, router_weight=None, router_bias=None):
        super().__init__()
        self.num_experts = num_experts
        self.expert_embedding = nn.Embedding(num_experts, embed_dim)
        self.deepsets_phi = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
        )
        self.termination_head = nn.Sequential(
            nn.Linear(hidden_size + mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        with torch.no_grad():
            # low initial switch probability (sigmoid(-3) ~ 0.047) -> mostly
            # holds at the start, has to learn to switch when it pays off
            self.termination_head[2].weight.mul_(0.01)
            self.termination_head[2].bias.fill_(switch_init_bias)

        self.selection_head = nn.Linear(hidden_size, num_experts)
        if router_weight is not None:
            with torch.no_grad():
                self.selection_head.weight.copy_(router_weight)
        if router_bias is not None:
            with torch.no_grad():
                self.selection_head.bias.copy_(router_bias)

        self.value_head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.value_head.weight.mul_(0.01)
            self.value_head.bias.zero_()

        self.h_norm = nn.LayerNorm(hidden_size)
        self.s_norm = nn.LayerNorm(mlp_hidden)

    def set_embedding(self, option_idx):
        """DeepSets embedding of a k-expert option: per-expert embed -> phi
        MLP -> mean-pool. option_idx: [B, k] (all valid, no -1 sentinel --
        our option is always exactly k experts, unlike the reference's
        variable-length sets)."""
        embeds = self.expert_embedding(option_idx)          # [B, k, embed_dim]
        return self.deepsets_phi(embeds).mean(dim=1)         # [B, mlp_hidden]

    def forward(self, h, current_option):
        """h: [B, hidden_size], current_option: [B, k] -> (switch_logit [B],
        selection_logits [B, num_experts], value [B])."""
        # LLM hidden states are bf16; controller params are float32 (matches
        # the reference implementation's own dtype handling)
        h = h.to(self.selection_head.weight.dtype)
        s = self.set_embedding(current_option)
        term_in = torch.cat([self.h_norm(h), self.s_norm(s)], dim=-1)
        switch_logit = self.termination_head(term_in).squeeze(-1)
        selection_logits = self.selection_head(h)
        value = self.value_head(h).squeeze(-1)
        return switch_logit.float(), selection_logits.float(), value.float()


def plackett_luce_sample(logits, k):
    """Sample a k-permutation without replacement (Plackett-Luce: repeatedly
    sample from the softmax over remaining items). Returns (selections [B,k],
    logprob [B])."""
    B, E = logits.shape
    remaining = logits.clone()
    selections = torch.zeros(B, k, dtype=torch.long, device=logits.device)
    logprob = torch.zeros(B, device=logits.device, dtype=logits.dtype)
    batch_idx = torch.arange(B, device=logits.device)
    for step in range(k):
        probs = F.softmax(remaining, dim=-1)
        choice = torch.multinomial(probs, 1).squeeze(-1)
        logprob = logprob + F.log_softmax(remaining, dim=-1)[batch_idx, choice]
        selections[:, step] = choice
        remaining[batch_idx, choice] = float("-inf")
    return selections, logprob


# ============================================================================
# Dataset (same convention as finetune_moe_grpo.py / finetune_moe_sft.py)
# ============================================================================

def build_prompt_dataset(tokenizer, dataset_name, split, max_samples, prompt_len,
                         completion_len, seed, skip_first=EVAL_POOL_PER_SPLIT):
    """Nemotron rows -> Dataset({'prompt', 'target_ids'}) -- prompt tokenized
    and truncated to prompt_len, target_ids = the dataset's own ground-truth
    assistant response, truncated to completion_len. Identical convention to
    finetune_moe_grpo.py's build_prompt_dataset (skip_first reserves the same
    held-out rows), except here 'prompt' is plain text (this script builds
    the full teacher-forced prompt+completion sequence itself, no chat
    template rendering needed since there's no on-policy generation).

    Uses reservoir sampling (Algorithm R) within each split so the result is
    a genuine random sample, not a fixed prefix -- capped at a scan window
    (see MAX_SCAN_PER_SPLIT) so very large splits don't have to be streamed
    to completion just to draw a few hundred rows from them. Tokenization
    happens AFTER the reservoir is finalized, not during the scan: doing it
    eagerly per scanned row cost up to MAX_SCAN_PER_SPLIT x num_splits BPE
    calls for a result that only keeps per_split x num_splits of them --
    with 9 splits x 50k that's ~450k wasted tokenizer calls, which single-
    handedly blew well past accelerate's 600s multi-GPU rendezvous timeout
    (rank 0 stuck tokenizing while other ranks waited and gave up)."""
    from datasets import Dataset, load_dataset
    import random

    MAX_SCAN_PER_SPLIT = 50_000
    rng = random.Random(seed)
    splits = [s.strip() for s in split.split(",") if s.strip()]
    per_split = max_samples // len(splits)
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
            target_ids = tokenizer(target_text, truncation=True,
                                   max_length=completion_len,
                                   add_special_tokens=False)["input_ids"]
            prompts.append(text)
            targets.append(target_ids)
    return Dataset.from_dict({"prompt": prompts, "target_ids": targets}) \
        .shuffle(seed=seed)


def collate(batch, tokenizer, prompt_len, completion_len):
    """prompt (left-padded) + target_ids (right-padded) concatenated, plus an
    action_mask marking the target span (the only positions the controller
    and the NLL loss are scored on -- the prompt only warms the option, same
    convention as eval_soft_cache.py's prompt_lens)."""
    pad_id = tokenizer.pad_token_id
    prompt_ids = [tokenizer(b["prompt"], truncation=True,
                            max_length=prompt_len)["input_ids"] for b in batch]
    target_ids = [b["target_ids"] for b in batch]
    P = max(len(p) for p in prompt_ids)
    T = max(len(t) for t in target_ids)
    B = len(batch)
    input_ids = torch.full((B, P + T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, P + T, dtype=torch.long)
    action_mask = torch.zeros(B, P + T, dtype=torch.bool)
    labels = torch.full((B, P + T), -100, dtype=torch.long)
    for i, (p, t) in enumerate(zip(prompt_ids, target_ids)):
        # left-pad the prompt, right-pad the target -- prompt ends exactly
        # at P for every row, so the target always starts at the same index
        input_ids[i, P - len(p):P] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, P - len(p):P] = 1
        if t:
            input_ids[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)
            attention_mask[i, P:P + len(t)] = 1
            action_mask[i, P:P + len(t)] = True
            labels[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "action_mask": action_mask, "labels": labels, "prompt_len": P}


# ============================================================================
# Controller forward + Option-Critic losses (teacher-forced, one pass)
# ============================================================================

def controller_losses(model, controller, input_ids, attention_mask,
                      action_mask, cache_layer, cache_size, gamma,
                      deliberation_cost, sampling_temp):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_router_logits=True, output_hidden_states=True,
                    use_cache=False)
    lm_logits = outputs.logits
    # hidden_states[i] = INPUT to decoder layer i (output of layer i-1), so
    # hidden_states[cache_layer] is exactly what that layer's own router
    # reads -- no forward hook needed.
    h_seq = outputs.hidden_states[cache_layer]              # [B, T, H]
    B, T, _ = h_seq.shape
    # router_logits comes out flattened as [B*T, E], not [B, T, E] (same
    # gotcha handled in eval_soft_cache.py's analyze())
    router_logits = outputs.router_logits[cache_layer].float().view(B, T, -1)
    teacher_probs = F.softmax(router_logits, dim=-1)         # [B, T, E]
    device = h_seq.device

    current_option = torch.topk(teacher_probs[:, 0, :], cache_size, dim=-1).indices

    rewards, values, betas, switches, pl_logprobs = [], [], [], [], []
    for t in range(T):
        switch_logit, selection_logits, value = controller(h_seq[:, t, :], current_option)
        beta = torch.sigmoid(switch_logit)
        switch = (torch.rand_like(beta) < beta)
        new_option, pl_logprob = plackett_luce_sample(
            selection_logits / sampling_temp, cache_size)
        exec_option = torch.where(switch.unsqueeze(-1), new_option, current_option)

        Z = teacher_probs[:, t, :].gather(-1, exec_option).sum(-1).clamp_min(1e-8)
        rewards.append(torch.log(Z))
        values.append(value)
        betas.append(beta)
        switches.append(switch.float())
        pl_logprobs.append(pl_logprob)

        current_option = exec_option

    rewards = torch.stack(rewards, dim=1)     # [B, T]  reward_t = log Z_t
    values = torch.stack(values, dim=1)
    betas = torch.stack(betas, dim=1)
    switches = torch.stack(switches, dim=1)
    pl_logprobs = torch.stack(pl_logprobs, dim=1)

    # one-step TD target/advantage (Bacon et al. 2017); bootstrap with 0 at
    # the last valid step of each sequence (no next state to bootstrap from)
    values_next = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    td_target = rewards + gamma * values_next.detach()
    advantage = (td_target - values).detach()

    m = action_mask.float()
    n_tok = m.sum().clamp_min(1)

    value_loss = (((values - td_target) ** 2) * m).sum() / n_tok
    # Direct/pathwise termination gradient (Bacon et al.; Harb et al.'s
    # lambda=0 regime keeps the deliberation cost OUT of the reward and
    # value target, only inside this gradient): d/dtheta_beta J = beta *
    # (A + eta). We ascend J -> minimize -J.
    term_loss = -((betas * (advantage + deliberation_cost)) * m).sum() / n_tok
    # REINFORCE for the (non-differentiable) discrete expert-set choice,
    # masked to steps that actually switched.
    switch_m = (switches * m)
    n_switch = switch_m.sum().clamp_min(1)
    selection_loss = -((pl_logprobs * advantage * switch_m).sum() / n_switch)

    with torch.no_grad():
        mean_reward = (rewards * m).sum() / n_tok
        switch_rate = switch_m.sum() / n_tok

    return {
        "lm_logits": lm_logits,
        "value_loss": value_loss,
        "term_loss": term_loss,
        "selection_loss": selection_loss,
        "mean_reward": mean_reward,
        "switch_rate": switch_rate,
    }


def sft_nll_loss(lm_logits, labels):
    shift_logits = lm_logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1) \
        .gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return -(logp * mask).sum() / mask.sum().clamp_min(1)


class Preemption:
    """Same SLURM preemption pattern as the other scripts (PreemptionCallback
    in finetune_moe_grpo.py) adapted to a manual training loop: SIGUSR1/
    SIGTERM just set a flag, checked once per step."""

    def __init__(self):
        self.triggered = False
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        print(f"[preemption] signal {signum} received -- checkpointing and "
              f"stopping at the next step boundary", flush=True)
        self.triggered = True


def main():
    parser = argparse.ArgumentParser(
        description="Option-Critic MoE controller baseline (Shen & "
                    "Henderson 2026), minimal reimplementation")
    parser.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-split", default="math,code")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--completion-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LoRA (LM weight) lr -- ours, not the paper's")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-cache-reinforce")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/controller_baseline")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")

    cache_group = parser.add_argument_group("Cache layer / option size")
    cache_group.add_argument("--cache-layer", type=int, default=-1, help="-1 = middle layer")
    cache_group.add_argument("--cache-size", type=int, default=4,
                             help="k, the option size -- ours (4), not the "
                                  "paper's gpt-oss-scale default (16), for "
                                  "direct comparability with our cache-hit "
                                  "reward runs")

    ctrl_group = parser.add_argument_group("Controller (paper's own hparams)")
    ctrl_group.add_argument("--controller-lr", type=float, default=1e-5)
    ctrl_group.add_argument("--controller-expert-embed-dim", type=int, default=32)
    ctrl_group.add_argument("--controller-mlp-hidden", type=int, default=512)
    ctrl_group.add_argument("--switch-init-bias", type=float, default=-3.0)
    ctrl_group.add_argument("--deliberation-cost", type=float, default=0.02,
                            help="eta -- README example run value (argparse "
                                 "default in the reference repo is 0.1)")
    ctrl_group.add_argument("--value-coef", type=float, default=0.1)
    ctrl_group.add_argument("--gamma", type=float, default=0.99)
    ctrl_group.add_argument("--sampling-temperature", type=float, default=0.7)
    ctrl_group.add_argument("--sft-coef", type=float, default=1.0,
                            help="LM NLL loss weight -- matches our other "
                                 "runs' SFT term, not part of the paper")

    lora_group = parser.add_argument_group("LoRA (ours, not the paper's)")
    lora_group.add_argument("--lora-r", type=int, default=16)
    lora_group.add_argument("--lora-alpha", type=int, default=32)

    args = parser.parse_args()

    import os
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building dataset {args.dataset} [{args.dataset_split}] ...")
    import hashlib, pickle
    key = hashlib.md5(str((args.dataset, args.dataset_split, args.max_samples,
                           args.prompt_len, args.completion_len, args.seed,
                           args.model, "controller_v1")).encode()).hexdigest()[:12]
    cache_file = Path("data") / f"controller_prompt_cache_{key}.pkl"
    from accelerate import PartialState
    with PartialState().main_process_first():
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                prompts_list, targets_list = pickle.load(f)
            from datasets import Dataset
            train_dataset = Dataset.from_dict(
                {"prompt": prompts_list, "target_ids": targets_list})
            print(f"[data] loaded cached prompts from {cache_file}")
        else:
            train_dataset = build_prompt_dataset(
                tokenizer, args.dataset, args.dataset_split, args.max_samples,
                args.prompt_len, args.completion_len, args.seed)
            if PartialState().is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump((list(train_dataset["prompt"]),
                                list(train_dataset["target_ids"])), f)
                print(f"[data] cached prompts to {cache_file}")
    print(f"[data] {len(train_dataset)} training rows")

    model_cfg = AutoConfig.from_pretrained(args.model)
    lora_kwargs = {}
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if model_cfg.model_type == "phimoe":
        lora_kwargs = dict(
            target_parameters=["experts.gate_up_proj", "experts.down_proj",
                               "router.weight"],
            rank_pattern={r".*\.gate_up_proj": args.lora_r * 2},
            alpha_pattern={r".*\.gate_up_proj": args.lora_alpha * 2},
        )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.0, target_modules=target_modules, **lora_kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    num_layers = model.config.num_hidden_layers
    if args.cache_layer < 0:
        args.cache_layer = num_layers // 2
    print(f"[controller] layer {args.cache_layer}/{num_layers}, "
          f"option size k={args.cache_size}")

    # router weight/bias at the cache layer, to init the selection head the
    # same way the reference implementation does (starts equivalent to
    # native routing, not a random guess)
    router_module = None
    for name, mod in model.named_modules():
        if (type(mod).__name__ == "PhimoeTopKRouter"
                and f"layers.{args.cache_layer}.mlp" in name):
            router_module = mod
            break
    assert router_module is not None, f"no router found at layer {args.cache_layer}"
    router_weight = router_module.weight.data.clone()
    router_bias = router_module.bias.data.clone() if router_module.bias is not None else None

    peft_model = get_peft_model(model, lora_config)
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()

    controller = ExpertSelectionController(
        hidden_size=model.config.hidden_size,
        num_experts=model.config.num_local_experts,
        embed_dim=args.controller_expert_embed_dim,
        mlp_hidden=args.controller_mlp_hidden,
        switch_init_bias=args.switch_init_bias,
        router_weight=router_weight, router_bias=router_bias,
    ).to(dtype=torch.float32)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in peft_model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": controller.parameters(), "lr": args.controller_lr},
    ])

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.prompt_len, args.completion_len))

    peft_model, controller, optimizer, dataloader = accelerator.prepare(
        peft_model, controller, optimizer, dataloader)

    if accelerator.is_main_process and args.wandb_run_name:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                  config=vars(args))

    save_dir = Path(args.save_dir)
    start_step = 0
    if args.resume and save_dir.is_dir():
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(save_dir)
        if ckpt:
            print(f"Resuming from {ckpt}")
            unwrapped = accelerator.unwrap_model(peft_model)
            unwrapped.load_adapter(ckpt, adapter_name="default", is_trainable=True)
            ctrl_state = torch.load(Path(ckpt) / "controller.pt", map_location="cpu")
            accelerator.unwrap_model(controller).load_state_dict(ctrl_state["controller"])
            optimizer.load_state_dict(ctrl_state["optimizer"])
            start_step = ctrl_state["step"]

    preemption = Preemption()
    step = start_step
    data_iter = iter(dataloader)
    peft_model.train()
    controller.train()
    while step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(peft_model):
            out = controller_losses(
                peft_model, controller, batch["input_ids"], batch["attention_mask"],
                batch["action_mask"], args.cache_layer, args.cache_size, args.gamma,
                args.deliberation_cost, args.sampling_temperature)
            nll = sft_nll_loss(out["lm_logits"], batch["labels"])
            loss = (args.sft_coef * nll + args.value_coef * out["value_loss"]
                   + out["term_loss"] + out["selection_loss"])
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                logs = {
                    "loss": loss.item(), "sft_nll": nll.item(),
                    "value_loss": out["value_loss"].item(),
                    "term_loss": out["term_loss"].item(),
                    "selection_loss": out["selection_loss"].item(),
                    "controller_reward": out["mean_reward"].item(),
                    "switch_rate": out["switch_rate"].item(),
                }
                print(f"step {step}: {logs}", flush=True)
                if args.wandb_run_name:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(logs, step=step)

            if step % args.save_every == 0 or step >= args.num_steps or preemption.triggered:
                if accelerator.is_main_process:
                    ckpt_dir = save_dir / f"checkpoint-{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(peft_model).save_pretrained(str(ckpt_dir))
                    torch.save({
                        "controller": accelerator.unwrap_model(controller).state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step,
                    }, ckpt_dir / "controller.pt")
                    print(f"[save] checkpoint-{step} -> {ckpt_dir}", flush=True)
                if preemption.triggered:
                    print("[preemption] stopping after checkpoint", flush=True)
                    break

    print("done", flush=True)


if __name__ == "__main__":
    main()
