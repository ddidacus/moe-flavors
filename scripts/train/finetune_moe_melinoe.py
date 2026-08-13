"""MELINOE baseline (Raje, Nayak & Joshi, "Fine-Tuning Enables Memory-
Efficient Inference for Mixture-of-Experts Models", arXiv:2602.11192, 2026)
-- https://arxiv.org/abs/2602.11192

The paper fine-tunes an MoE model to concentrate per-sequence expert
activation onto a small, cache-friendly subset, using two auxiliary losses
on top of the ordinary SFT NLL loss:

  Cache Simulation Loss Lcs: a differentiable proxy for expert-cache misses
    under a recency-weighted (gamma-discounted) soft cache of capacity C.
    Let r_f(t) in {0,1}^E be the fine-tuned router's top-K request vector at
    token t (straight-through: hard forward value, gradient flows through
    the softmax as if r_f(t) = p_f(t)) and c(t) in R^E_>=0 (||c||_1 = C) the
    soft cache state carried in from all PRIOR tokens (not updated with
    r_f(t) itself yet):
        Lcs = mean_t < r_f(t), 1 - c(t) >
        c(t+1) = [gamma*Z(t)*c(t) + r_f(t)] / Z(t+1),  Z(t+1) = gamma*Z(t) + K/C
    (paper Eq. 4, 11; Appendix C.1 Prop. C.3). gamma=0.9 interpolates between
    LRU (gamma->0) and LFU (gamma=1) semantics.

  Rank Matching Loss Lrm: a margin ranking loss that keeps the fine-tuned
    router's relative expert ORDERING close to the frozen base router's
    (prevents router collapse onto a globally-fixed subset, unlike a plain
    entropy/KL penalty which the paper argues is less aligned with a Top-K
    selection mechanism -- Appendix C.2):
        m(t) = sum_{i,j} 1{p_b,i(t) > p_b,j(t)} * relu(rho - (p_f,i(t) - p_f,j(t)))
        Lrm = mean_t m(t)
    p_b is the SAME model's router with the adapter disabled (frozen base),
    matching this repo's convention elsewhere (finetune_moe_grpo.py's ref
    model for KL) rather than a separately-loaded base copy.

  Total: L = L_nll + lambda_cs * Lcs + lambda_rm * Lrm      (Eq. 6)

Deviations from the paper, matching this repo's existing conventions (see
finetune_moe_controller.py's own docstring for the same rationale):
  - ONE cache layer (--cache-layer), not every MoE layer, for direct
    comparability with cache_sft / temporal_moe / controller_baseline and
    with eval/eval_router.py's own single-layer analysis.
  - No Stage 2 (BGE-embedding activation predictor + proactive GPU-cache
    prefetching): that stage targets real CPU-GPU transfer throughput on a
    memory-constrained deployment, which this repo doesn't measure --
    eval/eval_router.py evaluates routing-quality metrics (hit_ratio,
    switch_rate, expert_run_length) on the fine-tuned ROUTER alone, so only
    Stage 1 (Section 3.1.1) is in scope.
  - Cache capacity C = --cache-size (default 4) = E/4 for this model's E=16
    local experts, matching the paper's own C = E/4 rule exactly; K =
    --top-k defaults to this model's real num_experts_per_tok (2), not
    OLMoE's 8/64.
  - lambda_cs/lambda_rm/gamma/rho, LoRA r=32/alpha=16 on the expert up/down
    projections: the paper's own Dolly15K (general instruction-following)
    hyperparameters (Table 7), the closest of their two workloads to this
    repo's own mixed instruction/math/code/multilingual dataset. The paper
    fully fine-tunes the router + gate projection (not LoRA'd) and only
    LoRAs the MLP up/down projections -- reproduced here by leaving the
    cache-layer router's own weight/bias trainable outside the LoRA config
    (see --cache-layer router lookup below) while --lora-r/--lora-alpha
    apply to the fused expert projections as elsewhere in this repo.
  - --lr/--num-steps/dataset/sequence-length follow this repo's own small-
    scale convention (not the paper's per-model epoch counts), for direct
    comparability with sft_baseline / cache_sft / temporal_moe /
    controller_baseline.
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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


def _load_adapter_tensors(peft_model, ckpt_dir, adapter_name="default"):
    """Load LoRA tensors from ckpt_dir's adapter_model.safetensors directly
    via load_state_dict, bypassing peft's set_peft_model_state_dict / load_
    adapter (broken for target_parameters/ParamWrapper adapters in peft 0.19
    -- 'PhimoeExperts' has no attribute 'weight' -- see finetune_moe_grpo.py,
    same helper). Shared by --resume (own save-dir) and --init-adapter (a
    different checkpoint, weights only)."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    model_keys = set(peft_model.state_dict().keys())
    remapped = {}
    skipped_router = 0
    for k, v in sd.items():
        nk = k.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight") \
              .replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
        if nk not in model_keys:
            head, _, tail = nk.rpartition(".")
            cand = f"{head}.modules_to_save.{adapter_name}.{tail}"
            if cand in model_keys:
                nk = cand
        if nk not in model_keys and ".mlp.router." in nk:
            # source checkpoint (e.g. sft_baseline) LoRAs the router;
            # melinoe fully fine-tunes the router directly instead (no LoRA
            # slot for it here) -- the router simply starts from the raw
            # base weights and gets its own full fine-tuning regardless, so
            # this delta is expected to have nowhere to go.
            skipped_router += 1
            continue
        remapped[nk] = v
    if skipped_router:
        print(f"[init] skipped {skipped_router} router LoRA tensors from "
              f"{ckpt_dir} (melinoe fully fine-tunes the router, no LoRA "
              f"slot for it)")
    res = peft_model.load_state_dict(remapped, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(
            f"adapter load failed, unexpected keys: {res.unexpected_keys[:5]}")
    missing_lora = [k for k in res.missing_keys if "lora" in k]
    if missing_lora:
        raise RuntimeError(
            f"adapter load failed, missing lora keys: {missing_lora[:5]}")
    return len(remapped)


EVAL_POOL_PER_SPLIT = 1000  # matches finetune_moe_grpo.py -- same held-out rows


# ============================================================================
# Dataset (same convention as finetune_moe_grpo.py / finetune_moe_controller.py)
# ============================================================================

def build_prompt_dataset(tokenizer, dataset_name, split, max_samples, prompt_len,
                         completion_len, seed, skip_first=EVAL_POOL_PER_SPLIT):
    """Nemotron rows -> Dataset({'prompt', 'target_ids'}); prompt is FILTERED
    to prompt_len tokens rather than truncated (see
    src.nemotron_data.sample_filtered_prompts for the scan/filter logic,
    shared with the grpo/sft/controller training scripts) -- identical
    convention to finetune_moe_controller.py's build_prompt_dataset."""
    from datasets import Dataset
    from src.nemotron_data import sample_filtered_prompts

    prompts, targets = [], []
    for ids, target_text in sample_filtered_prompts(
            tokenizer, dataset_name, split, max_samples, prompt_len, seed, skip_first):
        text = tokenizer.decode(ids)
        target_ids = tokenizer(target_text, truncation=True,
                               max_length=completion_len,
                               add_special_tokens=False)["input_ids"]
        prompts.append(text)
        targets.append(target_ids)
    return Dataset.from_dict({"prompt": prompts, "target_ids": targets}) \
        .shuffle(seed=seed)


def collate(batch, tokenizer, prompt_len, completion_len):
    """prompt (left-padded) + target_ids (right-padded); action_mask marks
    the target span -- only these positions are scored by the NLL and both
    auxiliary losses (the prompt only warms the cache state)."""
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
        input_ids[i, P - len(p):P] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, P - len(p):P] = 1
        if t:
            input_ids[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)
            attention_mask[i, P:P + len(t)] = 1
            action_mask[i, P:P + len(t)] = True
            labels[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "action_mask": action_mask, "labels": labels, "prompt_len": P}


def sft_nll_loss(lm_logits, labels):
    shift_logits = lm_logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1) \
        .gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return -(logp * mask).sum() / mask.sum().clamp_min(1)


# ============================================================================
# MELINOE auxiliary losses (Section 3.1.1 / Appendix C)
# ============================================================================

def melinoe_losses(peft_model, unwrapped_model, input_ids, attention_mask, labels,
                   cache_layer, cache_size, top_k, gamma, rho):
    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask,
                         output_router_logits=True, use_cache=False)
    lm_logits = outputs.logits
    B, T = input_ids.shape
    router_logits_f = outputs.router_logits[cache_layer].float().view(B, T, -1)
    p_f = F.softmax(router_logits_f, dim=-1)
    E = p_f.shape[-1]

    # frozen base router (adapter disabled) for the rank-matching target --
    # same "ref = adapters disabled" convention as finetune_moe_grpo.py's KL.
    # NOTE: disable_adapter() must be called on the UNWRAPPED peft model --
    # DistributedDataParallel.__getattr__ only forwards _parameters/_buffers/
    # _modules, not arbitrary methods like disable_adapter, so calling it on
    # the DDP-wrapped peft_model raises AttributeError under multi-GPU
    # accelerate (same fix as finetune_moe_grpo.py's PerplexityCallback,
    # which unwraps via accelerator.unwrap_model() first). This no_grad
    # forward also intentionally bypasses the DDP wrapper entirely -- no
    # gradient sync needed since nothing here is backpropagated.
    with torch.no_grad(), unwrapped_model.disable_adapter():
        base_out = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask,
                                   output_router_logits=True, use_cache=False)
        router_logits_b = base_out.router_logits[cache_layer].float().view(B, T, -1)
        p_b = F.softmax(router_logits_b, dim=-1)

    # --- Cache Simulation Loss (Eq. 4, 11) ---
    # straight-through top-K request vector: forward value is the hard 0/1
    # Top-K indicator (sums to top_k), backward gradient flows as if
    # r_f = p_f (Top-K itself has no useful gradient).
    topk_idx = p_f.topk(top_k, dim=-1).indices
    r_hard = torch.zeros_like(p_f).scatter_(-1, topk_idx, 1.0)
    r_f = r_hard + (p_f - p_f.detach())

    C = float(cache_size)
    K = float(top_k)
    c = torch.full((B, E), C / E, device=p_f.device, dtype=torch.float32)
    Z = torch.ones(B, device=p_f.device, dtype=torch.float32)
    lcs_per_t = []
    for t in range(T):
        r_t = r_f[:, t, :]
        # loss uses c(t) as-of BEFORE this token's own request updates it --
        # "was this token's pick already warm from prior history" (matches
        # eval/eval_router.py's hit_ratio: `cached = cache.experts` read
        # before the current token's cache.access() calls).
        lcs_per_t.append((r_t * (1.0 - c)).sum(-1))
        r_hard_t = r_hard[:, t, :].detach()
        Z_next = gamma * Z + K / C
        c = (gamma * Z.unsqueeze(-1) * c + r_hard_t) / Z_next.unsqueeze(-1)
        Z = Z_next
    lcs_per_t = torch.stack(lcs_per_t, dim=1)      # [B, T]

    # --- Rank Matching Loss (Eq. 5, 12) ---
    diff_b = p_b.unsqueeze(-1) - p_b.unsqueeze(-2)  # [B,T,E,E] i,j = p_b_i - p_b_j
    mask_b = (diff_b > 0).float()
    diff_f = p_f.unsqueeze(-1) - p_f.unsqueeze(-2)
    margin_term = (rho - diff_f).clamp_min(0.0)
    m_t = (mask_b * margin_term).sum(dim=(-1, -2))  # [B, T]

    action = (labels[:, :] != -100).float()
    # both losses are defined at token t using router probs at t (not a
    # next-token shift, unlike the NLL loss) -- align to the same action
    # positions the NLL scores (labels != -100 at the same index t).
    n_tok = action.sum().clamp_min(1)
    lcs = (lcs_per_t * action).sum() / n_tok
    lrm = (m_t * action).sum() / n_tok

    with torch.no_grad():
        cache_hit_proxy = 1.0 - lcs

    return {"lm_logits": lm_logits, "lcs": lcs, "lrm": lrm,
           "cache_hit_proxy": cache_hit_proxy}


class Preemption:
    """Same SLURM preemption pattern as finetune_moe_controller.py: SIGUSR1/
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
        description="MELINOE baseline (Raje, Nayak & Joshi 2026, "
                    "arXiv:2602.11192), minimal single-layer reimplementation")
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
                        help="ours, not the paper's own 1e-5 -- matches the "
                             "LR used across this repo's other small-scale "
                             "runs for direct comparability")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm (HF Trainer's own "
                             "default) -- the cache-layer router here is "
                             "fully unfrozen (no LoRA down-scaling), so an "
                             "occasional large gradient can otherwise push "
                             "it (or a fused-expert LoRA tensor) to NaN in "
                             "a single optimizer step without the logged "
                             "loss/lcs/lrm (computed pre-update) ever "
                             "showing anything wrong.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-cache-reinforce")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints/melinoe_baseline")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-adapter", type=str, default=None,
                        help="Initialize the LoRA adapter's weights from a "
                             "different checkpoint's adapter (e.g. a "
                             "completed SFT run) before MELINOE training "
                             "starts -- weights only, no optimizer/step "
                             "state. Ignored when --resume finds a "
                             "checkpoint already in --save-dir.")

    cache_group = parser.add_argument_group("Cache layer / MELINOE loss (paper's own hparams)")
    cache_group.add_argument("--cache-layer", type=int, default=-1, help="-1 = middle layer")
    cache_group.add_argument("--cache-size", type=int, default=4,
                             help="C, soft cache capacity -- matches E/4 for "
                                  "this model's 16 local experts, exactly "
                                  "the paper's own C=E/4 rule")
    cache_group.add_argument("--top-k", type=int, default=None,
                             help="K, experts requested per token -- default "
                                  "None reads the model's own "
                                  "num_experts_per_tok")
    cache_group.add_argument("--gamma", type=float, default=0.9,
                             help="cache decay -- paper's fixed value across "
                                  "all experiments")
    cache_group.add_argument("--rho", type=float, default=0.1,
                             help="rank-matching margin -- paper's fixed value")
    cache_group.add_argument("--lambda-cs", type=float, default=0.5,
                             help="paper's Dolly15K (general instruction) "
                                  "coefficient -- closest of their two "
                                  "workloads to this repo's own dataset")
    cache_group.add_argument("--lambda-rm", type=float, default=0.1,
                             help="paper's Dolly15K coefficient")

    lora_group = parser.add_argument_group("LoRA (paper's own hparams for the expert projections)")
    lora_group.add_argument("--lora-r", type=int, default=32)
    lora_group.add_argument("--lora-alpha", type=int, default=16)

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
                          args.model, "melinoe_v1")).encode()).hexdigest()[:12]
    cache_file = Path("data") / f"melinoe_prompt_cache_{key}.pkl"
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
        # NOTE: unlike finetune_moe_grpo.py/finetune_moe_sft.py's LoRA
        # config, "router.weight" is deliberately NOT in target_parameters
        # here -- the paper fully fine-tunes the router (see below), only
        # LoRA-ing the expert MLP projections.
        lora_kwargs = dict(
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
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
    top_k = args.top_k or model.config.num_experts_per_tok
    print(f"[melinoe] layer {args.cache_layer}/{num_layers}, C={args.cache_size}, "
         f"K={top_k}, lambda_cs={args.lambda_cs}, lambda_rm={args.lambda_rm}")

    router_module = None
    for name, mod in model.named_modules():
        if (type(mod).__name__ == "PhimoeTopKRouter"
                and f"layers.{args.cache_layer}.mlp" in name):
            router_module = mod
            break
    assert router_module is not None, f"no router found at layer {args.cache_layer}"

    peft_model = get_peft_model(model, lora_config)
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    # fully fine-tune the cache-layer router (paper: "update only the router
    # weights and gate projection") -- router_module is the SAME object peft
    # wraps (get_peft_model mutates model in place, doesn't copy it), so this
    # unfreezes exactly the parameters actually used in the forward pass.
    for p in router_module.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=args.lr)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.prompt_len, args.completion_len))

    peft_model, optimizer, dataloader = accelerator.prepare(
        peft_model, optimizer, dataloader)

    if accelerator.is_main_process and args.wandb_run_name:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                  config=vars(args))

    save_dir = Path(args.save_dir)
    start_step = 0
    ckpt = None
    if args.resume and save_dir.is_dir():
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(save_dir)
    if ckpt:
        print(f"Resuming from {ckpt}")
        unwrapped = accelerator.unwrap_model(peft_model)
        n = _load_adapter_tensors(unwrapped, ckpt)
        print(f"[resume] manually loaded {n} adapter tensors")
        router_state = torch.load(Path(ckpt) / "router.pt", map_location="cpu")
        accelerator.unwrap_model(router_module).load_state_dict(router_state["router"])
        optimizer.load_state_dict(router_state["optimizer"])
        start_step = router_state["step"]
    elif args.init_adapter:
        unwrapped = accelerator.unwrap_model(peft_model)
        n = _load_adapter_tensors(unwrapped, args.init_adapter)
        print(f"[init] loaded {n} adapter tensors from {args.init_adapter}")
        router_ckpt = Path(args.init_adapter) / "router.pt"
        if router_ckpt.exists():
            router_state = torch.load(router_ckpt, map_location="cpu")
            accelerator.unwrap_model(router_module).load_state_dict(router_state["router"])
            print(f"[init] loaded router weights from {router_ckpt}")

    preemption = Preemption()
    step = start_step
    data_iter = iter(dataloader)
    peft_model.train()
    while step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(peft_model):
            out = melinoe_losses(
                peft_model, accelerator.unwrap_model(peft_model),
                batch["input_ids"], batch["attention_mask"],
                batch["labels"], args.cache_layer, args.cache_size, top_k,
                args.gamma, args.rho)
            nll = sft_nll_loss(out["lm_logits"], batch["labels"])
            loss = nll + args.lambda_cs * out["lcs"] + args.lambda_rm * out["lrm"]
            accelerator.backward(loss)
            grad_norm = None
            skip_step = False
            if accelerator.sync_gradients:
                # Unlike the other scripts' LoRA-only params, this router is
                # FULLY unfrozen (no LoRA down-scaling of its updates).
                # Clipping alone isn't enough: observed once that a specific
                # batch produces an outright NaN gradient (not just a large
                # one) straight out of backward() -- likely a near-zero
                # denominator edge case in the cache-state recursion or the
                # rank-matching indicator -- which no rescaling can fix
                # (NaN * scalar is still NaN). loss/lcs/lrm logged that step
                # were all finite (computed BEFORE the update), so this is
                # invisible without checking grad_norm itself. Skip the
                # optimizer step entirely rather than apply a corrupting
                # update; the model just keeps last step's clean weights and
                # moves on to the next (different) batch.
                grad_norm = accelerator.clip_grad_norm_(
                    peft_model.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skip_step = True
                    print(f"[warn] non-finite grad_norm ({grad_norm}) -- "
                          f"skipping this optimizer step", flush=True)
            if not skip_step:
                optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                logs = {
                    "loss": loss.item(), "sft_nll": nll.item(),
                    "lcs": out["lcs"].item(), "lrm": out["lrm"].item(),
                    "cache_hit_proxy": out["cache_hit_proxy"].item(),
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
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
                        "router": accelerator.unwrap_model(router_module).state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step,
                    }, ckpt_dir / "router.pt")
                    print(f"[save] checkpoint-{step} -> {ckpt_dir}", flush=True)
                if preemption.triggered:
                    print("[preemption] stopping after checkpoint", flush=True)
                    break

    print("done", flush=True)


if __name__ == "__main__":
    main()
