"""Standard LoRA SFT baseline (TRL's SFTTrainer) for the MoE cache-reward
experiments -- no RL, no cache reward, no temporal mixin. Trains directly on
the dataset's own ground-truth assistant completions (teacher-forced NLL,
loss masked to completion tokens only), so it's the "what does plain SFT
alone get you" reference point for the GRPO+cache-reward and temporal-moe
runs in scripts/train/finetune_moe_grpo.py.

Same dataset (Nemotron-v2 math/code), same held-out eval pool convention
(first EVAL_POOL_PER_SPLIT rows of each split reserved, never trained on),
same fused-expert LoRA target_parameters setup for phimoe, and the same
SLURM preemption/resume pattern, so checkpoints from this script slot
directly into the eval/eval_router.py / eval/eval_benchmarks.py pipeline alongside
the RL-trained checkpoints.
"""

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from transformers import AutoTokenizer, TrainerCallback
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

# peft 0.19 x transformers 5.8 bug -- see finetune_moe_grpo.py for the same
# workaround; kept identical so resumed/loaded adapters behave the same way
# across every script in this pipeline.
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


def build_sft_dataset(tokenizer, dataset_name, split, max_samples, prompt_len,
                      completion_len, seed, skip_first=EVAL_POOL_PER_SPLIT,
                      max_scan_per_split=50_000):
    """Nemotron rows -> Dataset({'prompt', 'completion'}), both conversational
    (list of chat messages) so SFTTrainer applies the chat template itself
    and masks the loss to the completion (assistant) turn only.

    Prompts are FILTERED to prompt_len tokens, not truncated: rows whose
    prompt is longer than prompt_len are dropped rather than clipped (see
    src.nemotron_data.sample_filtered_prompts for the scan/filter logic,
    shared with the grpo/controller/melinoe training scripts). Completions
    are still truncated to completion_len -- only used for the loss, no
    generation happens against them. Same skip_first eval-pool reservation
    as finetune_moe_grpo.py."""
    from datasets import Dataset
    from src.nemotron_data import sample_filtered_prompts

    prompts, completions = [], []
    for ids, target_text in sample_filtered_prompts(
            tokenizer, dataset_name, split, max_samples, prompt_len, seed, skip_first,
            max_scan_per_split=max_scan_per_split):
        text = tokenizer.decode(ids)
        c_ids = tokenizer(target_text, truncation=True,
                          max_length=completion_len,
                          add_special_tokens=False)["input_ids"]
        completion_text = tokenizer.decode(c_ids)
        prompts.append([{"role": "user", "content": text}])
        completions.append([{"role": "assistant", "content": completion_text}])
    return Dataset.from_dict(
        {"prompt": prompts, "completion": completions}).shuffle(seed=seed)


class PreemptionCallback(TrainerCallback):
    """Identical to finetune_moe_grpo.py's -- SLURM sends SIGUSR1 shortly
    before the time limit (or on preemption of a --requeue job); this just
    flags it, and the Trainer's own step loop checkpoints + stops cleanly at
    the next step boundary. Pairs with --resume + a fixed --save-dir."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Plain LoRA SFT baseline (TRL SFTTrainer) for the MoE "
                    "cache-reward experiments -- no RL, no cache reward")
    parser.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    parser.add_argument("--dataset", type=str,
                        default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--dataset-split", type=str, default="math,code")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-scan-per-split", type=int, default=50_000,
                        help="rows scanned (streamed+tokenized for the length "
                             "filter) per split before the reservoir sample is "
                             "finalized -- raise this to draw the sample from "
                             "more of the dataset instead of just the first "
                             "N rows of each split (see "
                             "src.nemotron_data.sample_filtered_prompts)")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--completion-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-epochs", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Standard LoRA SFT lr (much higher than the "
                             "~3e-5 GRPO/RL range -- SFT gradients are much "
                             "less noisy)")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="moe-cache-reinforce")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str,
                        default="checkpoints/sft_baseline")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in --save-dir")
    parser.add_argument("--temporal", action="store_true",
                        help="Wrap MoE blocks with boundary prediction + "
                             "hold/switch routing (TemporalWrapMixin, STE "
                             "variant) before SFT, matching the mixin used "
                             "downstream by finetune_moe_grpo.py --temporal "
                             "-- so a later DAPO stage can --init-adapter "
                             "from this checkpoint and keep training the "
                             "same wrapped architecture.")

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

    print(f"Building SFT dataset {args.dataset} [{args.dataset_split}] ...")
    import hashlib, pickle
    key = hashlib.md5(str((args.dataset, args.dataset_split, args.max_samples,
                           args.prompt_len, args.completion_len, args.seed,
                           args.model, args.max_scan_per_split,
                           "sft_v2_scan")).encode()).hexdigest()[:12]
    cache_file = Path("data") / f"sft_prompt_cache_{key}.pkl"
    from accelerate import PartialState
    with PartialState().main_process_first():
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                prompts_list, completions_list = pickle.load(f)
            from datasets import Dataset
            train_dataset = Dataset.from_dict(
                {"prompt": prompts_list, "completion": completions_list})
            print(f"[data] loaded cached prompts from {cache_file}")
        else:
            train_dataset = build_sft_dataset(
                tokenizer, args.dataset, args.dataset_split, args.max_samples,
                args.prompt_len, args.completion_len, args.seed,
                max_scan_per_split=args.max_scan_per_split)
            if PartialState().is_main_process:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump((list(train_dataset["prompt"]),
                                list(train_dataset["completion"])), f)
                print(f"[data] cached prompts to {cache_file}")
    print(f"[data] {len(train_dataset)} training rows")

    # Same fused-expert LoRA setup as finetune_moe_grpo.py -- phimoe's router
    # returns a tuple (LoRA's module wrapper can't handle that) and its
    # expert weights are fused 3D params, so both are targeted as
    # target_parameters instead of ordinary target_modules.
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(args.model)
    target_modules = list(args.lora_target_modules)
    lora_kwargs = {}
    if model_cfg.model_type == "phimoe":
        target_modules = [t for t in target_modules
                          if t not in ("router", "gate")]
        lora_kwargs = dict(
            target_parameters=["experts.gate_up_proj", "experts.down_proj",
                               "router.weight"],
            rank_pattern={r".*\.gate_up_proj": args.lora_r * 2},
            alpha_pattern={r".*\.gate_up_proj": args.lora_alpha * 2},
        )

    # Temporal MoE: load the model ourselves and wrap the MoE blocks before
    # peft sees it, exactly like finetune_moe_grpo.py --temporal. The
    # boundary-prediction layers (term_proj1/2) are new non-LoRA params ->
    # trained + checkpointed via peft modules_to_save.
    model_or_id = args.model
    if args.temporal:
        from transformers import AutoModelForCausalLM
        from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin
        model_or_id = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        TemporalWrapMixin.apply(model_or_id, TemporalWrapConfig(ste=True))
        lora_kwargs["modules_to_save"] = ["term_proj1", "term_proj2"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # ParamWrapper (fused experts) forbids dropout
        target_modules=target_modules,
        **lora_kwargs,
    )

    sft_config = SFTConfig(
        output_dir=args.save_dir,
        run_name=args.wandb_run_name,
        report_to=["wandb"],
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.num_steps,
        logging_steps=1,
        save_steps=args.save_every,
        # No pruning: earlier checkpoints (e.g. checkpoint-32) are load-
        # bearing anchors for other scripts' --init-adapter (controller_
        # baseline, sft_then_dapo, melinoe all init from a specific SFT
        # step-count) -- save_total_limit=3 silently deleted checkpoint-32
        # out from under them the moment this run was resumed past step 32,
        # breaking every downstream --init-adapter load with a bare
        # FileNotFoundError.
        save_total_limit=None,
        max_length=args.prompt_len + args.completion_len,
        packing=False,  # keep prompt/completion loss masking exact per-row
        # No trust_remote_code / model_init_kwargs when --temporal already
        # loaded + wrapped a model instance (matches finetune_moe_grpo.py).
        model_init_kwargs=None if args.temporal else {"dtype": torch.bfloat16},
        # TRL's default (0.001) enables its chunked-CE MoE aux-loss path,
        # which reads text_config.num_experts -- a Mixtral-style attribute
        # name PhimoeConfig doesn't have (it uses num_local_experts),
        # crashing every forward pass. 0 disables that path entirely; it
        # also matches finetune_moe_grpo.py's --router-aux-loss-coef 0.0
        # (load-balancing pushes toward uniform expert usage, the opposite
        # of what the cache-reward/consolidation experiments want anyway).
        router_aux_loss_coef=0.0,
    )

    trainer = SFTTrainer(
        model=model_or_id,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.add_callback(PreemptionCallback())

    resume_ckpt = None
    if args.resume and Path(args.save_dir).is_dir():
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(args.save_dir)
        if resume_ckpt:
            print(f"Resuming from {resume_ckpt}")

    if resume_ckpt:
        # peft 0.19 cannot load target_parameters (ParamWrapper) adapters via
        # set_peft_model_state_dict -- see finetune_moe_grpo.py for the same
        # workaround, kept identical.
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
                    # modules_to_save entries (e.g. term_proj*, --temporal
                    # only): saved without the wrapper infix.
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
