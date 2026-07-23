# Training setup — large scale (planned, 2026-07-23)

See [08-training-setup-small.md](08-training-setup-small.md) for the
smaller/faster companion scale. Same models, same per-model hyperparameters,
same sequence length and hardware -- only the dataset size (and therefore
step count) differs.

Compute-matched replan across all four experiment lines, dataset cap lowered
from an earlier 100k-sequence plan to **10,000** once the GRPO variants' cost
(rollout generation, not just teacher-forced NLL) made a full epoch over 100k
prompts a ~20-34 day / ~2,000-3,300 GPU-hour proposition per run.

Implemented as of 2026-07-23: `scripts/train_large_scale.sh` submits all
four jobs with the config below (all-splits random sampling via reservoir
sampling, and the controller script's multi-GPU launch, are both live).

## Models

| model | script | mechanism |
| --- | --- | --- |
| `sft_baseline` | `scripts/finetune_moe_sft.py` | plain LoRA SFT (TRL `SFTTrainer`), no RL, no cache reward -- standard-finetuning reference point |
| `cache_sft` | `scripts/finetune_moe_grpo.py --soft-cache` | GRPO/DAPO + dense-router soft cache-hit reward + SFT NLL term, non-temporal |
| `temporal_moe` | `scripts/finetune_moe_grpo.py --temporal` | same, + boundary/hold-switch mixin (`src/temporal_moe_wrapper.py`) at the cache layer; forces `--cache-topk` (soft-cache is incompatible with the mixin) |
| `controller_baseline` | `scripts/finetune_moe_controller.py` | minimal reimplementation of the Option-Critic MoE controller (Shen & Henderson 2026, `princeton-polaris-lab/rl_moe`) -- termination + Plackett-Luce selection + value heads, self-distillation (closed-form `log Z` reward), teacher-forced (no on-policy rollout needed since reward_type=kl) |

## Dataset

- **Source:** `nvidia/Nemotron-Post-Training-Dataset-v2`, **all splits** (chat,
  code, math, STEM, multi-en/de/es/fr/it/ja) -- widened from the earlier
  math+code-only setup.
- **Size:** 10,000 sequences total, **randomly sampled** across all splits
  (not a fixed per-split allocation).
- First `EVAL_POOL_PER_SPLIT` (1,000) rows of each split reserved as held-out
  eval pool, never trained on (existing convention, unchanged).

## Sequence length (all models)

- `--prompt-len 1024`
- `--completion-len 1024` (GRPO: also `--max-completion-length` /
  rollout generation cap)

## Hyperparameters per model

Learning rate kept at each model's own empirically-best value from prior
runs (not re-tuned here). All models train on **4x A100-80GB**
(`--gres=gpu:a100l:4`, `accelerate launch --multi_gpu --num_processes 4`) --
`controller_baseline` needs its launch script updated to this pattern (it
previously ran single-GPU, plain `python`, and its one completed run
actually landed on a 48GB L40S, not an 80GB A100).

| model | lr | batch/device | grad accum | other |
| --- | --- | --- | --- | --- |
| `sft_baseline` | 1e-4 | 4 | 4 | LoRA r=16, α=32 |
| `cache_sft` | 1e-4 | 8 | 2 | `num_generations=8` (forces batch to be a multiple of 8 -- halved from the 512-seq run's batch=16, since a straight 4x reduction would go below that floor); β=0.08, rl_coef=2.0, sft_coef=0.5, cache_size=4, cache_layer=-1 (middle) |
| `temporal_moe` | 1e-4 | 8 | 2 | same as `cache_sft` plus `--temporal`; STE boundary mixin, gradient checkpointing kept ON (see note below) |
| `controller_baseline` | 1e-4 (LM/LoRA); 1e-5 (controller heads, separate param group) | 4 | 4 | deliberation_cost η=0.02, value_coef=0.1, cache_size(k)=4, expert_embed_dim=32, mlp_hidden=512 |

Batch sizes above are the 1024+1024 values, adjusted down from the
512+512 configs that were empirically validated not to OOM on 80GB (a
directly-measured 512→1024+1024 scaling factor of **~2.23x per-step wall
time** was used to extrapolate memory/compute needs where not yet tested at
the new length; `sft_baseline`'s 1024+1024 config *has* been directly run
and confirmed OOM-free).

Known pitfall carried over from earlier debugging: `--temporal` forces
`gradient_checkpointing=False` UNLESS `ste=True` (the STE boundary variant,
which is what this script always uses) -- checkpointing is safe and kept ON
for STE; leaving it off unconditionally OOMs regardless of batch size, since
the fused-expert LoRA weight (`W + delta_weight`) is materialized per MoE
layer and stays resident across all layers in one forward pass without it.

## Gradient steps (1 epoch over 10,000 sequences)

| model | effective throughput/step | steps for 10k |
| --- | --- | --- |
| `sft_baseline` | 4 x 4 x 4 GPUs = 64 sequences/step | **157** |
| `cache_sft` | 8/8 x 2 x 4 GPUs = 8 prompts/step (x8 completions each) | **1,250** |
| `temporal_moe` | 8/8 x 2 x 4 GPUs = 8 prompts/step | **1,250** |
| `controller_baseline` | 4 x 4 x 4 GPUs = 64 sequences/step | **157** |

## Estimated GPU-hours

Per-step cost = measured wall-clock at the equivalent 512+512 config x 2.23
(seq-length scaling) x GPU count, except `sft_baseline` which is measured
directly at 1024+1024 (5.91s/step, batch=4/accum=4, 4 GPUs).

| model | GPU-s/step | steps | **total GPU-hours** | wall-clock (4 GPUs) |
| --- | --- | --- | --- | --- |
| `sft_baseline` | 23.6 (measured) | 157 | **~1.0** | ~15.5 min |
| `cache_sft` | 571.6 (extrapolated) | 1,250 | **~198.5** | ~49.6 hours (~2.1 days) |
| `temporal_moe` | 939.0 (extrapolated) | 1,250 | **~326.0** | ~81.5 hours (~3.4 days) |
| `controller_baseline` | 126.0 (extrapolated) | 157 | **~5.5** | ~1.4 hours |

The two GRPO-based runs (`cache_sft`, `temporal_moe`) remain the dominant
cost even at only 10k sequences -- their expense scales with rollout
generation (up to 8 x 1,024 generated tokens per training prompt), not
dataset size alone.

## Known-good reference runs this plan builds on

- `checkpoints/sft_phi-tiny-moe-instruct_mathcode_lr1e-4` (512+512, 150
  steps) and `..._seq1024-1024` (1024+1024, 150 steps) -- both completed
  cleanly, confirm the 1024+1024/batch=4/accum=4 config is OOM-safe on 4x
  A100-80GB.
- `checkpoints/grpo_phi-tiny-moe-instruct_cache_mathcode_sft0.5_b0.08_c4_softall_lr1e-4_dbg150_rl2.0`
  (`cache_sft`, 512+512, 150/150 steps completed) -- source of the lr/β/
  rl_coef/sft_coef choices kept here.
- `checkpoints/grpo_phi-tiny-moe-instruct_cache_mathcode_sft0.5_b0.08_c4_topk2_tmoeN8_lr1e-4_dbg150_rl2.0`
  (`temporal_moe`, 512+512, only 50/150 steps survived -- job crashed at the
  `short-unkillable` 3h wall-time without a clean preemption save; actual
  training reached ~step 97). Undertrained relative to `cache_sft`; worth
  rerunning to completion under this new plan.
- `checkpoints/controller_phi-tiny-moe-instruct_mathcode_c4_eta0.02`
  (`controller_baseline`, 512+512, 150/150 steps completed, single L40S GPU)
  -- source of η/value_coef/architecture choices kept here.
