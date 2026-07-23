# Training setup — small scale (planned, 2026-07-23)

Companion to [06-training-setup.md](06-training-setup.md) ("large" scale,
10k sequences). Same four models, same per-model lr/batch/grad-accum, same
1024+1024 sequence length, same 4x A100-80GB hardware -- the only thing that
changes is dataset size: **~2,000 sequences**, closer to the debug-scale
runs already completed in past days (the 150-step runs at 512+512 covered
roughly 400-2,400 sequences depending on the model; this puts all four on
the same, larger-but-still-quick footing at the new 1024+1024 length).

Implemented via `scripts/train_small_scale.sh` (submits all four jobs) --
reuses the exact same per-model sbatch scripts as the large-scale run, just
with `MAX_SAMPLES=2000` and a proportionally smaller `NUM_STEPS`.

## Models

Identical to the large-scale plan -- see
[06-training-setup.md](06-training-setup.md#models) for the full
model/script/mechanism table (`sft_baseline`, `cache_sft`, `temporal_moe`,
`controller_baseline`).

## Dataset

- **Source:** `nvidia/Nemotron-Post-Training-Dataset-v2`, all 9 splits
  (`stem, chat, math, code, multilingual_ja, multilingual_de,
  multilingual_it, multilingual_es, multilingual_fr`), randomly sampled
  (reservoir sampling, Algorithm R, per split).
- **Size:** **2,000 sequences total**.
- Same held-out eval pool convention (first 1,000 rows/split reserved,
  unchanged).

## Sequence length (all models)

Same as large scale: `--prompt-len 1024 --completion-len 1024`.

## Hyperparameters per model

Identical lr/batch/grad-accum/hardware to the large-scale plan (see
[06-training-setup.md](06-training-setup.md#hyperparameters-per-model) for
the full table and rationale) -- only `NUM_STEPS` changes below.

| model | lr | batch/device | grad accum |
| --- | --- | --- | --- |
| `sft_baseline` | 1e-4 | 4 | 4 |
| `cache_sft` | 1e-4 | 8 | 2 |
| `temporal_moe` | 1e-4 | 8 | 2 |
| `controller_baseline` | 1e-4 (LM); 1e-5 (controller heads) | 4 | 4 |

## Gradient steps (1 epoch over 2,000 sequences)

| model | effective throughput/step | steps for 2k |
| --- | --- | --- |
| `sft_baseline` | 64 sequences/step | **32** |
| `cache_sft` | 8 prompts/step (x8 completions each) | **250** |
| `temporal_moe` | 8 prompts/step | **250** |
| `controller_baseline` | 64 sequences/step | **32** |

## Estimated GPU-hours

Same per-step GPU-second costs as the large-scale plan (identical configs,
just fewer steps).

| model | GPU-s/step | steps | **total GPU-hours** | wall-clock (4 GPUs) |
| --- | --- | --- | --- | --- |
| `sft_baseline` | 23.6 (measured) | 32 | **~0.21** | ~3.2 min |
| `cache_sft` | 571.6 (extrapolated) | 250 | **~39.7** | ~9.9 hours |
| `temporal_moe` | 939.0 (extrapolated) | 250 | **~65.2** | ~16.3 hours |
| `controller_baseline` | 126.0 (extrapolated) | 32 | **~1.1** | ~17 min |

`sft_baseline` and `controller_baseline` are cheap enough to fit
comfortably inside `short-unkillable`'s 3h cap. `cache_sft`/`temporal_moe`
still exceed it (dominated by rollout generation, not dataset size) -- both
run on `long` with a 1-day budget in `scripts/train_small_scale.sh`.

## Suggested use

Run this scale first as a fast end-to-end check of the whole pipeline
(dataset sampling, all four training scripts, checkpoint saving/resume,
downstream eval) before committing to the large-scale run's ~2-3.4 day
`cache_sft`/`temporal_moe` jobs.
