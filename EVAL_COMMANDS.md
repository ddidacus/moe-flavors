# eval_complete.py run commands

One entry per `(model, variant)` to evaluate. Each entry gives the direct
`bash` command and the equivalent `sbatch` submission (single GPU, no
cluv -- see `scripts/slurm/eval_complete.sbatch`). All commands assume
you're in the repo root with `.venv` set up (`uv sync`).

`--cache-size`/`--cache-experts-per-token`/`--cache-topk` match how each
checkpoint was actually trained (4/2 for Phi-tiny-MoE, 16/8 for OLMoE --
see `scripts/cluv/train_*.sh`). Checkpoint paths are this project's
tamia-side paths; adjust to wherever your copies live.

Output lands under `evals/<model_name>/<date>/<variant>/eval_<part>.json`
(plain variants) or `evals/<model_name>/<date>/cache<N>/eval_<part>.json`
plus two scaling plots (cache-conditioned variants).

---

## phi base

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant base \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant base \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi sft_baseline

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant sft_baseline \
    --checkpoint-dir checkpoints/sft_baseline_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant sft_baseline \
    --checkpoint-dir checkpoints/sft_baseline_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi cache_sft

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant cache_sft \
    --checkpoint-dir checkpoints/cache_sft_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant cache_sft \
    --checkpoint-dir checkpoints/cache_sft_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi cache_reward

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant cache_reward \
    --checkpoint-dir checkpoints/cache_reward_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant cache_reward \
    --checkpoint-dir checkpoints/cache_reward_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi controller_baseline

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant controller_baseline \
    --checkpoint-dir checkpoints/controller_baseline_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant controller_baseline \
    --checkpoint-dir checkpoints/controller_baseline_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi melinoe_baseline

```bash
python scripts/eval/eval_complete.py \
    --model microsoft/Phi-tiny-MoE-instruct --variant melinoe_baseline \
    --checkpoint-dir checkpoints/melinoe_baseline_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct --variant melinoe_baseline \
    --checkpoint-dir checkpoints/melinoe_baseline_tamia_v2 \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
```

## phi prompt_conditioned (cache-size sweep 2/4/8)

Uses `eval_complete_cache_conditioned.py` instead -- this checkpoint
expects a `[CACHE_SIZE=X]` prompt prefix, so it's evaluated once per
cache size it was trained on, not as a single unconditioned run.

```bash
python scripts/eval/eval_complete_cache_conditioned.py \
    --model microsoft/Phi-tiny-MoE-instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \
    --cache-sizes 2,4,8 --cache-experts-per-token 2 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete_cache_conditioned.sbatch \
    --model microsoft/Phi-tiny-MoE-instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps \
    --cache-sizes 2,4,8 --cache-experts-per-token 2 --cache-topk
```

## olmoe base

```bash
python scripts/eval/eval_complete.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct --variant base \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model allenai/OLMoE-1B-7B-0125-Instruct --variant base \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk
```

## olmoe sft_baseline

```bash
python scripts/eval/eval_complete.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct --variant sft_baseline \
    --checkpoint-dir checkpoints/sft_baseline_olmoe_tamia \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete.sbatch \
    --model allenai/OLMoE-1B-7B-0125-Instruct --variant sft_baseline \
    --checkpoint-dir checkpoints/sft_baseline_olmoe_tamia \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk
```

## olmoe prompt_conditioned (cache-size sweep 8/16/32)

```bash
python scripts/eval/eval_complete_cache_conditioned.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_olmoe_tamia \
    --cache-sizes 8,16,32 --cache-experts-per-token 8 --cache-topk
```
```bash
sbatch scripts/slurm/eval_complete_cache_conditioned.sbatch \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_olmoe_tamia \
    --cache-sizes 8,16,32 --cache-experts-per-token 8 --cache-topk
```

---

## Running two at once (packed job)

To run any two of the above together on one 2-GPU allocation (matching
this project's own submission pattern), write each command's script body
to a file and pass both to the pair job:

```bash
cat > /tmp/job_a.sh <<'EOF'
#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete.py --model microsoft/Phi-tiny-MoE-instruct \
    --variant base --cache-size 4 --cache-experts-per-token 2 --cache-topk
EOF

cat > /tmp/job_b.sh <<'EOF'
#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete.py --model microsoft/Phi-tiny-MoE-instruct \
    --variant sft_baseline --checkpoint-dir checkpoints/sft_baseline_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
EOF

sbatch scripts/slurm/eval_complete_pair.sbatch /tmp/job_a.sh /tmp/job_b.sh
```

Or just run the whole sweep at once: `bash
scripts/slurm/submit_eval_sweep.sh` (plain variants) and `bash
scripts/slurm/submit_eval_sweep_cache_conditioned.sh` (the two
prompt_conditioned sweeps) -- see `README.md`.
