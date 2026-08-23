#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete_cache_conditioned.py --model microsoft/Phi-tiny-MoE-instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_tamia_200steps --cache-sizes 2,4,8 \
    --cache-experts-per-token 2 --cache-topk
