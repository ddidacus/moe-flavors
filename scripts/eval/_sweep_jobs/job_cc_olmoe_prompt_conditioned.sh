#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete_cache_conditioned.py --model allenai/OLMoE-1B-7B-0125-Instruct \
    --checkpoint-dir checkpoints/prompt_conditioned_olmoe_tamia --cache-sizes 8,16,32 \
    --cache-experts-per-token 8 --cache-topk
