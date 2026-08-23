#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete.py --model allenai/OLMoE-1B-7B-0125-Instruct --variant base  \
    --cache-size 16 --cache-experts-per-token 8 --cache-topk
