#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
python scripts/eval/eval_complete.py --model microsoft/Phi-tiny-MoE-instruct --variant sft_baseline --checkpoint-dir checkpoints/sft_baseline_tamia \
    --cache-size 4 --cache-experts-per-token 2 --cache-topk
