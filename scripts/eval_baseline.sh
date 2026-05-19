#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --output=eval_baseline_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --time=2:00:00

source .venv/bin/activate
export HF_HOME=/home/mila/d/diego.calanzone/scratch/cache
export UV_CACHE_DIR=/home/mila/d/diego.calanzone/scratch/cache
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_DIR="eval_results/baseline_qwen3_0.6b"
mkdir -p "$OUTPUT_DIR"

python -c "
import json, torch, wandb, lm_eval
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM

MODEL_NAME = 'Qwen/Qwen3-0.6B'
TASKS = [
    'mmlu_abstract_algebra',
    'mmlu_college_mathematics',
    'mmlu_high_school_mathematics',
    'mmlu_elementary_mathematics',
]
OUTPUT_DIR = '$OUTPUT_DIR'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda().eval()

wandb.init(
    project='moe-chunking-poc',
    job_type='eval',
    name='eval-baseline-qwen3-0.6b',
    config={
        'base_model': MODEL_NAME,
        'tasks': TASKS,
        'limit': None,
        'num_fewshot': None,
        'moe_type': 'baseline',
    },
)

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
results = lm_eval.simple_evaluate(model=lm, tasks=TASKS, batch_size='auto')

print('\n' + '=' * 60)
print('RESULTS')
print('=' * 60)
for task_name, task_results in results['results'].items():
    print(f'\n--- {task_name} ---')
    for metric, value in sorted(task_results.items()):
        if isinstance(value, float):
            print(f'  {metric}: {value:.4f}')
        else:
            print(f'  {metric}: {value}')

wandb_metrics = {}
for task_name, task_results in results['results'].items():
    for metric, value in task_results.items():
        if isinstance(value, (int, float)):
            wandb_metrics[f'eval/harness_{task_name}/{metric}'] = value
wandb.log(wandb_metrics)

out_path = Path(OUTPUT_DIR)
out_path.mkdir(parents=True, exist_ok=True)
(out_path / 'eval_results.json').write_text(json.dumps(results['results'], indent=2, default=str))
print(f'\nResults saved to {out_path / \"eval_results.json\"}')

wandb.finish()
"
