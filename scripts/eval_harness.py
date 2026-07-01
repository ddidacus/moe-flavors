"""
Evaluate a local MoE checkpoint using EleutherAI's lm-evaluation-harness.

Uses the same default tasks and evaluation setup as moe_mixin_poc.py.

Usage:
    python scripts/eval_harness.py --checkpoint-dir checkpoints/step_10000
    python scripts/eval_harness.py --checkpoint-dir checkpoints/step_10000 --tasks mmlu gsm8k --limit 256
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.deepseek_moe import MoEConfig as DeepSeekMoEConfig, MoEMixin as DeepSeekMoEMixin
from src.temporal_moe import MoEConfig as TemporalMoEConfig, MoEMixin as TemporalMoEMixin
from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin
from src.vanilla_moe import MoEConfig as VanillaMoEConfig, MoEMixin as VanillaMoEMixin

import wandb
import lm_eval
from lm_eval.models.huggingface import HFLM

HARNESS_TASKS_DEFAULT = [
    "mmlu_abstract_algebra",
    "mmlu_college_mathematics",
    "mmlu_high_school_mathematics",
    "mmlu_elementary_mathematics",
]


def load_moe_model(checkpoint_dir: str, device: str = "cuda"):
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "moe_meta.json"
    meta = json.loads(meta_path.read_text())

    print(f"Loading base model: {meta['base_model']}")
    print(f"MoE type: {meta['moe_type']}, config: {meta['moe_config']}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        meta["base_model"], torch_dtype=torch.bfloat16
    )

    moe_type = meta["moe_type"]
    if moe_type == "temporal":
        moe_config = TemporalMoEConfig(**meta["moe_config"])
        TemporalMoEMixin.apply(base_model, moe_config)
    elif moe_type == "deepseek":
        moe_config = DeepSeekMoEConfig(**meta["moe_config"])
        DeepSeekMoEMixin.apply(base_model, moe_config)
    elif moe_type == "temporal-wrap":
        moe_config = TemporalWrapConfig(**meta["moe_config"])
        TemporalWrapMixin.apply(base_model, moe_config)
    else:
        moe_config = VanillaMoEConfig(**meta["moe_config"])
        VanillaMoEMixin.apply(base_model, moe_config)

    safetensors_files = list(checkpoint_dir.glob("*.safetensors"))
    bin_files = list(checkpoint_dir.glob("*.bin"))
    if safetensors_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in sorted(safetensors_files):
            state_dict.update(load_file(f, device="cpu"))
    elif bin_files:
        state_dict = {}
        for f in sorted(bin_files):
            state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

    base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device).eval()

    return base_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE checkpoint with lm-evaluation-harness")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to a saved checkpoint directory (contains moe_meta.json)")
    parser.add_argument("--tasks", nargs="+", default=HARNESS_TASKS_DEFAULT,
                        help="lm-eval task names")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (default: None = full eval)")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Number of few-shot examples (uses task default if not set)")
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="Batch size for evaluation (default: auto)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--wandb-project", default="moe-chunking-poc",
                        help="W&B project to log eval results to")
    args = parser.parse_args()

    model, tokenizer = load_moe_model(args.checkpoint_dir, args.device)

    meta = json.loads((Path(args.checkpoint_dir) / "moe_meta.json").read_text())
    wandb.init(
        project=args.wandb_project,
        job_type="eval",
        name=f"eval-{meta['moe_type']}-step{meta['step']}",
        config={
            "checkpoint_dir": args.checkpoint_dir,
            "tasks": args.tasks,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            **meta,
        },
    )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n--- {task_name} ---")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    wandb_metrics = {}
    for task_name, task_results in results["results"].items():
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                wandb_metrics[f"eval/harness_{task_name}/{metric}"] = value
    wandb.log(wandb_metrics)

    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result_file = out_path / "eval_results.json"
        result_file.write_text(json.dumps(results["results"], indent=2, default=str))
        print(f"\nResults saved to {result_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
