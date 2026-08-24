"""Sanity check: eval the untouched microsoft/Phi-tiny-MoE-instruct base model
on HumanEval ONLY via lm-eval-harness, using the task's own default
generation_kwargs (greedy, do_sample=false -- standard pass@1 methodology)
instead of eval/eval_benchmarks.py's SAMPLING_KWARGS override (do_sample=True,
temperature=1.0, top_p=0.95), which get merged into (not replace) the task's
generation_kwargs dict via lm_eval's TaskConfig.set_config(update=True) --
flipping do_sample False->True. That's the suspected reason our HumanEval
numbers (~0.00-0.04 pass@1) look far below any declared score. This script
isolates the variable: full 164-problem test set, no --limit, explicit
max_length=4096 (microsoft/Phi-tiny-MoE-instruct's max_position_embeddings),
no gen_kwargs override -> should land close to the model card's declared
HumanEval score.

Usage: python scripts/sanity_check_humaneval.py
"""
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "microsoft/Phi-tiny-MoE-instruct"

def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size="auto",
             device=device, max_length=4096)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["humaneval"],
        num_fewshot=0,
        limit=None,
        confirm_run_unsafe_code=True,
        log_samples=False,
        random_seed=42, numpy_random_seed=42,
        torch_random_seed=42, fewshot_random_seed=42,
    )["results"]

    print("[sanity_check_humaneval] results:", results["humaneval"])


if __name__ == "__main__":
    main()
