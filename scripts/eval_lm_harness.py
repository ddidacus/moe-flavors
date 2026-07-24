"""Downstream task eval (EleutherAI lm-evaluation-harness) for the MoE
cache-reward experiments: MMLU + MMMLU (knowledge/reasoning, English +
14-language pooled sample), GSM8K + MATH (math), HumanEval (code) -- so
cache-hit/temporal-consolidation gains from eval_soft_cache.py can be
checked against actual task performance, not just routing metrics.

MMMLU has no native "all languages pooled" task in lm-eval -- only 798
separate per-(language, subject) leaf tasks (14 languages x 57 subjects,
see MMMLU_LANGUAGES/MMLU_SUBJECT_SIZES below). To get a literal 200-question
sample from the FULL multilingual pool (not 200 per language or per
subject), build_mmmlu_samples() draws 200 uniformly at random across the
(language, subject, doc_index) space and hands lm_eval the exact indices via
its `samples=` dict -- MMLU_SUBJECT_SIZES are each subject's document count,
reused as-is for every language since MMMLU is a direct per-row translation
of MMLU (identical example counts), taken from our own completed English
MMLU run rather than re-downloading all 798 configs just to measure sizes.

Variants (each loaded as its own model instance -- see build_variant_model):
  base          plain microsoft/Phi-tiny-MoE-instruct, no adapter
  cache_sft     GRPO + soft cache-hit reward + SFT NLL loss, lr=1e-4
                (checkpoints/grpo_..._softall_lr1e-4_..., non-temporal)
  temporal_moe  same, but with the boundary/hold-switch mixin at the cache
                layer (checkpoints/grpo_..._tmoeN8_lr1e-4_...)
  sft_baseline  plain LoRA SFT (scripts/finetune_moe_sft.py), no RL, no
                cache reward -- the standard-finetuning reference point
  controller_baseline  Option-Critic MoE controller (Shen & Henderson 2026),
                scripts/finetune_moe_controller.py

Each variant runs in its own process (see run_eval_lm_harness.sh -- one
sbatch job per variant, GPU-pinned, so they can run concurrently), writing
results_<variant>.json. --variant merge then loads every results_*.json in
--out-dir and writes summary.json + a printed table -- no model/GPU needed,
mirrors eval_soft_cache.py's base/tuned/merge split.

Generation is fixed at temperature=1.0, top_p=0.95, max 2048 new tokens
(SAMPLING_KWARGS below) -- applied uniformly to every generate_until task
(GSM8K, HumanEval, MATH), overriding whatever each task's own yaml specifies
(some default to greedy do_sample=false). MMLU and MMMLU are loglikelihood-
scored multiple-choice tasks -- no sampling involved, so temperature/top_p/
seed have no effect on them at all.

--num-seeds (default 4, SEEDS below) reruns ONLY the generate_until tasks
under N different fixed seeds and reports mean/std across them, since that's
the only axis these tasks actually vary along with sampling enabled; MMLU/
MMMLU are computed once (rerunning a deterministic score 4x would just waste
compute for identical numbers). Each results_<variant>.json holds both the
per-seed raw scores and the aggregate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# HumanEval's own utils.py runs a code_eval self-test at import time (before
# lm_eval's own confirm_run_unsafe_code gate is ever consulted) -- the
# underlying `evaluate` HF metric refuses to execute generated code at all
# without this env var, crashing the import outright. Must be set before
# anything imports lm_eval.tasks.humaneval (i.e. before simple_evaluate's
# task_manager.load() runs), and this is the sanctioned, authorized-eval-only
# use case the checkbox exists for.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# peft 0.19 x transformers 5.8 bug -- see finetune_moe_grpo.py for the same
# workaround, kept identical across every script in this pipeline.
import peft.utils.transformers_weight_conversion as _pwc

_orig_convert = _pwc.convert_peft_adapter_state_dict_for_transformers


def _convert_only_if_legacy(model, peft_config, adapter_state_dict, adapter_name):
    legacy = any(".w1." in k or ".w2." in k or ".w3." in k
                 or "block_sparse_moe" in k for k in adapter_state_dict)
    if not legacy:
        return adapter_state_dict
    return _orig_convert(model=model, peft_config=peft_config,
                         adapter_state_dict=adapter_state_dict,
                         adapter_name=adapter_name)


_pwc.convert_peft_adapter_state_dict_for_transformers = _convert_only_if_legacy

# lm_eval's few-shot chat-message builder asserts the answer is a plain str,
# with a `list[str]` special case for multi-target tasks (its own code
# comments this as a known hack). Some hendrycks_math training-split rows
# (used as few-shot exemplars once --num-fewshot > 0, see MATH_NUM_FEWSHOT)
# have a multi-value answer stored as list[int] (e.g. [-77, 77] for a
# two-solution equation), which fails that assert since the hack only
# coerces list[str]. Only exercised once few-shot is enabled for this task,
# hence never hit before. Coerce every element to str before delegating to
# the original implementation.
import lm_eval.api.task as _lm_task

_orig_build_qa_turn = _lm_task.ConfigurableTask.build_qa_turn


def _build_qa_turn_str_safe(self, *, q=None, c=None, a=None, gen_prefix=None,
                            tgt_delim=" ", few_delim="\n\n"):
    if isinstance(a, list):
        a = [str(x) for x in a]
    elif a is not None and not isinstance(a, (str, int)):
        a = str(a)
    return _orig_build_qa_turn(self, q=q, c=c, a=a, gen_prefix=gen_prefix,
                               tgt_delim=tgt_delim, few_delim=few_delim)


_lm_task.ConfigurableTask.build_qa_turn = _build_qa_turn_str_safe


VARIANT_CHECKPOINTS = {
    "cache_sft": "checkpoints/grpo_phi-tiny-moe-instruct_cache_mathcode_"
                "sft0.5_b0.08_c4_softall_lr1e-4_dbg150_rl2.0",
    "temporal_moe": "checkpoints/grpo_phi-tiny-moe-instruct_cache_mathcode_"
                   "sft0.5_b0.08_c4_topk2_tmoeN8_lr1e-4_dbg150_rl2.0",
    "sft_baseline": "checkpoints/sft_phi-tiny-moe-instruct_mathcode_lr1e-4",
    "sft_baseline_seq1024": "checkpoints/sft_phi-tiny-moe-instruct_mathcode_"
                            "lr1e-4_seq1024-1024",
    "controller_baseline": "checkpoints/controller_phi-tiny-moe-instruct_"
                           "mathcode_c4_eta0.02",
}
# mmlu/mmmlu_<lang>: multiple_choice, loglikelihood-scored -- deterministic,
# unaffected by temperature/top_p/seed. gsm8k/humaneval/hendrycks_math:
# generate_until -- affected by SAMPLING_KWARGS and reseeded per SEEDS.
STOCHASTIC_TASKS = ["gsm8k", "humaneval", "hendrycks_math"]
MMMLU_TOTAL_SAMPLES = 200  # pooled across ALL languages, not per-language

# hendrycks_math ships with 0 few-shot examples by default -- without a
# worked-example demonstrating the "$...\boxed{...}...$" answer format its
# extractor looks for, the model has no cue to produce it (see run_variant).
MATH_NUM_FEWSHOT = 4

# The 14 languages openai/MMMLU ships (each mmmlu_<lang> is a group of 57
# per-subject leaf tasks, identical taxonomy to MMLU).
MMMLU_LANGUAGES = ["ar_xy", "bn_bd", "de_de", "es_la", "fr_fr", "hi_in",
                  "id_id", "it_it", "ja_jp", "ko_kr", "pt_br", "sw_ke",
                  "yo_ng", "zh_cn"]

# Per-subject document counts, from our own completed English MMLU run
# (results_base.json's mmlu_<subject> "sample_len") -- MMMLU is a row-for-row
# translation of MMLU, so every language has the exact same per-subject
# counts. Sums to 14042, the known MMLU test-set size.
MMLU_SUBJECT_SIZES = {
    "abstract_algebra": 100, "anatomy": 135, "astronomy": 152,
    "college_biology": 144, "college_chemistry": 100,
    "college_computer_science": 100, "college_mathematics": 100,
    "college_physics": 102, "computer_security": 100,
    "conceptual_physics": 235, "electrical_engineering": 145,
    "elementary_mathematics": 378, "high_school_biology": 310,
    "high_school_chemistry": 203, "high_school_computer_science": 100,
    "high_school_mathematics": 270, "high_school_physics": 151,
    "high_school_statistics": 216, "machine_learning": 112,
    "business_ethics": 100, "clinical_knowledge": 265,
    "college_medicine": 173, "global_facts": 100, "human_aging": 223,
    "management": 103, "marketing": 234, "medical_genetics": 100,
    "miscellaneous": 783, "nutrition": 306, "professional_accounting": 282,
    "professional_medicine": 272, "virology": 166, "econometrics": 114,
    "high_school_geography": 198, "high_school_government_and_politics": 193,
    "high_school_macroeconomics": 390, "high_school_microeconomics": 238,
    "high_school_psychology": 545, "human_sexuality": 131,
    "professional_psychology": 612, "public_relations": 110,
    "security_studies": 245, "sociology": 201, "us_foreign_policy": 100,
    "formal_logic": 126, "high_school_european_history": 165,
    "high_school_us_history": 204, "high_school_world_history": 237,
    "international_law": 121, "jurisprudence": 108, "logical_fallacies": 163,
    "moral_disputes": 346, "moral_scenarios": 895, "philosophy": 311,
    "prehistory": 324, "professional_law": 1534, "world_religions": 171,
}


def build_mmmlu_samples(seed, n_total=MMMLU_TOTAL_SAMPLES):
    """Uniformly-random n_total (language, subject, doc_index) triples
    across the full 14-language x 57-subject pool, grouped into lm_eval's
    `samples={task_name: [indices]}` format -- the only way to get a
    literal N-total pooled sample rather than N-per-subtask."""
    import random

    pool = [(lang, subject, i)
            for lang in MMMLU_LANGUAGES
            for subject, size in MMLU_SUBJECT_SIZES.items()
            for i in range(size)]
    picks = random.Random(seed).sample(pool, n_total)

    samples = {}
    for lang, subject, idx in picks:
        task = f"mmmlu_{lang}_{subject}"
        samples.setdefault(task, []).append(idx)
    return samples


# Logical table/summary tasks -- "mmmlu" here is our synthetic pooled
# aggregate (see build_mmmlu_samples), not a real lm_eval task name.
TASKS = ["mmlu", "mmmlu"] + STOCHASTIC_TASKS

SEEDS = [42, 43, 44, 45]
SAMPLING_KWARGS = {"do_sample": True, "temperature": 1.0, "top_p": 0.95,
                  "max_gen_toks": 2048}


def load_adapter(peft_model, ckpt_dir):
    """Manual adapter load (peft 0.19 can't load ParamWrapper/target_parameters
    adapters) -- identical to eval_soft_cache.py's load_adapter."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    model_keys = set(peft_model.state_dict().keys())
    remapped = {}
    for k, v in sd.items():
        nk = k.replace(".lora_A.weight", ".lora_A.default.weight") \
              .replace(".lora_B.weight", ".lora_B.default.weight")
        if nk not in model_keys:
            head, _, tail = nk.rpartition(".")
            cand = f"{head}.modules_to_save.default.{tail}"
            if cand in model_keys:
                nk = cand
        remapped[nk] = v
    res = peft_model.load_state_dict(remapped, strict=False)
    assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys[:5]}"
    missing = [k for k in res.missing_keys if "lora" in k]
    assert not missing, f"missing lora keys: {missing[:5]}"
    print(f"[eval] loaded {len(remapped)} adapter tensors from {ckpt_dir}")


def build_variant_model(variant, base_model_name, device, checkpoint_dir=None):
    """Returns a ready-to-eval nn.Module for one variant -- no dense-router
    patching here (unlike eval_soft_cache.py): this is a downstream task
    eval, so every variant runs with its actual deployment-time routing
    (native sparse top-k, or the temporal mixin's real hold/switch forward),
    not an analysis-only monkeypatch.

    checkpoint_dir: overrides VARIANT_CHECKPOINTS[variant] when given --
    for evaluating a checkpoint trained elsewhere (e.g. via cluv on a
    non-mila cluster, where the SAVE_DIR naming differs from the mila
    run_finetune_moe_*.sh convention baked into VARIANT_CHECKPOINTS)."""
    from transformers import AutoModelForCausalLM

    if variant == "base":
        return AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)

    from peft import LoraConfig, get_peft_model
    ckpt = checkpoint_dir or VARIANT_CHECKPOINTS[variant]
    from transformers.trainer_utils import get_last_checkpoint
    ckpt_dir = get_last_checkpoint(ckpt) or ckpt
    assert Path(ckpt_dir, "adapter_config.json").exists(), \
        f"no adapter checkpoint found under {ckpt}"

    m = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)

    if variant == "temporal_moe":
        # architectural (not something LoRA adds) -> wrap before the adapter
        # is loaded on top of it, exactly like training/eval_soft_cache.py.
        from src.temporal_moe_wrapper import TemporalWrapConfig, TemporalWrapMixin
        TemporalWrapMixin.apply(m, TemporalWrapConfig(ste=True))

    peft_cfg = LoraConfig.from_pretrained(ckpt_dir)
    peft_model = get_peft_model(m, peft_cfg)
    load_adapter(peft_model, ckpt_dir)
    return peft_model


def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def _aggregate_stochastic(per_seed, tasks):
    """Mean/std across SEEDS for every numeric metric key, for each task in
    `tasks` -- shared by run_variant and run_math_only."""
    agg = {}
    seeds = list(per_seed.keys())
    for task in tasks:
        keys = set()
        for seed in seeds:
            keys.update(k for k in per_seed[seed].get(task, {})
                       if isinstance(per_seed[seed][task][k], (int, float)))
        agg[task] = {}
        for key in keys:
            vals = [per_seed[seed][task][key] for seed in seeds
                    if task in per_seed[seed] and key in per_seed[seed][task]]
            if len(vals) == len(seeds):
                mean, std = _mean_std(vals)
                agg[task][key] = {"mean": mean, "std": std, "seeds": vals}
    return agg


def run_variant(variant, base_model_name, batch_size, limit, out_dir, checkpoint_dir=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_variant_model(variant, base_model_name, device, checkpoint_dir)
    model.eval()

    # Wrap the already-instantiated (and, for cache_sft/temporal_moe,
    # already adapter-loaded) nn.Module directly -- HFLM's own `peft=`
    # kwarg would reload the adapter itself via peft's set_peft_model_
    # state_dict, which is exactly the path that's broken for our fused
    # target_parameters (ParamWrapper) adapters (see load_adapter above).
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size,
             device=device)

    # Deterministic (loglikelihood-scored) tasks: one pass, no sampling
    # involved so no seed dependence -- see module docstring.
    mmlu_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=None,  # per-task default (5-shot)
        limit=limit,
        log_samples=False,
        random_seed=SEEDS[0], numpy_random_seed=SEEDS[0],
        torch_random_seed=SEEDS[0], fewshot_random_seed=SEEDS[0],
    )["results"]

    # MMMLU: 200 questions pooled across all 14 languages (not per-language,
    # not per-subject) -- see build_mmmlu_samples/module docstring. Only the
    # specific leaf (language, subject) tasks that got at least one sampled
    # doc are actually run.
    mmmlu_samples = build_mmmlu_samples(SEEDS[0])
    mmmlu_leaf_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(mmmlu_samples.keys()),
        num_fewshot=None,
        samples=mmmlu_samples,
        log_samples=False,
        random_seed=SEEDS[0], numpy_random_seed=SEEDS[0],
        torch_random_seed=SEEDS[0], fewshot_random_seed=SEEDS[0],
    )["results"]
    # Weighted pool: each leaf task's acc weighted by how many of the 200
    # sampled docs landed in it, so the combined score reflects the true
    # global sample rather than an unweighted mean-of-subtask-means.
    n_sampled = sum(len(v) for v in mmmlu_samples.values())
    mmmlu_acc = sum(mmmlu_leaf_results[t]["acc,none"] * len(mmmlu_samples[t])
                    for t in mmmlu_samples) / n_sampled
    mmmlu_pooled = {"acc,none": mmmlu_acc, "n_sampled": n_sampled,
                    "n_leaf_tasks_touched": len(mmmlu_samples)}

    det_results = {"mmlu": mmlu_results["mmlu"], "mmmlu": mmmlu_pooled,
                   "mmmlu_leaf_results": mmmlu_leaf_results}
    for task, metrics in det_results.items():
        print(f"  [{variant}] {task}: {metrics}")

    # Stochastic (generate_until) tasks: SAMPLING_KWARGS fixed across every
    # seed, so the only thing that varies is the sampled completions.
    #
    # hendrycks_math gets its own call with num_fewshot=MATH_NUM_FEWSHOT: its
    # yaml has zero few-shot examples by default, and its answer extractor
    # only looks for a "$...\boxed{...}...$"-wrapped final answer -- with no
    # in-context demonstration of that format, the model has no reason to
    # produce it (this is exactly why MATH scored a flat 0.0 for every
    # variant, including the untouched base model, before this fix). GSM8K/
    # HumanEval keep their own task-default few-shot counts (GSM8K's 5-shot
    # exemplars already demonstrate the "####" convention its own extractor
    # expects, hence why it scored normally without this treatment).
    per_seed = {}
    for seed in SEEDS:
        seed_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=["gsm8k", "humaneval"],
            num_fewshot=None,
            limit=limit,
            gen_kwargs=SAMPLING_KWARGS,
            confirm_run_unsafe_code=True,  # humaneval executes generated code
            log_samples=False,
            random_seed=seed, numpy_random_seed=seed,
            torch_random_seed=seed, fewshot_random_seed=seed,
        )["results"]
        math_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=["hendrycks_math"],
            num_fewshot=MATH_NUM_FEWSHOT,
            limit=limit,
            gen_kwargs=SAMPLING_KWARGS,
            log_samples=False,
            random_seed=seed, numpy_random_seed=seed,
            torch_random_seed=seed, fewshot_random_seed=seed,
        )["results"]
        seed_results.update(math_results)
        per_seed[seed] = seed_results
        print(f"  [{variant}] seed={seed}:")
        for task, metrics in seed_results.items():
            print(f"    {task}: {metrics}")

    stochastic_agg = _aggregate_stochastic(per_seed, STOCHASTIC_TASKS)
    results = {**det_results, **stochastic_agg}

    out_path = Path(out_dir) / f"results_{variant}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant,
            "seeds": SEEDS,
            "sampling_kwargs": SAMPLING_KWARGS,
            "deterministic_results": det_results,
            "per_seed_stochastic_results": per_seed,
            "results": results,  # deterministic tasks as-is, stochastic tasks as {key: {mean,std,seeds}}
        }, f, indent=2)
    print(f"[eval] wrote {out_path}")


def run_math_only(variant, base_model_name, batch_size, limit, out_dir, checkpoint_dir=None):
    """Rerun ONLY hendrycks_math (with the few-shot + build_qa_turn fixes)
    and patch the result into an existing results_<variant>.json, leaving
    every other task's numbers untouched. For variants evaluated before
    both MATH fixes landed (flat 0.0 for every variant, including the
    untouched base model) -- avoids re-running the other 4, much more
    expensive tasks just to fix one."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    out_path = Path(out_dir) / f"results_{variant}.json"
    assert out_path.exists(), f"no existing results to patch: {out_path}"
    existing = json.load(open(out_path))

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_variant_model(variant, base_model_name, device, checkpoint_dir)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device)

    per_seed = existing.get("per_seed_stochastic_results", {})
    for seed in SEEDS:
        math_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=["hendrycks_math"],
            num_fewshot=MATH_NUM_FEWSHOT,
            limit=limit,
            gen_kwargs=SAMPLING_KWARGS,
            log_samples=False,
            random_seed=seed, numpy_random_seed=seed,
            torch_random_seed=seed, fewshot_random_seed=seed,
        )["results"]
        seed_key = str(seed) if str(seed) in per_seed else seed
        per_seed.setdefault(seed_key, {})["hendrycks_math"] = math_results["hendrycks_math"]
        print(f"  [{variant}] seed={seed}: hendrycks_math: {math_results['hendrycks_math']}")

    # JSON round-trips int seed keys as strings -- normalize before aggregating.
    per_seed_norm = {int(k): v for k, v in per_seed.items()}
    agg = _aggregate_stochastic(per_seed_norm, ["hendrycks_math"])
    existing["results"]["hendrycks_math"] = agg["hendrycks_math"]
    existing["per_seed_stochastic_results"] = per_seed
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[eval] patched hendrycks_math into {out_path}")


def merge(out_dir):
    out_dir = Path(out_dir)
    all_variants = ["base", "cache_sft", "temporal_moe", "sft_baseline",
                   "sft_baseline_seq1024", "controller_baseline"]
    summary = {}
    for variant in all_variants:
        p = out_dir / f"results_{variant}.json"
        if not p.exists():
            print(f"[merge] skipping {variant}: {p} not found")
            continue
        summary[variant] = json.load(open(p))["results"]

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # primary metric per task. Deterministic tasks (mmlu, and our synthetic
    # pooled "mmmlu") store plain {key: value}; stochastic tasks (gsm8k/
    # humaneval/hendrycks_math) store {key: {"mean":, "std":, "seeds":}}
    # aggregated over SEEDS -- see run_variant. Table shows "mean±std" for
    # the latter.
    primary = {"mmlu": "acc,none", "mmmlu": "acc,none",
              "gsm8k": "exact_match,strict-match",
              "humaneval": "pass@1,create_test",
              "hendrycks_math": "exact_match,none"}
    header = ["variant"] + TASKS
    rows = [header]
    for variant, tasks in summary.items():
        row = [variant]
        for t in TASKS:
            key = primary[t]
            entry = tasks.get(t, {}).get(key)
            if entry is None:
                row.append("-")
            elif isinstance(entry, dict):  # stochastic: {"mean","std","seeds"}
                row.append(f"{entry['mean']:.4f}±{entry['std']:.4f}")
            else:
                row.append(f"{entry:.4f}")
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    print("\n[summary]")
    for row in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print(f"\n[merge] wrote {out_dir / 'summary.json'}")


def main():
    global SEEDS, SAMPLING_KWARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant",
                    choices=["base", "cache_sft", "temporal_moe",
                             "sft_baseline", "sft_baseline_seq1024",
                             "controller_baseline", "merge"],
                    required=True)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=float, default=200,
                    help="cap examples per task/subtask (200 by default, "
                         "matching the paper's own eval sample size); None "
                         "= full set. Note: for grouped benchmarks (mmlu, "
                         "mmmlu_fr_fr, hendrycks_math) this applies per "
                         "constituent subtask, not once for the whole group")
    ap.add_argument("--out-dir", default="eval_lm_harness")
    ap.add_argument("--num-seeds", type=int, default=len(SEEDS),
                    help=f"how many of {SEEDS} to actually run for the "
                         f"stochastic tasks (fewer = faster, less variance "
                         f"info); default uses all {len(SEEDS)}")
    ap.add_argument("--max-gen-toks", type=int,
                    default=SAMPLING_KWARGS["max_gen_toks"],
                    help="max new tokens for generate_until tasks (gsm8k/"
                         "humaneval/hendrycks_math)")
    ap.add_argument("--math-only", action="store_true",
                    help="rerun only hendrycks_math and patch it into an "
                         "existing results_<variant>.json -- for variants "
                         "evaluated before the few-shot/build_qa_turn MATH "
                         "fixes landed, without re-running everything else")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="override VARIANT_CHECKPOINTS[variant] with an "
                         "explicit adapter checkpoint dir -- for evaluating "
                         "a checkpoint trained elsewhere (e.g. via cluv on "
                         "a non-mila cluster) whose SAVE_DIR doesn't match "
                         "the mila run_finetune_moe_*.sh naming convention. "
                         "Ignored for --variant base.")
    args = ap.parse_args()

    if args.variant == "merge":
        merge(args.out_dir)
        return

    SEEDS = SEEDS[:args.num_seeds]
    SAMPLING_KWARGS = {**SAMPLING_KWARGS, "max_gen_toks": args.max_gen_toks}

    if args.math_only:
        run_math_only(args.variant, args.model, args.batch_size, args.limit,
                      args.out_dir, args.checkpoint_dir)
        return
    run_variant(args.variant, args.model, args.batch_size, args.limit,
               args.out_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
