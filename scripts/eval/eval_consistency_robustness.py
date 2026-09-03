"""Consistency -> modularity -> robustness eval, for the narrative pivot:
routing consistency is not interesting on its own -- it matters because it
is the enabling condition for experts to specialize into reusable modules,
and modular specialization is what should make a model's behavior robust
under a changing environment (input perturbation / surface-form noise).
This script runs three independent, cheap parts for one (model, variant)
checkpoint and writes `evals/<model>/<date>/eval_consistency_robustness.json`.

Deliberately does NOT touch lm-eval-harness: part 3's downstream check is a
small, self-contained GSM8K exact-match eval (regex "#### <n>" extraction,
4 hand-written few-shot exemplars), so this script stays cheap enough to
run on top of the already-completed 1024-prompt harness sweep rather than
repeating it.

  1. Routing consistency under perturbation (eval_consistency.json entry
     "consistency"): for a shared pool of held-out prompts, build two
     surface-form-perturbed copies of each (character-level typo noise,
     and case/punctuation/whitespace noise) that preserve meaning. Do a
     teacher-forced forward pass (no generation) at --cache-layer for
     original/typo/format, take each prompt's per-expert top-1 usage
     histogram (pooled over valid tokens, robust to the token-misalignment
     a perturbation causes -- no per-token pairing needed), and report the
     Jensen-Shannon divergence between original and each perturbed
     histogram, averaged over prompts. Lower divergence = the router
     assigns experts based on meaning, not surface form.

  2. Modularity / expert specialization (eval_consistency.json entry
     "modularity"): sample two DIFFERENT-domain prompt pools (Nemotron's
     math and code splits) and forward-pass each separately, building a
     per-domain, per-expert usage histogram. For every expert e compute
     P(domain=math | e) = count_math(e) / (count_math(e) + count_code(e))
     and a specialization index in [0,1], |P(math|e) - 0.5| * 2 -- 1 means
     the expert is used almost exclusively by one domain. Also reports the
     Jensen-Shannon divergence between the two domains' overall expert
     distributions (higher = routing genuinely differs by domain, the
     complement of part 1's "invariant to noise" story: good modularity
     means invariant to surface noise but sensitive to real domain shift).

  3. Robustness to a changing environment (eval_consistency.json entry
     "robustness"): a small custom GSM8K exact-match eval (no lm-eval-
     harness), run once on the original test questions and once on a
     typo-perturbed copy of the same questions (same perturbation as part
     1), for --num-robustness-examples questions. Reports accuracy on
     each and the retention drop, per variant.

Usage:
    python scripts/eval/eval_consistency_robustness.py \\
        --model microsoft/Phi-tiny-MoE-instruct --variant base
    python scripts/eval/eval_consistency_robustness.py \\
        --model allenai/OLMoE-1B-7B-0125-Instruct --variant cache_reward \\
        --checkpoint-dir ddidacus/olmoe-cache-reward \\
        --cache-experts-per-token 8
"""
import argparse
import datetime
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # scripts/eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))      # scripts/train/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))         # repo root

import torch

from eval_benchmarks import build_variant_model  # noqa: E402
from eval_cache_conditioning import sample_eval_prompt_texts  # noqa: E402
from eval_complete import num_experts_of, resolve_layers, _right_pad_batch  # noqa: E402


# ---------------------------------------------------------------------------
# Perturbations: deterministic, dependency-free, meaning-preserving.
# ---------------------------------------------------------------------------

_TYPO_NEIGHBORS = "qwertyuiopasdfghjklzxcvbnm"


def typo_perturb(text, rate, rng):
    """Character-level noise: at each alphabetic character, with
    probability `rate`, either swap it with an adjacent-on-keyboard-ish
    random letter (kept simple: any other lowercase letter) or delete it.
    Preserves word boundaries and meaning for a human reader; token-level
    alignment with the original is NOT assumed by any downstream code
    here (see module docstring, part 1 uses pooled histograms)."""
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            if rng.random() < 0.5:
                out.append(rng.choice(_TYPO_NEIGHBORS))
            # else: drop the character (skip appending)
            continue
        out.append(ch)
    return "".join(out)


def format_perturb(text, rng):
    """Surface-form noise that keeps every character but reshuffles
    formatting: randomly upper/lower-cases whole words, and randomly
    doubles some whitespace runs / adds trailing spaces before newlines.
    Meaning-preserving, purely cosmetic."""
    words = text.split(" ")
    out_words = []
    for w in words:
        if w and rng.random() < 0.15:
            w = w.upper() if rng.random() < 0.5 else w.lower()
        out_words.append(w)
    text2 = " ".join(out_words)
    text2 = re.sub(r"\n", lambda m: "\n" + (" " if rng.random() < 0.3 else ""), text2)
    return text2


# ---------------------------------------------------------------------------
# Shared: per-expert top-1 usage histogram from one teacher-forced batch.
# ---------------------------------------------------------------------------

@torch.no_grad()
def expert_usage_histogram(model, tokenizer, texts, layer, num_experts,
                           prompt_len, device, batch_size):
    """Returns a list of (num_experts,) float arrays, one per input text,
    each a normalized top-1-expert usage histogram over that text's valid
    (non-pad) tokens at `layer`. Teacher-forced, no generation -- cheap."""
    pad_id = tokenizer.pad_token_id
    hists = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        ids_chunk = [tokenizer(t, truncation=True, max_length=prompt_len,
                               add_special_tokens=False)["input_ids"]
                    for t in chunk]
        ids_chunk = [ids if len(ids) > 0 else [pad_id] for ids in ids_chunk]
        input_ids, attn_mask = _right_pad_batch(ids_chunk, pad_id, device)
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_router_logits=True, use_cache=False)
        B, P = input_ids.shape
        router_logits = out.router_logits[layer].view(B, P, -1).float()
        top1 = router_logits.argmax(-1)  # (B, P)
        valid = attn_mask.bool()
        for b in range(B):
            ids_b = top1[b][valid[b]]
            counts = torch.bincount(ids_b, minlength=num_experts).float()
            total = counts.sum().clamp(min=1)
            hists.append((counts / total).cpu().numpy())
        print(f"[consistency_robustness] usage-histogram {min(i + batch_size, len(texts))}/{len(texts)}",
             flush=True)
    return hists


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence (base-2 bits) between two discrete
    distributions given as numpy arrays of the same length."""
    import numpy as np
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * np.log2(p / m)).sum()
    kl_qm = (q * np.log2(q / m)).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


# ---------------------------------------------------------------------------
# Part 1: routing consistency under surface-form perturbation.
# ---------------------------------------------------------------------------

def run_consistency(model, tokenizer, base_texts, layer, num_experts,
                    prompt_len, device, batch_size, typo_rate, seed):
    rng = random.Random(seed)
    typo_texts = [typo_perturb(t, typo_rate, rng) for t in base_texts]
    format_texts = [format_perturb(t, rng) for t in base_texts]

    hist_orig = expert_usage_histogram(model, tokenizer, base_texts, layer,
                                       num_experts, prompt_len, device, batch_size)
    hist_typo = expert_usage_histogram(model, tokenizer, typo_texts, layer,
                                       num_experts, prompt_len, device, batch_size)
    hist_fmt = expert_usage_histogram(model, tokenizer, format_texts, layer,
                                      num_experts, prompt_len, device, batch_size)

    js_typo = [js_divergence(a, b) for a, b in zip(hist_orig, hist_typo)]
    js_fmt = [js_divergence(a, b) for a, b in zip(hist_orig, hist_fmt)]

    def agg(vals):
        return {"mean": statistics.fmean(vals),
               "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}

    return {
        "n_prompts": len(base_texts),
        "typo_rate": typo_rate,
        "js_divergence_typo": agg(js_typo),
        "js_divergence_format": agg(js_fmt),
    }


# ---------------------------------------------------------------------------
# Part 2: modularity / cross-domain expert specialization.
# ---------------------------------------------------------------------------

def run_modularity(model, tokenizer, math_texts, code_texts, layer,
                   num_experts, prompt_len, device, batch_size):
    hist_math = expert_usage_histogram(model, tokenizer, math_texts, layer,
                                       num_experts, prompt_len, device, batch_size)
    hist_code = expert_usage_histogram(model, tokenizer, code_texts, layer,
                                       num_experts, prompt_len, device, batch_size)

    import numpy as np
    total_math = np.sum(hist_math, axis=0)
    total_code = np.sum(hist_code, axis=0)
    p_math_given_e = total_math / np.clip(total_math + total_code, 1e-12, None)
    specialization = np.abs(p_math_given_e - 0.5) * 2.0  # in [0, 1]

    cross_domain_js = js_divergence(total_math, total_code)
    specialized_frac = float((specialization > 0.6).mean())

    return {
        "n_prompts_per_domain": len(math_texts),
        "cross_domain_js_divergence": cross_domain_js,
        "specialization_mean": float(specialization.mean()),
        "specialization_std": float(specialization.std()),
        "specialized_expert_fraction_gt_0.6": specialized_frac,
        "per_expert_specialization": specialization.tolist(),
    }


# ---------------------------------------------------------------------------
# Part 3: robustness -- small custom GSM8K exact-match eval, no lm-eval.
# ---------------------------------------------------------------------------

_GSM8K_FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has 100/2 = $50. Her grandparents gave her 15*2 = $30. In total Betty has 50+15+30 = $95. Betty still needs 100-95 = $5. #### 5

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: James writes 3*2 = 6 pages each time. He does this twice a week, so 6*2 = 12 pages a week. In a year, he writes 12*52 = 624 pages. #### 624

"""

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_LAST_NUMBER_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def _extract_answer(text):
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _LAST_NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _normalize_num(s):
    if s is None:
        return None
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def load_gsm8k_questions(n, seed):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    questions, golds = [], []
    for i in idxs:
        row = ds[i]
        questions.append(row["question"])
        golds.append(_extract_answer(row["answer"]))
    return questions, golds


@torch.no_grad()
def run_gsm8k_batch(model, tokenizer, questions, device, batch_size, max_new_tokens):
    pad_id = tokenizer.pad_token_id
    preds = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        prompts = [_GSM8K_FEWSHOT + f"Question: {q}\nAnswer:" for q in chunk]
        ids_chunk = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        from eval_complete import _left_pad_batch
        input_ids, attn_mask = _left_pad_batch(ids_chunk, pad_id, device)
        gen = model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad_id,
        )
        completions = tokenizer.batch_decode(gen[:, input_ids.shape[1]:],
                                             skip_special_tokens=True)
        for c in completions:
            first_q = c.split("\nQuestion:")[0]
            preds.append(_extract_answer(first_q))
        print(f"[consistency_robustness] gsm8k {min(i + batch_size, len(questions))}/{len(questions)}",
             flush=True)
    return preds


def run_robustness(model, tokenizer, device, batch_size, num_examples, seed,
                   typo_rate, max_new_tokens):
    questions, golds = load_gsm8k_questions(num_examples, seed)
    rng = random.Random(seed)
    perturbed_questions = [typo_perturb(q, typo_rate, rng) for q in questions]

    preds_orig = run_gsm8k_batch(model, tokenizer, questions, device, batch_size, max_new_tokens)
    preds_pert = run_gsm8k_batch(model, tokenizer, perturbed_questions, device, batch_size, max_new_tokens)

    def accuracy(preds):
        correct = sum(1 for p, g in zip(preds, golds)
                     if _normalize_num(p) is not None and _normalize_num(p) == _normalize_num(g))
        return correct / len(golds)

    acc_orig = accuracy(preds_orig)
    acc_pert = accuracy(preds_pert)
    return {
        "n_examples": len(questions),
        "typo_rate": typo_rate,
        "accuracy_original": acc_orig,
        "accuracy_perturbed": acc_pert,
        "accuracy_drop": acc_orig - acc_pert,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--cache-layer", type=int, default=-1)
    ap.add_argument("--dataset", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    ap.add_argument("--num-prompts", type=int, default=256,
                    help="prompts for part 1 (consistency) and per-domain for part 2 (modularity)")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--typo-rate", type=float, default=0.05)
    ap.add_argument("--num-robustness-examples", type=int, default=150)
    ap.add_argument("--robustness-max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--out-dir-root", default="evals")
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(args.model)
    num_experts = num_experts_of(config)
    cache_layer, _ = resolve_layers(config.num_hidden_layers, args.cache_layer)

    out_dir = Path(args.out_dir_root) / args.model.split("/")[-1] / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_consistency_robustness_{args.variant}.json"
    print(f"[consistency_robustness] model={args.model} variant={args.variant} "
         f"cache_layer={cache_layer} num_experts={num_experts} out={out_path}", flush=True)

    model = build_variant_model(args.variant, args.model, device, args.checkpoint_dir)
    model.eval()

    base_texts = sample_eval_prompt_texts(tok, args.dataset, "math,code",
                                          args.num_prompts, args.seed)
    print(f"[consistency_robustness] [1/3] consistency under perturbation "
         f"({len(base_texts)} prompts)", flush=True)
    consistency = run_consistency(model, tok, base_texts, cache_layer, num_experts,
                                  args.prompt_len, device, args.batch_size,
                                  args.typo_rate, args.seed)
    print(f"[consistency_robustness] [1/3] js_typo={consistency['js_divergence_typo']['mean']:.4f} "
         f"js_format={consistency['js_divergence_format']['mean']:.4f}", flush=True)

    print(f"[consistency_robustness] [2/3] modularity / domain specialization "
         f"({args.num_prompts} prompts/domain)", flush=True)
    math_texts = sample_eval_prompt_texts(tok, args.dataset, "math", args.num_prompts, args.seed)
    code_texts = sample_eval_prompt_texts(tok, args.dataset, "code", args.num_prompts, args.seed)
    modularity = run_modularity(model, tok, math_texts, code_texts, cache_layer,
                                num_experts, args.prompt_len, device, args.batch_size)
    print(f"[consistency_robustness] [2/3] cross_domain_js={modularity['cross_domain_js_divergence']:.4f} "
         f"specialized_frac={modularity['specialized_expert_fraction_gt_0.6']:.4f}", flush=True)

    print(f"[consistency_robustness] [3/3] robustness (GSM8K, "
         f"{args.num_robustness_examples} examples)", flush=True)
    robustness = run_robustness(model, tok, device, args.batch_size,
                                args.num_robustness_examples, args.seed,
                                args.typo_rate, args.robustness_max_new_tokens)
    print(f"[consistency_robustness] [3/3] acc_orig={robustness['accuracy_original']:.4f} "
         f"acc_pert={robustness['accuracy_perturbed']:.4f} "
         f"drop={robustness['accuracy_drop']:.4f}", flush=True)

    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "variant": args.variant, "cache_layer": cache_layer,
            "num_experts": num_experts,
            "consistency": consistency,
            "modularity": modularity,
            "robustness": robustness,
        }, f, indent=2)
    print(f"[consistency_robustness] done. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
