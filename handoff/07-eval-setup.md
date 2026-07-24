# Eval setup (2026-07-23)

Two complementary eval tracks, run on the same set of trained checkpoints:

- **Quantitative** (`scripts/eval_lm_harness.py`): downstream task
  performance -- does the cache-hit/temporal-consolidation/controller
  training cost any actual capability? (MMLU, MMMLU, GSM8K, HumanEval,
  MATH)
- **Qualitative** (`scripts/eval_soft_cache.py`): routing-distribution
  analysis at the cache layer -- is the training objective actually doing
  what it's supposed to (concentrating routing mass on a small working
  set, holding experts across contiguous tokens), independent of whether
  that shows up in downstream accuracy?

Both matter: a checkpoint could look fine on quantitative benchmarks while
having learned nothing about cache consolidation (reward hacking via some
other channel), or could show strong routing consolidation that turns out
not to preserve language-modeling quality. Neither track substitutes for
the other.

## Part 1 -- Quantitative (`scripts/eval_lm_harness.py`)

Run via `scripts/run_eval_lm_harness.sh`. Results land in `evals/<date>/`,
one `results_<variant>.json` per variant + a `--variant merge` step that
writes `summary.json` and prints the comparison table.

## Models evaluated

| variant | checkpoint | notes |
| --- | --- | --- |
| `base` | `microsoft/Phi-tiny-MoE-instruct`, no adapter | untouched reference |
| `cache_sft` | `checkpoints/grpo_..._softall_lr1e-4_dbg150_rl2.0` | GRPO + soft cache-hit reward + SFT NLL, lr=1e-4, non-temporal |
| `temporal_moe` | `checkpoints/grpo_..._topk2_tmoeN8_lr1e-4_dbg150_rl2.0` | same + boundary/hold-switch mixin; wrapped with `TemporalWrapMixin` before adapter load (architectural, not something LoRA adds) |
| `sft_baseline_seq1024` | `checkpoints/sft_..._lr1e-4_seq1024-1024` | plain LoRA SFT, trained at prompt+completion=1024+1024 (the original 512+512 `sft_baseline` checkpoint is excluded from this eval round) |
| `controller_baseline` | `checkpoints/controller_..._c4_eta0.02` | Option-Critic MoE controller reimplementation; standard peft adapter at inference (the controller heads never alter real generation, so no special wrapping needed for eval) |

Each variant loads as its own model instance (`build_variant_model`) and
runs in its own process, GPU-pinned via `CUDA_VISIBLE_DEVICES`.
`run_eval_lm_harness.sh` takes 1-2 variant names as positional args and runs
them in parallel within one sbatch job (2 GPUs/job); current batch is 3 jobs
(2+2+1) covering all 5 variants above.

## Datasets / tasks

| task | benchmark | shots | scoring |
| --- | --- | --- | --- |
| `mmlu` | MMLU (English), 57 subjects | 5-shot | loglikelihood multiple-choice, deterministic |
| `mmmlu` (synthetic, see below) | MMMLU, 14 languages x 57 subjects | 5-shot | loglikelihood multiple-choice, deterministic |
| `gsm8k` | GSM8K | 5-shot | `generate_until`, stochastic |
| `humaneval` | HumanEval | 0-shot | `generate_until` + code execution (`pass@1`), stochastic |
| `hendrycks_math` | MATH (Hendrycks), 7 subject groups | 4-shot (task default) | `generate_until`, stochastic |

**MMMLU pooling:** lm-eval has no native "all languages pooled" MMMLU task,
only 798 separate per-(language, subject) leaf tasks. `build_mmmlu_samples()`
draws exactly 200 (language, subject, doc_index) triples uniformly at random
from the full pool (weighted by each subject's real size, reusing the exact
per-subject document counts already measured from the completed English
MMLU run -- sums to 14,042, the known MMLU test-set size) and hands
lm_eval the specific indices via its `samples=` dict. Only leaf tasks that
receive at least one sampled doc are actually run (typically ~150-160 of
798). The reported `mmmlu` score is the resulting pooled accuracy, weighted
by how many of the 200 samples landed in each touched leaf task -- not an
unweighted mean of subtask accuracies.

## Prompt / generation length

- `generate_until` tasks (GSM8K, HumanEval, MATH): **max 2048 new tokens**
  (`max_gen_toks` in `SAMPLING_KWARGS`), overriding each task's own yaml
  default.
- `mmlu`/`mmmlu`: standard few-shot loglikelihood scoring, no free
  generation -- prompt length is whatever the model's context window
  supports, not separately capped.

## Sampling parameters

- **Temperature = 1.0, top_p = 0.95, do_sample = True** -- fixed uniformly
  across every `generate_until` task via `gen_kwargs=SAMPLING_KWARGS`,
  overriding whatever each task's own yaml specifies (several default to
  greedy `do_sample=false`).
- `mmlu`/`mmmlu` are loglikelihood-scored -- temperature/top_p/seed have no
  effect on them at all (no sampling involved).
- **4 fixed seeds** (42, 43, 44, 45): applied only to the 3 stochastic
  tasks (GSM8K, HumanEval, MATH) via `random_seed`/`numpy_random_seed`/
  `torch_random_seed`/`fewshot_random_seed` -- `torch_random_seed` is
  confirmed to actually seed the global torch RNG that HF `generate()`
  samples from. Each `results_<variant>.json` stores the raw per-seed
  scores plus the mean/std aggregate. `mmlu`/`mmmlu` run once each (seed
  has no effect on a deterministic score, so reseeding would just waste
  compute reproducing identical numbers).

## Sample size

- **200 examples per task**, `--limit 200` -- for grouped benchmarks (mmlu,
  hendrycks_math) this applies **per constituent subtask**, not once for
  the whole group (a lm-eval mechanic, not a deliberate choice for those
  two). MMMLU is the one true "200 total, pooled" case (see above).
- Matches the paper's own eval budget (Shen & Henderson 2026): "200
  randomly selected questions from MATH, MMLU, and MMMLU respectively" --
  same source as the temperature/top_p/2048-token choices above, extended
  here to also cover GSM8K/HumanEval for a uniform methodology across all
  five tasks.

## Metrics reported (primary, per task)

| task | metric key |
| --- | --- |
| `mmlu` | `acc,none` |
| `mmmlu` | pooled `acc,none` (weighted, synthetic -- see above) |
| `gsm8k` | `exact_match,strict-match` |
| `humaneval` | `pass@1,create_test` |
| `hendrycks_math` | `exact_match,none` |

## Part 2 -- Qualitative (`scripts/eval_soft_cache.py`)

Run via `scripts/run_eval_soft_cache.sh` (one sbatch job, 2 GPUs: base +
tuned variant processes run concurrently, then a CPU-only merge). On-policy
only -- each variant is scored on its own generations (T=1.0, matching
training rollouts), no teacher-forced mode. Outputs land in `--out-dir`
(one directory per checkpoint compared), with per-layer subfolders.

### Models evaluated

Base vs. one tuned checkpoint per invocation (not all-variants-at-once like
the quantitative eval): `cache_sft`, `temporal_moe`, `sft_baseline` /
`sft_baseline_seq1024` have each been run against a matched baseline.

- **Non-temporal checkpoints** (`cache_sft`, `sft_baseline*`): baseline =
  same model with the cache layer's router monkeypatched to dense routing
  (`dense_patch_router`) -- both tuned and baseline forwards read the
  *same* dense-patched module at the *same* layer, so the comparison is
  same-layer, same-mechanism.
- **`temporal_moe`**: baseline is the fully vanilla, un-adapted, un-wrapped
  model (no temporal mixin at all, no dense patch) -- since the hold/switch
  mixin is architectural, disabling the adapter can't strip it back out,
  so a *literally* mixin-free instance is used instead (corrected after an
  earlier version of this eval used an "adapter-disabled but still
  mixin-wrapped" baseline by mistake).

### Layers analyzed

- The cache layer itself (`--cache-layer`, default = middle layer, 16/32
  for Phi-tiny-MoE).
- Plus `--other-layers-each-side` (default 2) equally-spaced layers before
  AND after it, written to `other_layer_<N>/` subfolders -- checks whether
  the cache reward has side effects on layers it was never trained on,
  using each layer's own native (unpatched, sparse) routing.

### Metrics (per action/generated token, at each analyzed layer)

| metric | definition |
| --- | --- |
| `variance` | Var_e[p_e] -- spread of the full softmax over ALL experts (population variance in probability space, not entropy) |
| `kl_uniform` | KL(p \|\| Uniform(E)) = log(E) - H(p) -- distance to the uniform distribution |
| `skew` | Fisher-Pearson standardized skewness coefficient, m3/m2^1.5 |
| `hit_ratio` | soft LRU cache-hit mass: probability mass this token's routing distribution places on the experts *currently* in the per-sequence LRU working set (cache_size, default 4; LRU touched by top `--touch-topk`, default 2, experts each step) |
| `expert_id` | top-1 expert per token (not an aggregate metric -- feeds `expert_trace`/run-length below) |

Aggregated as mean/median/min/max/std across all tokens+sequences, plus an
8-random-sequence x 4-equally-spaced-token-position case study.

### Visualizations produced (per analyzed layer)

| output | shows |
| --- | --- |
| `expert_probs.png` | expected per-expert routing probability, base vs tuned |
| `skew.png` | sorted cumulative expert mass (Lorenz curve) + per-seq mean hit-ratio histograms |
| `aggregate/*.png` | one grouped bar chart per metric: mean/median/min/max/std, base vs tuned |
| `case_study/*.png` | one figure per metric: stack of small histograms, one per random held-out sequence, at 4 equally-spaced token positions, base vs tuned grouped bars |
| `expert_trace/*.png` | top-1 expert id per generated token as a color strip, one (base, tuned) row pair per random sequence -- "does the policy hold the same expert over contiguous tokens," styled after the routing-timeline figures in Shen & Henderson 2026 |
| `expert_trace/sequences.txt` | the same sequences as plain text, with a `\|` inserted between tokens exactly where the top-1 expert changes -- text-mode equivalent of the color-segment plot |
| `perplexity.png` | standard teacher-forced perplexity on held-out ground-truth completions, base vs tuned (independent of the routing analysis, checks LM quality wasn't traded away) |

### Derived stats (not raw per-token metrics, reported alongside perplexity in `metrics.json`)

| stat | definition |
| --- | --- |
| `expert_run_length` | mean contiguous run length of the top-1 expert per sequence (`run_lengths()` on `expert_id`) -- direct operationalization of "does the policy hold the same expert across consecutive tokens" |
| `switch_rate` | fraction of adjacent action-token pairs where the top-1 expert changes, averaged per sequence then across sequences -- same definition as Table 1 of Shen & Henderson 2026 (Sec. 2.2), computed per analyzed layer here rather than also averaged across layers |

These two are inversely related (`switch_rate` ~= `1/expert_run_length` for
long sequences) but both are reported since `expert_run_length` also
carries distributional info (min/max/std) that `switch_rate` alone doesn't.
