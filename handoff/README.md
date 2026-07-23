# Handoff notes — RL for cache-friendly MoE routing

Working notes for the experiment line started 2026-07-14: finetuning MoE LLMs
with RL so their routers maintain a small LRU working set of experts
(cache-friendly routing for memory-constrained serving, following up on C3T),
while a knowledge-distillation signal anchors text quality to the frozen base.

| file | content |
| --- | --- |
| [01-research-directions.md](01-research-directions.md) | approaches explored, with design rationale |
| [02-results.md](02-results.md) | intermediate results, incl. the reward-hacking incident |
| [03-hypotheses.md](03-hypotheses.md) | what the currently running experiments are testing |
| [04-next-steps.md](04-next-steps.md) | pitfalls, pending work (hold-aware generation), sweep ideas |
| [05-sweep-results.md](05-sweep-results.md) | live sweep table, soft-cache off-policy eval finding |
| [06-training-setup.md](06-training-setup.md) | **large-scale** training spec: 10k sequences, 4-model, 4x80GB (models, dataset, lengths, batch/lr, steps, GPU-hour estimates) |
| [07-eval-setup.md](07-eval-setup.md) | quantitative (lm-eval-harness: MMLU/MMMLU/GSM8K/HumanEval/MATH) + qualitative (routing/cache-metric) eval methodology |
| [08-training-setup-small.md](08-training-setup-small.md) | **small-scale** companion to 06: same models/hyperparameters, ~2k sequences -- run this first |

Code entry points:
- `scripts/finetune_moe_grpo.py` + `scripts/run_finetune_moe_grpo.sh` — current main path (TRL GRPO)
- `scripts/finetune_moe_reinforce.py` + `scripts/run_finetune_moe_reinforce.sh` — custom REINFORCE (superseded)
- `src/cache_reinforce.py` — LRU cache sim, cache/KD reward math (shared)
- `src/temporal_moe_wrapper.py` — temporal boundary/hold mixin (STE variant)

wandb project: `moe-cache-reinforce` (diegocalanzone).
