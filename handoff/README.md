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

Code entry points:
- `scripts/finetune_moe_grpo.py` + `scripts/run_finetune_moe_grpo.sh` — current main path (TRL GRPO)
- `scripts/finetune_moe_reinforce.py` + `scripts/run_finetune_moe_reinforce.sh` — custom REINFORCE (superseded)
- `src/cache_reinforce.py` — LRU cache sim, cache/KD reward math (shared)
- `src/temporal_moe_wrapper.py` — temporal boundary/hold mixin (STE variant)

wandb project: `moe-cache-reinforce` (diegocalanzone).
