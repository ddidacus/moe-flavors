# Research directions explored

Goal: RL-finetune an MoE LLM so one router (middle layer) keeps its expert
traffic inside a fixed-size LRU working set (cache emulation reward), while a
KD/behaviour-cloning signal to the frozen base model preserves text quality.
Combined reward: `r = α·r^BC + (1−α)·r^cache` (Diego's notes / rl_moe paper,
arXiv 2604.20156).

## 1. Custom REINFORCE with baseline (superseded)

`scripts/finetune_moe_reinforce.py`, model `allenai/OLMoE-1B-7B-0924` (64
experts, top-8, 16 layers), prompts from Nemotron-Post-Training-Dataset-v2.

- Rollouts sampled from the mixture `p_mix = (1−τ)p_student + τp_teacher`
  (anti-hacking, per the paper), importance weights `w = p_student/p_mix ≤ 1/(1−τ)`.
- Teacher = frozen base via peft `disable_adapter()` (no second model in memory).
- Cache reward per token `±1[e_t∈C]/T` (sign flipped to positive hit reward
  mid-way), expert sampled `e_t ~ softmax(router logits)`, LRU warmed on prompt.
- Policy term included `log G(e_t)` (router log-prob of the sampled expert) for
  direct credit assignment to the router.
- Scalar EMA baseline on returns-to-go.
- Perf work that carried over: batched full-optimizer-step rollouts + "lazy
  teacher" hierarchical mixture sampling (teacher only runs on the τ-fraction
  of steps, chunked KV catch-up) → ~3–16× throughput.

Why superseded: scalar baseline is structurally mismatched to monotone-decaying
returns-to-go (early tokens always +advantage, late always −), and at lr 1e-5
nothing moved. GRPO's group-relative advantages fix the baseline problem for free.

## 2. TRL GRPO (current main path)

`scripts/finetune_moe_grpo.py`, TRL 1.8 `GRPOTrainer`, G=8 completions/prompt.

- **Cache reward** (custom reward func): sequence-level hit fraction ∈ [0,1] =
  Σ_t 1[e_t∈C]/T — same math as REINFORCE version, one no-grad forward with
  `output_router_logits` + LRU sim (`RewardEngine`, memoized shared pass).
- **KD reward**: negative sampled KL per token,
  `kd_scale × mean_t(log p_ref − log p_policy)` — verified identical to
  `KLReward` in the paper's repo (princeton-polaris-lab/rl_moe), incl. their
  `reward_scale` knob. Ref = frozen base (adapter-disable).
- **β-KL**: TRL's native per-token KL(policy‖base) in-loss penalty; same anchor
  as r^BC in a different form; both knobs exposed (`--alpha`, `--beta`).
- Model switched to **microsoft/Phi-tiny-MoE-instruct** (SlimMoE 3.8B/1.1B
  active, 32 layers, 16 experts top-2, native `phimoe` classes) — smaller,
  2.3× faster steps, chat template, LR from literature (3e-5 = ~10× full-FT
  GRPO per LoRA-without-regret; constant+warmup).
- Cache reward evolved sampled-1-expert → **deterministic top-2** (`--cache-topk
  --cache-experts-per-token 2`): the sampled variant's within-group variance was
  simulation noise, not credit (see 02-results). Top-2 is deployment-faithful.
- LoRA everywhere incl. router and fused experts; cache 4-of-16 experts at
  layer 16/32.

## 3. Temporal MoE mixin on top of GRPO (running)

`src/temporal_moe_wrapper.py` applied inside the GRPO script (`--temporal`).

- Boundary predictor (term_proj1/2 on hidden-state deltas) + hold/switch: at
  non-boundary tokens the routing decision (weights + experts) is forward-filled
  from the last boundary → holding raises cache hits by construction.
- Evolution within this session: v1 trained the boundary net with the mixin's
  ratio(N)+entropy losses; **replaced at Diego's direction by a straight-through
  estimator** — forward keeps hard hold/switch, backward gets
  `∂y/∂p_t = y_fresh − y_held` (both expert outputs computed, STE-mixed), so
  term_proj learns from the GRPO objective alone. No auxiliary losses.
- Cache reward reads the wrapper's *effective executed* per-token decisions
  (`_last_top_k_indices`), not raw router logits — identical reward interface.
- term_proj params trained/checkpointed via peft `modules_to_save`.
- Known v1 gap: generation is hold-free (see 04-next-steps).
