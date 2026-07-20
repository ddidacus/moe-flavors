# Intermediate results (as of 2026-07-15)

All numbers: Phi-tiny-MoE unless stated; cache = LRU of 4 of 16 experts at
layer 16/32; hit rate = fraction of top-2 expert accesses served by the LRU;
wandb project `moe-cache-reinforce`.

## Baselines / reference points

- **Uniform routing** would give hit rate 0.25 (4/16). **Untrained Phi-tiny
  router: 0.55** for deterministic top-2 against the LRU — natural routing is
  already strongly skewed/reuse-heavy (consistent with C3T's findings).
- Sampled-1-expert simulation baseline (OLMoE 16-of-64: ~0.32; Phi: ~0.28).
- Step cost: ~60 s/optimizer step (64 completions, 4×A100) → ~170 steps per 3h
  `short-unkillable` window. OLMoE was ~137 s/step.

## Negative results (all "flat hit rate" runs)

1. REINFORCE + OLMoE, lr 1e-5: flat ~0.33 for 76 steps. Cause: tiny updates
   (Adam step ≈ lr; LoRA-from-zero) + scalar-baseline position bias.
2. GRPO + OLMoE, lr 1e-5: flat, grad_norm ~0.06, kl ~0.001 after 20 steps —
   same tiny-update diagnosis.
3. GRPO + Phi, lr 3e-5, **sampled-expert reward**: policy moved (kl 0.017) but
   hit rate flat at 0.28 for 70 steps. Diagnosis confirmed: within-group reward
   std (~0.02) was dominated by expert-sampling noise in the *reward
   simulation*, not by behaviour differences → advantages ranked luck.
   Fix: deterministic top-2 reward.

## Positive result: routing consolidation works

GRPO + Phi + top-2 reward, lr 3e-5, α=0 (cache-only) + β=0.04:
hit rate **0.55 → 0.79 by step 175**, smooth monotone climb, within-group std
~0.04 (real signal). α=0.4 β=0.04 run: same trajectory to ~0.68 by step 340 at
a bounded KD cost ~−0.045 nats/token, coherent completions (verified samples).

## Reward hacking incident (important)

α=0.4 β=0.04 run, steps ~340–364: cache reward surged 0.71→0.78 while
completions degenerated into `\boxed{}`-spam / pseudo-code gibberish.
Signature: entropy 0.58→1.18 (slow rise all along, sharp at the end), KL
excursions (one window avg ~97 nats), KD decline accelerating. Detected via
logged completion text (parquets in save dir + wandb table), which **led the
metrics** — entropy alone lagged. β=0.08 resumed from the (already
contaminated) checkpoint-350 stabilized entropy but did not reverse the
learned degenerate mode → restarted fresh. Lesson: keep ≥3 checkpoints
(save_total_limit was 1; the clean rewind point had been rotated away).

## Reward-scale calibration

Within-group stds: cache ~0.042 vs KD ~0.011 → with weights (0.6, 0.4) the
effective influence was ~85/15. GRPO advantages are group-centered, so only
stds matter (means cancel). Fix: `--kd-scale 4` (= rl_moe's `reward_scale`),
making α≈true influence ratio. Means on wandb (cache ~0.55, kd ~1e-3) are
expected and harmless.

## Infrastructure that now works (was nontrivial)

- Resume across 3h walls: peft 0.19 × transformers 5.8 has two adapter-load
  bugs (WeightConverter kwarg; ParamWrapper state-dict load) — patched in the
  GRPO script (conversion bypass + manual `load_state_dict` remap, handles
  lora_A/B and modules_to_save keys). Verified end-to-end.
- PhiMoE LoRA: peft's fused-experts conversion lacks a `phimoe` entry →
  explicit `target_parameters=[experts.gate_up_proj, experts.down_proj,
  router.weight]` with r/α doubled on gate_up. Router module can't be
  module-wrapped (returns tuple) → param-wrap its weight.
- TRL enables the MoE **load-balancing aux loss by default** — directly opposes
  cache consolidation; we set `router_aux_loss_coef=0`.

## Update 2026-07-15 (afternoon): exact reward balancing + ppl eval

- **Held-out perplexity eval added** (`eval/ppl`, 256 reserved Nemotron math/code
  conversations, every 10 steps, vs one-time `eval/ppl_base`=2.661). Two NCCL
  watchdog crashes on the way: (a) rank-0-only eval stalls other ranks at a
  collective; (b) calling trainer.log() mid-step triggers TRL's completions
  upload on rank 0 (minutes). Fixes: all ranks run the eval synchronously,
  eval logs go straight to wandb without explicit step, ddp_timeout=3600.
- **normalize_then_sum aggregation** (TRL): per-component group z-scoring makes
  reward_weights exact influence ratios; kd_scale becomes a no-op.
- **50-50 run (α=0.5)**: 165 steps, cache 0.563→0.596 (+4 pts, slow), kd −0.002,
  KL 0.018, entropy flat, eval/ppl 2.655 (slightly BETTER than base). Cleanest
  run so far; quality fully preserved.
- **Why 50-50 is slow (measured, 1184 groups)**: within-group corr(cache, kd)
  = −0.14 (mild real trade-off), and the kd component's raw group-std is only
  ~0.006 nats at KL 0.018 — z-scoring amplifies near-noise to 50% of the
  advantage. Net cache learning pressure ~¼ of the α=0 run; slopes match.
- **Paper recheck (Shen & Henderson)**: their return contains ONLY the KD
  reward (nats, unscaled); switching pressure enters as deliberation cost η
  (0.02–0.04, advantage-units margin) in the termination gradient — never a
  reward sum, so no scale-balancing problem exists there. Our reward-mix is a
  deliberate departure. Their anti-hacking is mixture sampling (τ=0.2), which
  our GRPO path lacks — likely why our α=0/β=0.04 run hacked.
- **DAPO**: TRL 1.8's loss_type default is already 'dapo' — all runs used it;
  now explicit.
- Current: α=0.3 (70/30) fresh run — reproduces the historically productive
  effective ratio under exact control, with eval/ppl as the quality guardrail.
