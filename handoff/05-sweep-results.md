# Sweep results (live table — updated at each monitoring check)

All runs: Phi-tiny-MoE, GRPO/DAPO, `normalize_then_sum` (weights = exact
influence), β=0.08, lr 3e-5, top-2 deterministic cache reward at layer 16/32,
eval/ppl on 256 held-out math/code seqs every 10 steps, **base ppl = 2.661**.
Target 500 steps; ~170 steps per 3h window with auto-resume on resubmit.

"cache" = mean top-2 hit fraction vs the LRU (start ≈ 0.55–0.56 for C=4).
"kd" = mean per-token log p_base − log p_policy (0 = at base; more negative =
more drift). Δppl = eval/ppl − 2.661 (negative is better than base).

## Sweep A — cache size C (α=0.3 fixed)

| C (of 16) | job | status | step | cache | kd | Δppl | notes |
|---|---|---|---|---|---|---|---|
| 2 | 10135074 | WALL@169 | 169 | 0.409 (from 0.358) | −0.011 | +0.001 | cont. died on triton/FS flake; retry wrapper added, resubmitted |
| 4 | 10135072 | WALL@310 | 310 | 0.614–0.619 (from 0.563) | −0.011 | −0.003 | no collapse through 310; resubmitted (next window crosses 340) |
| 8 | (cont.) | WALL@321 | 321 | 0.829 (from 0.796) | −0.011 | −0.009 | resubmitted |
| 12 | 10132112 | WALL@170 | 170 | 0.962 (from 0.938) | −0.012 | −0.004 | window complete |

## Sweep B — α (C=4 fixed); reward = (1−α)·cache + α·KD

| α | job | status | step | cache | kd | Δppl | notes |
|---|---|---|---|---|---|---|---|
| 0.1 | 10134727 | RUNNING | 32 | 0.574 (from 0.563) | −0.003 | −0.009 | started; watch for hacking as cache dominates |
| 0.3 | 10135072 | WALL@310 | 310 | 0.614–0.619 (from 0.563) | −0.011 | −0.003 | |
| 0.5 | 10134728 | PENDING | — | — | — | — | resumes archived 50-50 run @150 (cache 0.596, kd −0.002) |
| 0.7 | 10134729 | PENDING | — | — | — | — | KD-dominant; expect near-flat cache |

## Sweep C — variants at C=4, α=0.3, β=0.08

| variant | job | status | step | cache | kd | Δppl | notes |
|---|---|---|---|---|---|---|---|
| lr=1e-4 | 10135996 | WALL@171 | 171 | 0.631 (from 0.566) | −0.032 | −0.024 | 5× faster climb to ~0.61 by step 35, shelf 0.616 for ~115 steps, lifted to 0.631 in final 10 steps; ppl best-in-class 2.637; continuation not queued (yields short-unkillable to soft-cache run) |
| soft-cache (dense K=16 router @L16, reward = router prob mass on cached experts, LRU touched by top-2) | 10140337→10142627 | RUNNING | 259 | 0.349 (from 0.306; uniform floor 0.25) | −0.024 | +0.004 | dense-model base ppl = 2.672 (own baseline, not 2.661); slow monotone climb, no hard shelf; entropy stable ~0.55; clipped_ratio ~1.0 benign (long math/code, ppl at base) |

## Soft-cache eval @ checkpoint-250 (scripts/eval_soft_cache.py, 2026-07-17)

256 held-out math/code seqs (≤512 tok, prompt+completion), teacher-forced,
dense router at layer 16 for both variants. Result: **base and tuned are
identical off-policy** — soft hit 0.3100 vs 0.3102, dist entropy 2.697 vs
2.696 nats (97% of uniform), top-4 mass 0.329 vs 0.330. The layer-16 router's
LoRA delta is tiny (‖BA‖=0.005, rank 25/32 among routers; experts delta 0.29).
⇒ The on-policy training gain (0.306→0.349) does NOT come from consolidating
the router map; the policy instead steers its own generations toward token
trajectories that reuse recent experts. Off-policy text routes exactly as the
base model. Implication: for router-level consolidation, add direct router
credit (advantage-weighted router log-prob term) — see 04-next-steps.

## Historical reference points (different config, not directly comparable)

| config | steps | cache | quality | outcome |
|---|---|---|---|---|
| α=0 β=0.04 sum_then_norm | 175 | 0.55→0.79 | entropy 0.58→1.04 | fast consolidation, then... |
| α=0.4 β=0.04 kds4 sum_then_norm | 364 | 0.55→0.78 | collapsed | reward hacking (`\boxed{}` spam) at ~step 340 |

## Sweep A snapshot @ common step 169 (extracted 2026-07-16)

Mean over steps 160–169; delta vs steps 1–10; Δppl vs run's own base (2.661).

| C | cache (Δ) | kd | Δppl | miss-rate reduction |
|---|---|---|---|---|
| 2 | 0.409 (+0.051) | −0.011 | +0.001 | 8% |
| 4 | 0.613 (+0.050) | −0.009 | −0.004 | 11% |
| 8 | 0.819 (+0.031) | −0.007 | −0.005 | 15% |
| 12 | 0.962 (+0.031) | −0.012 | −0.005 | 49% |

Absolute gains ~+5 pts (C≤4) vs ~+3 pts (C≥8); relative miss elimination
grows with C. KD cost flat in C; quality at/better than base everywhere.
