# Hypotheses currently being tested

## H1 — balanced rewards + stronger anchor prevent the hacking collapse

Run: `grpo-phi-tiny-moe-instruct-cache-mathcode-a0.4-b0.08-kds4.0-c4-topk2`
(job 10129205, fresh from step 0).

Setup: α=0.4 (true 60/40 cache/KD influence via kd_scale=4), β=0.08 (double
the KL anchor of the run that collapsed), everything else identical to the
collapsed β=0.04 run.

Predictions:
- cache hit rate climbs past 0.65 but more slowly than the α=0/β=0.04 runs;
- KD reward stays shallow (>−0.03·4 = −0.12 scaled) or recovers, instead of
  accelerating downward around step ~340;
- entropy stays < 1.0 and completions remain coherent through 500 steps.

Falsifier: same `\boxed{}`-spam collapse despite the balanced reward → the
hacking mode is not a scale/anchor problem; next levers are lower LR, larger G,
or masking truncated completions.

## H2 — a boundary/hold policy can be learned from the GRPO objective alone

Run: `...-a0.4-b0.08-kds4.0-c4-topk2_tmoeN8` (job 10129382, temporal STE).

Setup: exact copy of H1 plus `TemporalMoEWrapper` (STE variant): hard
hold/switch decisions in forward, `∂y/∂p_t = y_fresh − y_held` in backward; no
ratio/entropy regularizers; cache reward computed on the effective held
decisions. Boundary rate logged as `train/boundary_rate`.

Predictions:
- boundary_rate drops below ~0.5 (holding emerges) driven purely by the cache
  reward's preference for held segments;
- hit rate exceeds H1's at matched KD cost (holding is a structurally easier
  path to hits than reshaping the router);
- if boundary_rate → 0 (degenerate full-hold), quality (KD/entropy) should
  degrade — the KD term must push back; where it equilibrates is the result.

Caveat for interpretation: rollouts are hold-free (v1 limitation), so
generation is off-policy w.r.t. the held-routing model being trained; treat
quality conclusions with care until hold-aware generation lands.

## Background hypothesis (the point of the whole line)

RL with a cache-emulation reward can push a pretrained MoE's *effective*
routing toward a small working set at modest quality cost — turning C3T's
serving-time trade-off (working-set restriction costs NLL) into a train-time
property (the model routes cache-friendly by itself). The α/β frontier maps
hit-rate gain vs. KL-to-base cost.
