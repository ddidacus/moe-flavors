# Things to keep in mind / next steps

## Committed next work

1. **Hold-aware generation** (Diego confirmed we need it). Today the temporal
   wrapper only holds routing on full-sequence forwards; KV-cache decoding
   passes 1-token windows, so `p_t≡1` → fresh routing during rollouts, and the
   trained (held) policy is off-policy w.r.t. the sampler. Needs a stateful
   wrapper: persist per-sequence last held (weights, experts) and the previous
   token's hidden state across decode steps (boundary needs the delta), keyed
   on cache_position, robust to batch reordering. See memory note
   `temporal-hold-aware-generation`.

2. **α/β/kd_scale/cache-size sweep** once H1/H2 verdicts are in. SAVE_DIR/run
   names already encode the config; every knob is an env var on
   `run_finetune_moe_grpo.sh` (ALPHA, BETA, KD_SCALE, CACHE_SIZE, CACHE_LAYER,
   TEMPORAL, RATIO_N, NUM_GEN, MODEL).

3. **Held-out evaluation** beyond training metrics: teacher-forced NLL vs the
   base model, plus a real transfer-count/latency replay like C3T's
   `cache_serving_stall_benchmark`, to connect hit-rate gains to ms/token.

## Ideas not yet tried

- Smaller cache (2-of-16) → harder task, richer within-group variance.
- Direct router credit in GRPO (advantage-weighted router log-prob term, as in
  the REINFORCE script) if indirect gradients prove too weak with holding.
- Multi-layer caches (one LRU per constrained layer, like C3T's layer set).
- `mask_truncated_completions=True` and/or larger G against hacking.
- OLMoE re-run with the lessons learned (top-2 reward, lr 3e-5, kd_scale).

## Pitfalls (hard-won; do not rediscover)

- **TRL enables MoE load-balancing aux loss by default** — it fights cache
  consolidation. Keep `router_aux_loss_coef=0`.
- **Sampled-expert cache reward = noise** in GRPO groups; use deterministic
  top-k (also deployment-faithful).
- **GRPO reward influence ∝ weight × within-group std**, not means. Calibrate
  scales (kd_scale) or α is a lie.
- **Reward hacking shows up in completion text before metrics.** Keep
  `log_completions=True` and read samples; entropy is a lagging indicator.
  Keep `save_total_limit≥3` for clean rewind points.
- **Gradient checkpointing detaches forward-side-effect losses** (no-grad
  first pass) — relevant to any auxiliary loss stored by module forwards.
  Currently checkpointing is off for temporal runs (also 2× expert compute
  from the STE double pass).
- peft 0.19 × transformers 5.8: `phimoe` missing from fused-experts conversion
  registry; ParamWrapper forbids dropout; adapter checkpoint load needs the
  manual loader in `finetune_moe_grpo.py`; PhiMoE router module returns a
  tuple → target `router.weight` as a parameter, never as a module.
- TRL 1.8 has no `max_prompt_length` — pre-truncate prompts in the dataset.
- tqdm shows micro-batches, not optimizer steps (past confusion: "step 68").
- 3h `short-unkillable` wall ≈ step ~170; resubmitting the same sbatch
  auto-resumes (config-encoded save dirs prevent cross-config resumes).
- Do not `uv sync` blindly: `trl==1.8.0` is pip-installed but not in
  pyproject.toml yet.
