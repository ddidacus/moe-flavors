# Loss math — cache_sft (soft-cache GRPO + SFT NLL)

How `cache_sft`'s total training loss is constructed: the soft-cache reward
(forward pass), how it enters the GRPO/DAPO policy loss, the SFT NLL term,
how the two combine, and where gradient actually flows into the router.
Code: `scripts/finetune_moe_grpo.py` (`GRPOTrainerWithSFT`), `src/
cache_reinforce.py` (`cache_emulation_rewards`).

## 1. Setup

At the cache layer `ℓ`, the router is monkeypatched dense (`--soft-cache`,
`dense_patch_router` in `eval_soft_cache.py` / the inline equivalent in
`finetune_moe_grpo.py`): for hidden state `h_t` at token `t`,

```
g_t = W_router h_t + b_router                      (router logits, ∈ R^E)
p_t = softmax(g_t)                                  (E = 16 experts, ALL active)
```

Every expert receives a weighted contribution `p_t,e` for every token — no
top-k truncation (see the earlier discussion of this in the conversation:
`selected = arange(E)` for every row). This full softmax is what makes the
reward below differentiable in principle.

A per-sequence LRU cache `C_t ⊆ {1..E}`, `|C_t| = cache_size` (4), is warmed
by both prompt and completion tokens: at every token the top-`touch_topk`
(2) experts by `p_t` are "accessed" (LRU-touched), updating `C_t → C_{t+1}`.

## 2. Soft-cache reward (forward pass, per action/generated token)

```
r_t^cache = ( Σ_{e ∈ C_t} p_t,e ) / T
```

`T` = number of generated (action) tokens in the sequence — dividing by `T`
turns the per-token cache-hit mass into a per-sequence **return** in `[0,1]`
when summed over `t`: `R^cache = Σ_t r_t^cache = (1/T) Σ_t Σ_{e∈C_t} p_t,e`,
i.e. the *mean* soft hit-rate over the sequence. `C_t` is the cache state
**as of the previous token** (computed under `torch.no_grad()` — see
code excerpt below), so `r_t^cache` only carries gradient through `p_t`
itself, not through the cache contents.

## 3. GRPO/DAPO policy loss

For a group of `G=8` completions sampled per prompt (temperature 1.0),
group-relative advantage replaces a learned value baseline:

```
A_i = ( R_i^cache − mean_j(R_j^cache) ) / std_j(R_j^cache)     (i = 1..G, one prompt's group)
```

(`multi_objective_aggregation=normalize_then_sum` is irrelevant here since
`reward_funcs = [cache_reward]` — a single reward, no aggregation needed.)

The DAPO-variant clipped policy loss (TRL's `loss_type="dapo"`, `β`=0.08 KL
coefficient against the frozen base via adapter-disable):

```
L_GRPO = − E_t [ min( ρ_t A_i, clip(ρ_t, 1−ε, 1+ε) A_i ) ] + β · KL(π_θ ‖ π_ref)_t

ρ_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)     (importance ratio, token-level)
```

**Crucially: `A_i` (built from `R^cache`) is treated as a constant scalar
here** — TRL's GRPO objective never backpropagates through the reward
function itself; the reward only ever multiplies a log-probability-derived
term. So even though `r_t^cache` is *mathematically* differentiable w.r.t.
`p_t` (dense softmax), **that gradient path is never taken** in the
GRPO/DAPO loss. The router only receives gradient from `L_GRPO` indirectly,
through however `p_t` at the cache layer affects the downstream token
log-probs `π_θ(a_t|s_t)` (this was diagnosed earlier in this project: the
"sampled" cache-reward mode was flat/uninformative for exactly this reason
— a reward decorrelated from the actual sampling distribution has no
signal to give through this path).

## 4. SFT NLL loss

Standard teacher-forced cross-entropy on the dataset's own ground-truth
completion (`target_ids`, independent of the reward/rollout above):

```
L_SFT = − (1/|target|) Σ_t log π_θ( target_t | prompt, target_{<t} )
```

This term's gradient flows directly and only through the ordinary
teacher-forced log-likelihood chain — INCLUDING through the dense router at
the cache layer, exactly as in plain SFT. **This is the primary gradient
path that actually reaches the router's weights via a signal correlated
with the model's own realized behavior**, complementing `L_GRPO`'s indirect
path above.

## 5. Combination + total gradient

```
L_total = rl_coef · L_GRPO + sft_coef · L_SFT        (rl_coef=2.0, sft_coef=0.5 for cache_sft)

∇_θ L_total = rl_coef · ∇_θ L_GRPO + sft_coef · ∇_θ L_SFT
```

Both terms are summed directly into ONE scalar loss before a single
`.backward()` call — not mixed into the reward, not alternated between
steps. By linearity of the gradient operator, `∇_θ L_total` is exactly the
weighted sum of each term's own gradient; there's no interaction term. In
practice:
- `∇_θ L_GRPO` nudges the *policy* (token distribution) toward
  higher-cache-hit-rate rollouts, via the standard policy-gradient/
  importance-ratio path — it does NOT directly push the router's raw
  logits `g_t` toward concentrating mass on `C_t`.
- `∇_θ L_SFT` nudges every parameter (LoRA-adapted attention + fused expert
  weights + router) toward reproducing the dataset's own completions,
  including whatever routing behavior was implicit in those examples —
  independent of the cache reward's advantage signal entirely.
- `rl_coef=2.0 > sft_coef=0.5`: after diagnosing a run where the SFT term
  dominated the gradient and drove KL/entropy drift without cache-reward
  gains, `rl_coef` was raised (see `handoff/02-results.md` for the
  underlying incident) so the RL term isn't swamped.

## Code excerpt

`scripts/finetune_moe_grpo.py`, `GRPOTrainerWithSFT`:

```python
class GRPOTrainerWithSFT(GRPOTrainer):
    def __init__(self, *args, sft_coef=0.0, rl_coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_coef = sft_coef
        self.rl_coef = rl_coef

    def _sft_nll_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        device = prompt_ids.device
        target_ids = inputs["target_ids"].to(device)
        target_mask = inputs["target_mask"].to(device)
        # prompt_ids: left-padded (TRL generation convention); target_ids:
        # right-padded. Concatenating them mirrors exactly how GRPOTrainer
        # itself scores prompt+completion, so the same attention-mask-derived
        # position handling applies.
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        full_mask = torch.cat([prompt_mask, target_mask], dim=1)
        logits = model(input_ids=full_ids, attention_mask=full_mask,
                      use_cache=False).logits
        P = prompt_ids.size(1)
        pred_logits = logits[:, P - 1:-1]  # predict target_ids[:, t] from position P-1+t
        logp = F.log_softmax(pred_logits.float(), dim=-1) \
            .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        mask = target_mask.float()
        return -(logp * mask).sum() / mask.sum().clamp(min=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)  # this IS L_GRPO
        total = self.rl_coef * loss
        if self.sft_coef > 0 and "target_ids" in inputs:
            sft_loss = self._sft_nll_loss(model, inputs)
            mode = "train" if model.training else "eval"
            self._metrics[mode]["sft_nll"].append(sft_loss.item())
            total = total + self.sft_coef * sft_loss
        return total
```

`src/cache_reinforce.py`, the soft-cache reward's per-token loop (`soft=True`
branch of `cache_emulation_rewards`):

```python
probs = torch.softmax(router_logits.float(), dim=-1)     # p_t, full E-dim softmax
if soft:
    probs_cpu = probs.cpu()
    experts = probs.topk(experts_per_token, dim=-1).indices   # top-touch_topk, LRU-touch only

...
for s in range(S):
    if not valid_cpu[b, s]:
        continue
    if probs_cpu is not None:
        # soft: probability mass on the cache as-of the previous token
        if action_cpu[b, s]:
            cached = cache.experts                      # C_t (before this token's touch)
            mass = float(probs_cpu[b, s, cached].sum()) if cached else 0.0
            rewards[b, s] = mass / T                     # r_t^cache
            hits += mass
            accesses += 1
        for e in experts_cpu[b, s].tolist():
            cache.access(e)                              # C_t -> C_{t+1}
    ...
```

`reward_funcs = [engine.cache_reward]`, `reward_weights = [1.0]` in `main()`
— a single reward, group-relative advantage computed by TRL's own
`GRPOTrainer.compute_loss` (inherited, not reimplemented here) from
`R^cache = Σ_t r_t^cache` per completion.
