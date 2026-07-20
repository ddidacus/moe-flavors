"""Cache-aware REINFORCE utilities for MoE finetuning.

Implements the two reward terms:

  (1) Knowledge distillation / behaviour cloning (per token):
          r_t^BC = log p_teacher(a_t | x, a_<t) - log p_student(a_t | x, a_<t)
      where -r_t^BC is an unbiased estimator of the reverse KL
      KL(p_student || p_teacher). Tokens are sampled from the mixture
          p_mix = (1 - tau) * p_student + tau * p_teacher
      (to prevent reward hacking) and corrected with importance weights
          w_t = p_student(a_t | x, a_<t) / p_mix(a_t | x, a_<t)  <=  1/(1-tau).

  (2) Cache emulation (per token): a fixed-size LRU working set C of experts
      is simulated over the sequence at one router; an expert e_t is sampled
      from the router distribution G(x_t) and
          r_t^cache = 1[e_t in C] / T
      with T the number of rewarded (generated) tokens in the sequence
      (positive reward for cache hits).
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F


class LRUExpertCache:
    """Fixed-capacity LRU working set of expert ids (one cache per router).

    On access: a hit refreshes recency; a miss inserts the expert and evicts
    the least-recently-used one once capacity is exceeded (the miss models an
    expert transfer into the working set).
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"cache capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._slots = OrderedDict()  # expert_id -> None, last item = most recent

    def reset(self):
        self._slots.clear()

    def __len__(self):
        return len(self._slots)

    def __contains__(self, expert_id: int):
        return expert_id in self._slots

    @property
    def experts(self):
        return list(self._slots.keys())

    def access(self, expert_id: int) -> bool:
        """Touch an expert. Returns True on hit, False on miss."""
        hit = expert_id in self._slots
        if hit:
            self._slots.move_to_end(expert_id)
        else:
            self._slots[expert_id] = None
            if len(self._slots) > self.capacity:
                self._slots.popitem(last=False)
        return hit


@torch.no_grad()
def cache_emulation_rewards(router_logits, valid_mask, action_mask, cache_size,
                            experts_per_token=1, use_topk=False, generator=None,
                            experts=None, soft=False):
    """Simulate one LRU cache per sequence and emit per-token cache rewards.

    Args:
        router_logits: (B, S, E) router logits at the cached layer, over the
            full sequence (prompt + generation). May be None when `experts`
            is given.
        valid_mask: (B, S) bool, True on real (non-padding) tokens. Valid
            prompt tokens warm the cache but emit no reward.
        action_mask: (B, S) bool, True on generated tokens (rewarded).
        cache_size: LRU capacity (number of experts in the working set).
        experts_per_token: experts drawn per token.
        use_topk: if True take the router's top-k experts (deterministic,
            mirrors real routing) instead of sampling e_t ~ G(x_t).
        generator: optional torch.Generator for reproducible sampling.
        experts: optional (B, S, k) long, precomputed per-token expert
            decisions (e.g. the effective held decisions of a temporal
            routing wrapper); overrides sampling/topk from router_logits.
        soft: if True the per-token reward is the router probability mass
            (full softmax over ALL experts) on the experts currently in the
            cache — a soft hit rate in [0, 1]. The LRU is still touched by
            the top-`experts_per_token` experts, so cache dynamics match the
            hard variant. Intended for a dense router (K = all experts) at
            the cached layer.

    Returns:
        rewards: (B, S) float32, r_t^cache = (hits_t / experts_per_token) / T
            on action positions (soft: cached probability mass / T), 0 elsewhere.
        experts: (B, S, experts_per_token) long, experts drawn at every position.
        hit_rate: float, hits / accesses over action positions (for logging;
            soft: mean cached probability mass per action token).
    """
    probs_cpu = None
    if experts is not None:
        B, S, experts_per_token = experts.shape
        device = experts.device
    else:
        B, S, E = router_logits.shape
        device = router_logits.device
        probs = torch.softmax(router_logits.float(), dim=-1)
        if soft:
            probs_cpu = probs.cpu()
            experts = probs.topk(experts_per_token, dim=-1).indices
        elif use_topk:
            experts = probs.topk(experts_per_token, dim=-1).indices
        else:
            experts = torch.multinomial(
                probs.reshape(-1, E), experts_per_token,
                replacement=False, generator=generator,
            ).reshape(B, S, experts_per_token)

    rewards = torch.zeros(B, S, dtype=torch.float32)
    experts_cpu = experts.cpu()
    valid_cpu, action_cpu = valid_mask.cpu(), action_mask.cpu()
    hits = accesses = 0
    for b in range(B):
        cache = LRUExpertCache(cache_size)
        T = max(int(action_cpu[b].sum()), 1)
        for s in range(S):
            if not valid_cpu[b, s]:
                continue
            if probs_cpu is not None:
                # soft: probability mass on the cache as-of the previous token
                if action_cpu[b, s]:
                    cached = cache.experts
                    mass = float(probs_cpu[b, s, cached].sum()) if cached else 0.0
                    rewards[b, s] = mass / T
                    hits += mass
                    accesses += 1
                for e in experts_cpu[b, s].tolist():
                    cache.access(e)
            else:
                token_hits = 0
                for e in experts_cpu[b, s].tolist():
                    if cache.access(e):
                        token_hits += 1
                if action_cpu[b, s]:
                    rewards[b, s] = (token_hits / experts_per_token) / T
                    hits += token_hits
                    accesses += experts_per_token
    hit_rate = hits / max(accesses, 1)
    return rewards.to(device), experts, hit_rate


def kd_reward_terms(student_logits, teacher_logits, actions, tau):
    """KD/BC reward, importance weights, and student log-probs of the actions.

    Args:
        student_logits: (B, S, V) logits predicting `actions` (keeps grad).
        teacher_logits: (B, S, V) teacher logits at the same positions.
        actions: (B, S) sampled tokens a_t.
        tau: teacher fraction of the sampling mixture p_mix.

    Returns:
        logp_student: (B, S) log p_student(a_t) — differentiable, this is the
            REINFORCE policy term.
        r_bc: (B, S) detached, log p_teacher(a_t) - log p_student(a_t).
        w: (B, S) detached importance weights p_student(a_t) / p_mix(a_t),
            bounded above by 1/(1-tau).
    """
    idx = actions.unsqueeze(-1)
    logp_student = F.log_softmax(student_logits.float(), dim=-1) \
        .gather(-1, idx).squeeze(-1)
    with torch.no_grad():
        logp_teacher = F.log_softmax(teacher_logits.float(), dim=-1) \
            .gather(-1, idx).squeeze(-1)
        lp_s = logp_student.detach()
        r_bc = logp_teacher - lp_s
        # log p_mix = logaddexp(log(1-tau) + lp_s, log(tau) + lp_t);
        # tau in {0,1} yields -inf terms which logaddexp handles exactly.
        t = torch.tensor(tau, dtype=lp_s.dtype, device=lp_s.device)
        logp_mix = torch.logaddexp(torch.log1p(-t) + lp_s,
                                   torch.log(t) + logp_teacher)
        w = (lp_s - logp_mix).exp()
    return logp_student, r_bc, w


def rewards_to_go(rewards, gamma=1.0):
    """Undiscounted/discounted return-to-go G_t = sum_{s>=t} gamma^{s-t} r_s.

    rewards: (B, S) with zeros on non-action positions.
    """
    if gamma == 1.0:
        return rewards.flip(1).cumsum(1).flip(1)
    G = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(0), device=rewards.device,
                          dtype=rewards.dtype)
    for s in range(rewards.size(1) - 1, -1, -1):
        running = rewards[:, s] + gamma * running
        G[:, s] = running
    return G
