import argparse
import random
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-20

@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    aux_loss_coeff: float = 0.01  # load-balancing loss weight
    jitter_noise: float = 0.0     # optional input jitter for training
    expert_dim: int | None = None
    num_copies: int = 0

class Router(nn.Module):
    """Top-k gating router with optional load-balancing auxiliary loss."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # x: (batch * seq_len, hidden_dim)
        logits = self.gate(x)                          # (tokens, num_experts)
        probs = F.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        aux_loss = self._load_balancing_loss(probs, top_k_indices)
        return top_k_probs, top_k_indices, aux_loss

    def _load_balancing_loss(self, probs: torch.Tensor, indices: torch.Tensor):
        """Switch Transformer style load-balancing loss."""
        num_tokens = probs.shape[0]
        # fraction of tokens routed to each expert
        one_hot = F.one_hot(indices, self.num_experts).to(probs.dtype)  # (tokens, top_k, E)
        tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)      # (E,)
        # mean routing probability per expert
        router_prob_per_expert = probs.mean(dim=0)               # (E,)
        # p(route_expert) * tokens_count => sum(tokens_per_expert_expectation * experts) => total tokens over all experts 
        return (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=None, device=None):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class SegmentedExperts(nn.Module):
    """Virtual experts carved from concatenated FFN weight matrices.

    Each virtual expert is a contiguous slice of expert_dim neurons along the
    intermediate axis of three shared weight matrices (gate_proj, up_proj,
    down_proj), preserving the original gated MLP pattern.
    """

    def __init__(self, hidden_dim: int, num_experts: int, expert_dim: int,
                 dtype=None, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        total_intermediate = num_experts * expert_dim
        self.gate_proj = nn.Linear(hidden_dim, total_intermediate, bias=False,
                                   dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_dim, total_intermediate, bias=False,
                                 dtype=dtype, device=device)
        self.down_proj = nn.Linear(total_intermediate, hidden_dim, bias=False,
                                   dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        s = expert_idx * self.expert_dim
        e = s + self.expert_dim
        gate_out = F.linear(x, self.gate_proj.weight[s:e])
        up_out = F.linear(x, self.up_proj.weight[s:e])
        hidden = self.act(gate_out) * up_out
        return F.linear(hidden, self.down_proj.weight[:, s:e])


def _init_segmented_from_ffn(segmented: SegmentedExperts, original_mlp: nn.Module,
                              num_copies: int, noise_std: float = 0.01):
    with torch.no_grad():
        orig_gate = original_mlp.gate_proj.weight.data
        orig_up = original_mlp.up_proj.weight.data
        orig_down = original_mlp.down_proj.weight.data
        segmented.gate_proj.weight.data.copy_(orig_gate.repeat(num_copies, 1))
        segmented.up_proj.weight.data.copy_(orig_up.repeat(num_copies, 1))
        segmented.down_proj.weight.data.copy_(orig_down.repeat(1, num_copies))
        if noise_std > 0:
            segmented.gate_proj.weight.data += torch.randn_like(segmented.gate_proj.weight) * noise_std
            segmented.up_proj.weight.data += torch.randn_like(segmented.up_proj.weight) * noise_std
            segmented.down_proj.weight.data += torch.randn_like(segmented.down_proj.weight) * noise_std


class MoELayer(nn.Module):
    """
    Replaces a single dense FFN with N small expert MLPs + a router.
    Each expert has hidden_dim // num_experts as its intermediate size.
    """

    def __init__(self, original_ffn: nn.Module, config: MoEConfig, hidden_dim: int, intermediate_size: int):
        super().__init__()
        self.config = config
        device = next(original_ffn.parameters()).device
        dtype = next(original_ffn.parameters()).dtype
        self.router = Router(hidden_dim, config.num_experts, config.top_k, dtype=dtype)
        if config.num_copies > 0:
            self.experts = SegmentedExperts(
                hidden_dim, config.num_experts, config.expert_dim,
                dtype=dtype, device=device,
            )
            self._segmented = True
        else:
            expert_intermediate = config.expert_dim or (hidden_dim // config.num_experts)
            self.experts = nn.ModuleList(
                [ExpertMLP(hidden_dim, expert_intermediate, dtype=dtype, device=device)
                 for _ in range(config.num_experts)]
            )
            self._segmented = False
        self._aux_loss = torch.tensor(0.0, device=device, dtype=dtype)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape                          # (batch, seq_len, hidden)

        flat = x.reshape(-1, orig_shape[-1])          # (tokens, hidden)
        if self.training and self.config.jitter_noise > 0:
            flat = flat * (1.0 + torch.randn_like(flat) * self.config.jitter_noise)

        top_k_probs, top_k_indices, aux_loss = self.router(flat)
        self._aux_loss = aux_loss
        self._last_top_k_indices = top_k_indices.detach().reshape(
            orig_shape[0], orig_shape[1], -1
        )

        out = torch.zeros_like(flat)

        for k in range(self.config.top_k):
            expert_indices = top_k_indices[:, k]       # (tokens,)
            weights = top_k_probs[:, k]                # (tokens,)

            for e_idx in range(self.config.num_experts):
                mask = expert_indices == e_idx
                if not mask.any():
                    continue
                expert_input = flat[mask]
                if self._segmented:
                    expert_output = self.experts.forward_expert(expert_input, e_idx)
                else:
                    expert_output = self.experts[e_idx](expert_input)
                out[mask] += weights[mask].unsqueeze(-1) * expert_output

        return out.reshape(orig_shape)


class MoEMixin:
    """
    Mixin that patches a HuggingFace causal LM to use MoE feed-forward layers.

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        MoEMixin.apply(model, moe_config)

    The mixin:
      1. Walks the model's decoder layers
      2. Replaces each .mlp with a MoELayer wrapping copies of the original
      3. Patches the forward pass to accumulate and return auxiliary losses
    """

    @staticmethod
    def apply(model: nn.Module, config: MoEConfig) -> nn.Module:
        hidden_dim = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        decoder_layers = list(MoEMixin._find_decoder_layers(model))

        if config.num_copies > 0:
            assert config.expert_dim is not None and config.expert_dim > 0
            total_intermediate = config.num_copies * intermediate_size
            assert total_intermediate % config.expert_dim == 0, (
                f"num_copies * intermediate_size ({total_intermediate}) must be "
                f"divisible by expert_dim ({config.expert_dim})"
            )
            config.num_experts = total_intermediate // config.expert_dim

        moe_layers = []
        for layer in decoder_layers:
            original_mlp = layer.mlp
            moe = MoELayer(original_mlp, config, hidden_dim, intermediate_size)
            if config.num_copies > 0:
                _init_segmented_from_ffn(moe.experts, original_mlp, config.num_copies)
            layer.mlp = moe
            moe_layers.append(moe)

        model._moe_layers = moe_layers
        model.get_moe_loss = lambda: sum(m.aux_loss for m in model._moe_layers)

        label = f"experts={config.num_experts}, top_k={config.top_k}"
        if config.num_copies > 0:
            label += f", segmented ({config.num_copies} copies, expert_dim={config.expert_dim})"
        print(f"[MoEMixin] Converted {len(moe_layers)} FFN layers -> MoE ({label})")
        return model

    @staticmethod
    def _find_decoder_layers(model: nn.Module):
        """Locate transformer decoder layers by searching for .mlp attribute."""
        for name, module in model.named_modules():
            if hasattr(module, 'mlp') and (hasattr(module, 'self_attn') or hasattr(module, 'attn')):
                yield module