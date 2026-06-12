from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vanilla_moe import (
    BatchedExperts,
    Router,
    SegmentedExperts,
    _init_segmented_from_ffn,
)


@dataclass
class MoEConfig:
    num_experts: int = 64
    num_shared_experts: int = 4
    top_k: int = 4
    aux_loss_coeff: float = 0.01
    jitter_noise: float = 0.0
    expert_dim: int | None = None
    num_copies: int = 0


class MoELayer(nn.Module):

    def __init__(self, original_ffn: nn.Module, config: MoEConfig,
                 hidden_dim: int, intermediate_size: int):
        super().__init__()
        self.config = config
        device = next(original_ffn.parameters()).device
        dtype = next(original_ffn.parameters()).dtype
        expert_intermediate = config.expert_dim or (hidden_dim // config.num_experts)

        self.router = Router(hidden_dim, config.num_experts, config.top_k, dtype=dtype)

        if config.num_copies > 0:
            self.routed_experts = SegmentedExperts(
                hidden_dim, config.num_experts, config.expert_dim,
                dtype=dtype, device=device,
            )
            self._segmented = True
        else:
            self.routed_experts = BatchedExperts(
                config.num_experts, hidden_dim, expert_intermediate,
                dtype=dtype, device=device,
            )
            self._segmented = False

        if config.num_shared_experts > 0:
            self.shared_experts = BatchedExperts(
                config.num_shared_experts, hidden_dim, expert_intermediate,
                dtype=dtype, device=device,
            )
        else:
            self.shared_experts = None

        self._aux_loss = torch.tensor(0.0, device=device, dtype=dtype)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])

        if self.training and self.config.jitter_noise > 0:
            flat = flat * (1.0 + torch.randn_like(flat) * self.config.jitter_noise)

        # shared experts — batched bmm, no Python loop
        if self.shared_experts is not None:
            shared_out = self.shared_experts.forward_all(flat)
        else:
            shared_out = 0.0

        # routed experts — sort-and-slice, O(E) loop
        top_k_probs, top_k_indices, balance_loss = self.router(flat)
        self._aux_loss = balance_loss * self.config.aux_loss_coeff
        self._last_top_k_indices = top_k_indices.detach().reshape(
            orig_shape[0], orig_shape[1], -1
        )

        if self._segmented:
            routed_out = torch.zeros_like(flat)
            for k in range(self.config.top_k):
                expert_indices = top_k_indices[:, k]
                weights = top_k_probs[:, k]
                for e_idx in range(self.config.num_experts):
                    mask = expert_indices == e_idx
                    if not mask.any():
                        continue
                    expert_input = flat[mask]
                    expert_output = self.routed_experts.forward_expert(expert_input, e_idx)
                    routed_out[mask] += weights[mask].unsqueeze(-1) * expert_output
        else:
            routed_out = self.routed_experts(flat, top_k_indices, top_k_probs)

        return (shared_out + routed_out).reshape(orig_shape)


class MoEMixin:

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
                _init_segmented_from_ffn(moe.routed_experts, original_mlp, config.num_copies)
            layer.mlp = moe
            moe_layers.append(moe)

        model._moe_layers = moe_layers
        model.get_moe_loss = lambda: sum(m.aux_loss for m in model._moe_layers)

        label = (f"routed={config.num_experts}, shared={config.num_shared_experts}, "
                 f"top_k={config.top_k}, aux_coeff={config.aux_loss_coeff}")
        if config.expert_dim:
            label += f", expert_dim={config.expert_dim}"
        if config.num_copies > 0:
            label += f", segmented ({config.num_copies} copies)"
        print(f"[DeepSeekMoE] Converted {len(moe_layers)} FFN layers -> MoE ({label})")
        return model

    @staticmethod
    def _find_decoder_layers(model: nn.Module):
        for name, module in model.named_modules():
            if hasattr(module, 'mlp') and (hasattr(module, 'self_attn') or hasattr(module, 'attn')):
                yield module
