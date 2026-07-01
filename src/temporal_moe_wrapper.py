from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalWrapConfig:
    ratio_loss_N: list[int] = field(default_factory=lambda: [3])
    ratio_loss_alpha: float = 0.3
    entropy_threshold: float = 0.1
    entropy_alpha: float = 1.0
    learnable_N: bool = False

    def __post_init__(self):
        if isinstance(self.ratio_loss_N, int):
            self.ratio_loss_N = [self.ratio_loss_N]


class _RouterCompat:
    """Exposes num_experts / top_k / _boundary_threshold for eval code."""
    def __init__(self, num_experts, top_k, threshold):
        self.num_experts = num_experts
        self.top_k = top_k
        self._boundary_threshold = threshold


class TemporalMoEWrapper(nn.Module):
    """Wraps an existing MoE sparse block to add temporal boundary routing.

    Keeps all original experts and the shared expert intact.  Only adds
    boundary prediction (delta-state termination) and forward-fills top-k
    routing decisions across segments."""

    def __init__(self, original_moe_block, config: TemporalWrapConfig, N: int):
        super().__init__()
        self.moe_block = original_moe_block

        gate = getattr(original_moe_block, 'gate', None) or original_moe_block.router
        self._gate_attr = 'gate' if hasattr(original_moe_block, 'gate') else 'router'
        hidden_dim = gate.hidden_dim
        num_experts = gate.num_experts
        top_k = gate.top_k
        dtype = gate.weight.dtype
        device = gate.weight.device

        self.term_proj1 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.term_proj2 = nn.Linear(hidden_dim, 1, dtype=dtype, device=device)
        self._boundary_threshold = 0.5

        self.router = _RouterCompat(num_experts, top_k, self._boundary_threshold)
        self.config = type("C", (), {"num_experts": num_experts, "top_k": top_k})()

        self._ratio_loss_alpha = config.ratio_loss_alpha
        self._entropy_threshold = config.entropy_threshold
        self._entropy_alpha = config.entropy_alpha
        self._learnable_N = config.learnable_N
        if config.learnable_N:
            raw = torch.log(torch.tensor(float(N) - 1.0).exp() - 1.0)
            self._log_N = nn.Parameter(raw.to(device=device, dtype=torch.float32))
        else:
            self._N = N
        self._ratio_loss = torch.tensor(0.0, device=device, dtype=dtype)
        self._padding_mask = None

    @property
    def learned_N(self) -> float:
        if self._learnable_N:
            return 1.0 + F.softplus(self._log_N).item()
        return float(self._N)

    @property
    def ratio_loss(self):
        return self._ratio_loss

    def _compute_boundary(self, x):
        B, L, D = x.shape
        delta = x[:, 1:, :] - x[:, :-1, :]
        pt_minus_1 = torch.sigmoid(
            self.term_proj2(F.relu(self.term_proj1(delta)))
        ).squeeze(-1)
        pt = torch.cat(
            [torch.ones((B, 1), device=x.device, dtype=x.dtype), pt_minus_1],
            dim=1,
        )
        bt = pt >= self._boundary_threshold
        return pt, bt

    def _forward_fill(self, routing_weights, selected_experts, bt, B, L):
        top_k = routing_weights.shape[-1]
        rw = routing_weights.view(B, L, top_k)
        se = selected_experts.view(B, L, top_k)

        positions = torch.arange(L, device=bt.device).unsqueeze(0).expand(B, -1)
        boundary_pos = torch.where(bt, positions, torch.zeros_like(positions) - 1)
        last_boundary, _ = boundary_pos.cummax(dim=1)

        gather_idx = last_boundary.unsqueeze(-1).expand_as(rw)
        rw = torch.gather(rw, 1, gather_idx)
        se = torch.gather(se, 1, gather_idx)
        return rw.reshape(-1, top_k), se.reshape(-1, top_k)

    def _compute_ratio_loss(self, p_t, b_t):
        N = 1.0 + F.softplus(self._log_N) if self._learnable_N else self._N
        mask = self._padding_mask
        F_val = b_t.float().detach()
        G = p_t
        if mask is not None:
            mask = mask.bool()
            F_val = F_val[mask].mean()
            G = G[mask].mean()
            pt_masked = p_t[mask]
        else:
            F_val = F_val.mean()
            G = G.mean()
            pt_masked = p_t
        ratio_loss = (
            (1 - F_val) * (1 - G) + F_val * G * (N - 1)
        ) * N / (N - 1)
        pt_f32 = pt_masked.float().clamp(1e-6, 1 - 1e-6)
        entropy = -(pt_f32 * pt_f32.log() + (1 - pt_f32) * (1 - pt_f32).log()).mean()
        entropy_penalty = F.relu(entropy - self._entropy_threshold)
        return self._ratio_loss_alpha * ratio_loss + self._entropy_alpha * entropy_penalty

    def forward(self, hidden_states):
        B, L, D = hidden_states.shape

        pt, bt = self._compute_boundary(hidden_states)

        flat = hidden_states.view(-1, D)

        gate = getattr(self.moe_block, self._gate_attr)
        _, routing_weights, selected_experts = gate(flat)

        routing_weights, selected_experts = self._forward_fill(
            routing_weights, selected_experts, bt, B, L,
        )

        expert_output = self.moe_block.experts(flat, selected_experts, routing_weights)

        if hasattr(self.moe_block, 'shared_expert') and self.moe_block.shared_expert is not None:
            shared_output = self.moe_block.shared_expert(flat)
            if hasattr(self.moe_block, 'shared_expert_gate') and self.moe_block.shared_expert_gate is not None:
                shared_output = F.sigmoid(self.moe_block.shared_expert_gate(flat)) * shared_output
            expert_output = expert_output + shared_output

        output = expert_output.view(B, L, D)

        if self.training:
            self._ratio_loss = self._compute_ratio_loss(pt, bt)

        self._last_pt = pt.detach()
        self._last_bt = bt.detach()
        self._last_top_k_indices = selected_experts.view(B, L, -1).detach()
        self._last_G = pt.detach().mean()
        self._last_F = bt.float().detach().mean()
        pt_f32 = pt.detach().float().clamp(1e-6, 1 - 1e-6)
        self._last_pt_entropy = -(pt_f32 * pt_f32.log() + (1 - pt_f32) * (1 - pt_f32).log()).mean()

        return output


class TemporalWrapMixin:
    """Patches a HuggingFace MoE model to use temporal boundary routing.

    Unlike the full MoEMixin (which replaces FFN layers with new experts),
    this wrapper keeps the original MoE experts and shared expert intact.
    It only adds boundary prediction and forward-fill routing logic."""

    @staticmethod
    def apply(model, config: TemporalWrapConfig):
        decoder_layers = list(TemporalWrapMixin._find_decoder_layers(model))

        def _is_moe_layer(layer):
            has_gate = hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "router")
            return has_gate and hasattr(layer.mlp, "experts")

        moe_indices = [
            i for i, layer in enumerate(decoder_layers)
            if _is_moe_layer(layer)
        ]

        num_moe = len(moe_indices)
        assert num_moe > 0, "No MoE layers found in model"

        N_list = config.ratio_loss_N
        if len(N_list) == 1:
            N_list = N_list * num_moe
        assert len(N_list) == num_moe, (
            f"ratio_loss_N has {len(N_list)} values but model has {num_moe} MoE layers"
        )

        moe_layers = []
        for list_idx, layer_idx in enumerate(moe_indices):
            layer = decoder_layers[layer_idx]
            wrapper = TemporalMoEWrapper(layer.mlp, config, N_list[list_idx])
            layer.mlp = wrapper
            moe_layers.append(wrapper)

        model._moe_layers = moe_layers
        model._padding_mask = None

        def _capture_mask(_module, _args, kwargs):
            mask = kwargs.get("attention_mask", None)
            for m in model._moe_layers:
                m._padding_mask = mask
        model.register_forward_pre_hook(_capture_mask, with_kwargs=True)

        def _get_moe_loss():
            return sum(m.ratio_loss for m in model._moe_layers)
        model.get_moe_loss = _get_moe_loss

        def _inject_moe_loss(_module, _input, output):
            if hasattr(output, "loss") and output.loss is not None:
                output.loss = output.loss + _get_moe_loss()
            return output
        model.register_forward_hook(_inject_moe_loss)

        gate = getattr(moe_layers[0].moe_block, moe_layers[0]._gate_attr)
        print(
            f"[TemporalWrapMixin] Wrapped {num_moe} MoE layers "
            f"(experts={gate.num_experts}, top_k={gate.top_k})"
        )
        return model

    @staticmethod
    def _find_decoder_layers(model):
        for module in model.modules():
            if hasattr(module, "mlp") and (
                hasattr(module, "self_attn") or hasattr(module, "attn")
            ):
                yield module
