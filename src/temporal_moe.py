from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    ratio_loss_N: list[int] = field(default_factory=lambda: [3])
    ratio_loss_alpha: float = 0.3
    entropy_threshold: float = 0.1
    entropy_alpha: float = 1.0
    expert_dim: int | None = None
    num_copies: int = 0
    learnable_N: bool = False

    def __post_init__(self):
        if isinstance(self.ratio_loss_N, int):
            self.ratio_loss_N = [self.ratio_loss_N]


class ChunkingRouter(nn.Module):
    """Boundary-aware router: learns a termination function (option-critic style)
    over delta states to segment tokens, then forward-fills top-k routing
    decisions across each segment."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)
        self.term_proj1 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.term_proj2 = nn.Linear(hidden_dim, 1, dtype=dtype)
        self._boundary_threshold = 0.5

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        B, L, D = x.shape

        # option-critic style termination over delta states
        delta = x[:, 1:, :] - x[:, :-1, :]                 # B, L-1, D
        pt_minus_1 = torch.sigmoid(
            self.term_proj2(F.relu(self.term_proj1(delta)))
        ).squeeze(-1)                                        # B, L-1

        pt = torch.cat([torch.ones((B, 1), device=x.device, dtype=x.dtype), pt_minus_1], dim=1) # B, L
        bt = (pt >= self._boundary_threshold) # B, L

        # routing segments to experts
        logits = self.gate(x) # B, L, E
        routing_probs = F.softmax(logits, dim=-1) # B, L, E

        # topk sampling of experts per salient token (yet to mask)
        top_k_probs, top_k_indices = routing_probs.topk(self.top_k, dim=-1) # B, L, top_k
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # forward-fill: propagate routing from bt=1 positions to subsequent bt=0 positions.
        # use cummax on boundary positions to find the last bt=1 index for each token,
        # then gather routing decisions from those boundary positions.
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1) # B, L
        boundary_pos = torch.where(bt, positions, torch.zeros_like(positions) - 1) # B, L
        last_boundary, _ = boundary_pos.cummax(dim=1) # B, L

        gather_idx = last_boundary.unsqueeze(-1).expand_as(top_k_probs) # B, L, top_k
        top_k_probs = torch.gather(top_k_probs, 1, gather_idx)
        top_k_indices = torch.gather(top_k_indices, 1, gather_idx)

        return top_k_probs, top_k_indices, pt, bt


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=None, device=None):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class SegmentedExperts(nn.Module):
    """Virtual experts carved from concatenated FFN weight matrices."""

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


class ChunkingMoELayer(nn.Module):
    """
        Implements MoE with dynamic chunking from H-Nets, and the ratio loss
    """
    def __init__(self, original_ffn: nn.Module, config: MoEConfig, hidden_dim: int, intermediate_size: int, N:int):
        super().__init__()
        self.config = config
        device = next(original_ffn.parameters()).device
        dtype = next(original_ffn.parameters()).dtype
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
        self.router = ChunkingRouter(hidden_dim, config.num_experts, config.top_k, dtype=dtype).to(device)
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
    def ratio_loss(self) -> torch.Tensor:
        return self._ratio_loss

    def _compute_ratio_loss(self, p_t, b_t) -> torch.Tensor:
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
            (1 - F_val) * (1 - G) +
            F_val * G * (N - 1)
        ) * N / (N - 1)
        pt_f32 = pt_masked.float().clamp(1e-6, 1 - 1e-6)
        entropy = F.binary_cross_entropy(pt_f32, pt_f32)
        entropy_penalty = F.relu(entropy - self._entropy_threshold)
        return self._ratio_loss_alpha * ratio_loss + self._entropy_alpha * entropy_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape                          # (batch, seq_len, hidden)
        B, L, D = orig_shape

        top_k_probs, top_k_indices, pt, bt = self.router(x)
        if self.training:
            self._ratio_loss = self._compute_ratio_loss(pt, bt)
        self._last_G = pt.detach().mean()
        self._last_F = bt.float().detach().mean()
        pt_f32 = pt.detach().float().clamp(1e-6, 1 - 1e-6)
        self._last_pt_entropy = F.binary_cross_entropy(pt_f32, pt_f32)
        self._last_pt = pt.detach()
        self._last_bt = bt.detach()
        self._last_top_k_indices = top_k_indices.detach()

        # batched expert dispatch: sort tokens by expert, run each expert on
        # a contiguous slice, then scatter results back in one pass.
        flat = x.reshape(-1, D)                                      # (B*L, D)
        N = flat.shape[0]
        top_k_probs = top_k_probs.reshape(-1, self.config.top_k)    # (N, top_k)
        top_k_indices = top_k_indices.reshape(-1, self.config.top_k) # (N, top_k)

        token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand_as(top_k_indices)
        flat_expert_ids = top_k_indices.reshape(-1)                  # (N*top_k,)
        flat_token_idx = token_idx.reshape(-1)                       # (N*top_k,)
        flat_weights = top_k_probs.reshape(-1)                       # (N*top_k,)

        sort_order = torch.argsort(flat_expert_ids, stable=True)
        flat_expert_ids = flat_expert_ids[sort_order]
        flat_token_idx = flat_token_idx[sort_order]
        flat_weights = flat_weights[sort_order]

        expert_tokens = flat[flat_token_idx]                         # (N*top_k, D)
        expert_counts = torch.bincount(flat_expert_ids, minlength=self.config.num_experts)

        out = torch.zeros_like(flat)
        if self._segmented:
            offset = 0
            for e_idx, count in enumerate(expert_counts.tolist()):
                if count == 0:
                    continue
                inp = expert_tokens[offset:offset + count]
                w = flat_weights[offset:offset + count].unsqueeze(-1)
                idx = flat_token_idx[offset:offset + count]
                out.index_add_(0, idx, w * self.experts.forward_expert(inp, e_idx))
                offset += count
        else:
            dummy = flat[:1].detach()
            offset = 0
            for e_idx, count in enumerate(expert_counts.tolist()):
                if count == 0:
                    out = out + 0.0 * self.experts[e_idx](dummy).sum()
                else:
                    inp = expert_tokens[offset:offset + count]
                    w = flat_weights[offset:offset + count].unsqueeze(-1)
                    idx = flat_token_idx[offset:offset + count]
                    out.index_add_(0, idx, w * self.experts[e_idx](inp))
                    offset += count

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

        num_layers = len(decoder_layers)
        N_list = config.ratio_loss_N
        if len(N_list) == 1:
            N_list = N_list * num_layers
        assert len(N_list) == num_layers, (
            f"ratio_loss_N has {len(N_list)} values but model has {num_layers} layers"
        )

        moe_layers = []
        for idx, layer in enumerate(decoder_layers):
            original_mlp = layer.mlp
            moe = ChunkingMoELayer(original_mlp, config, hidden_dim, intermediate_size, N_list[idx])
            if config.num_copies > 0:
                _init_segmented_from_ffn(moe.experts, original_mlp, config.num_copies)
            layer.mlp = moe
            moe_layers.append(moe)

        model._moe_layers = moe_layers
        model._padding_mask = None

        def _capture_mask(_module, _args, kwargs):
            mask = kwargs.get('attention_mask', None)
            for m in model._moe_layers:
                m._padding_mask = mask
        model.register_forward_pre_hook(_capture_mask, with_kwargs=True)

        def _get_moe_loss():
            return sum(m.ratio_loss for m in model._moe_layers)
        model.get_moe_loss = _get_moe_loss

        def _inject_moe_loss(_module, _input, output):
            if hasattr(output, 'loss') and output.loss is not None:
                output.loss = output.loss + _get_moe_loss()
            return output
        model.register_forward_hook(_inject_moe_loss)

        label = f"experts={config.num_experts}, top_k={config.top_k}"
        if config.num_copies > 0:
            label += f", segmented ({config.num_copies} copies, expert_dim={config.expert_dim})"
        print(f"[MoEMixin] Converted {len(moe_layers)} FFN layers -> MoE ({label})")
        return model

    @staticmethod
    def _find_decoder_layers(model: nn.Module):
        """Locate transformer decoder layers by searching for .mlp attribute."""
        for module in model.modules():
            if hasattr(module, 'mlp') and (hasattr(module, 'self_attn') or hasattr(module, 'attn')):
                yield module
