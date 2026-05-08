import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-20

@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2

class ChunkingRouter(nn.Module):
    """Boundary-aware router: segments tokens via cosine distance, then
    forward-fills top-k routing decisions across each segment."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=dtype)
        self.Wq = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self._boundary_threshold = 0.5

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        B, L, D = x.shape

        # project token qk
        q_t = self.Wq(x)[:, 1:, :] # B, L-1, D
        k_t_minus_1 = self.Wk(x)[:, :-1, :] # B, L-1, D

        # boundary segmentation
        qk_dot = (q_t * k_t_minus_1).sum(-1) # B, L-1
        n1 = torch.norm(q_t, p=2, dim=-1) # B, L-1
        n2 = torch.norm(k_t_minus_1, p=2, dim=-1) # B, L-1
        pt_minus_1 = 1/2 * (1 - (qk_dot / (n1 * n2 + EPS))) # B, L-1

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


class ChunkingMoELayer(nn.Module):
    """
        Implements MoE with dynamic chunking from H-Nets, and the ratio loss
    """
    def __init__(self, original_ffn: nn.Module, config: MoEConfig, hidden_dim: int, N:int):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [copy.deepcopy(original_ffn) for _ in range(config.num_experts)]
        )
        device = next(original_ffn.parameters()).device
        dtype = next(original_ffn.parameters()).dtype
        self.router = ChunkingRouter(hidden_dim, config.num_experts, config.top_k, dtype=dtype).to(device)
        self._ratio_loss_alpha = 0.05 # 0.03
        self._N = N
        self._ratio_loss = torch.tensor(0.0, device=device, dtype=dtype)

    @property
    def ratio_loss(self) -> torch.Tensor:
        return self._ratio_loss

    def _compute_ratio_loss(self, p_t, tau: float = 0.1) -> torch.Tensor:
        N = self._N
        f = torch.sigmoid((p_t - self.router._boundary_threshold) / tau).mean(-1) # B
        g = p_t.mean(-1) # B
        loss = (N/(N-1)) * ((N-1) * f * g + (1-f) * (1-g))
        return self._ratio_loss_alpha * loss.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape                          # (batch, seq_len, hidden)
        B, L, D = orig_shape

        top_k_probs, top_k_indices, pt, bt = self.router(x)
        self._ratio_loss = self._compute_ratio_loss(pt)

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
        # gets decoder layers modules
        decoder_layers = list(MoEMixin._find_decoder_layers(model))

        moe_layers = []
        Ns = len(decoder_layers) * [3]
        for idx, layer in enumerate(decoder_layers):
            original_mlp = layer.mlp
            moe = ChunkingMoELayer(original_mlp, config, hidden_dim, Ns[idx])
            layer.mlp = moe
            moe_layers.append(moe)

        model._moe_layers = moe_layers

        def _get_moe_loss():
            return sum(m.ratio_loss for m in model._moe_layers)
        model.get_moe_loss = _get_moe_loss

        def _inject_moe_loss(_module, _input, output):
            if hasattr(output, 'loss') and output.loss is not None:
                output.loss = output.loss + _get_moe_loss()
            return output
        model.register_forward_hook(_inject_moe_loss)

        print(f"[MoEMixin] Converted {len(moe_layers)} FFN layers -> "
              f"MoE (experts={config.num_experts}, top_k={config.top_k})")
        return model

    @staticmethod
    def _find_decoder_layers(model: nn.Module):
        """Locate transformer decoder layers by searching for .mlp attribute."""
        for module in model.modules():
            if hasattr(module, 'mlp') and hasattr(module, 'self_attn'):
                yield module