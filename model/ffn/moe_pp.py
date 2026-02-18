import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model.ffn.swiglu import SwiGLU


class MoEPP(nn.Module):
    """MoE++ with zero-computation, copy, and constant experts.

    Expert types:
    - FFN experts: standard SwiGLU
    - Zero experts: output zeros (no compute)
    - Copy experts: identity / passthrough
    - Const experts: blend with a learnable vector
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        n_zero_experts: int = 2,
        n_copy_experts: int = 1,
        n_const_experts: int = 1,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_ffn_experts = n_experts
        self.n_zero_experts = n_zero_experts
        self.n_copy_experts = n_copy_experts
        self.n_const_experts = n_const_experts
        self.total_experts = n_experts + n_zero_experts + n_copy_experts + n_const_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # FFN experts
        self.ffn_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout=dropout) for _ in range(n_experts)
        ])

        # Constant expert vectors (learnable)
        if n_const_experts > 0:
            self.const_vectors = nn.Parameter(torch.randn(n_const_experts, d_model) * 0.02)

        # Router
        self.router = nn.Linear(d_model, self.total_experts, bias=False)

        # Gating residual: condition on previous layer's gate logits
        self.gate_residual_weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        prev_gate_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, d_model)
            prev_gate_logits: optional gate logits from previous MoE layer

        Returns:
            (output, aux_dict)
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (B*S, D)
        N = x_flat.shape[0]

        # Router logits
        gate_logits = self.router(x_flat)  # (N, total_experts)

        # Gating residual
        if prev_gate_logits is not None and prev_gate_logits.shape == gate_logits.shape:
            gate_logits = gate_logits + torch.sigmoid(self.gate_residual_weight) * prev_gate_logits

        # Top-k routing
        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (N, top_k)

        # Normalize selected scores
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_ids = topk_indices[:, k_idx]  # (N,)
            weights = topk_scores[:, k_idx]  # (N,)

            for expert_idx in range(self.total_experts):
                token_mask = (expert_ids == expert_idx)
                if not token_mask.any():
                    continue

                tokens = x_flat[token_mask]  # (n_selected, D)

                if expert_idx < self.n_ffn_experts:
                    # FFN expert
                    expert_out, _ = self.ffn_experts[expert_idx](tokens)
                elif expert_idx < self.n_ffn_experts + self.n_zero_experts:
                    # Zero expert: output zeros
                    expert_out = torch.zeros_like(tokens)
                elif expert_idx < self.n_ffn_experts + self.n_zero_experts + self.n_copy_experts:
                    # Copy expert: identity
                    expert_out = tokens
                else:
                    # Const expert: blend with learnable vector
                    const_idx = expert_idx - (self.n_ffn_experts + self.n_zero_experts + self.n_copy_experts)
                    expert_out = self.const_vectors[const_idx].unsqueeze(0).expand_as(tokens)

                output[token_mask] += weights[token_mask].unsqueeze(-1) * expert_out

        output = output.view(B, S, D)

        # Load balance loss
        # Fraction of tokens routed to each expert
        with torch.no_grad():
            expert_counts = torch.zeros(self.total_experts, device=x.device)
            for k_idx in range(self.top_k):
                for e_idx in range(self.total_experts):
                    expert_counts[e_idx] += (topk_indices[:, k_idx] == e_idx).float().sum()
            expert_frac = expert_counts / (N * self.top_k)

        # Mean gate probability per expert
        mean_gate = gate_scores.mean(dim=0)  # (total_experts,)

        # Standard load balance loss
        lb_loss = (expert_frac * mean_gate).sum() * self.total_experts * self.aux_loss_weight

        aux = {
            "moe_aux_loss": lb_loss,
            "moe_gate_logits": gate_logits.view(B, S, -1),  # For gating residual
        }

        return output, aux
