import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalableSoftmax(nn.Module):
    """Scalable Softmax (SSMax): learnable per-head scaling based on sequence length.

    softmax((1 + s * log(n)) * logits) where s is a learnable per-head parameter.
    """

    def __init__(self, n_heads: int, head_dim: int = 0):
        super().__init__()
        # Initialize to 0 so scale starts at 1.0 (standard softmax), learns from there
        self.s = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

    def forward(self, attn_weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Args:
            attn_weights: (batch, n_heads, seq_len, kv_len) -- may contain -inf from causal mask
            dim: softmax dimension
        """
        n = attn_weights.size(-1)
        log_n = math.log(max(n, 2))
        scale = 1.0 + self.s * log_n

        # Replace -inf with 0 before scaling to avoid NaN gradients,
        # then restore -inf after scaling
        neg_inf_mask = torch.isinf(attn_weights) & (attn_weights < 0)
        safe_weights = attn_weights.masked_fill(neg_inf_mask, 0.0)
        scaled = safe_weights * scale
        scaled = scaled.masked_fill(neg_inf_mask, float("-inf"))

        return F.softmax(scaled, dim=dim)
