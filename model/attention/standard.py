import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from model.attention.rope import apply_rotary_emb


class StandardMHA(nn.Module):
    """Standard multi-head attention with RoPE/CoPE support and injectable softmax."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_cope: bool = False,
        cope_module: Optional[nn.Module] = None,
        softmax_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_cope = use_cope
        self.cope = cope_module
        self.softmax_fn = softmax_fn  # If None, use F.softmax

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _softmax(self, attn_weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.softmax_fn is not None:
            return self.softmax_fn(attn_weights, dim=dim)
        return F.softmax(attn_weights, dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            freqs_cis: precomputed RoPE frequencies
            mask: causal mask (batch, 1, seq_len, kv_len) or None
            position_ids: explicit positions for MoD (batch, seq_len)
            kv_cache: optional (cached_k, cached_v) for generation

        Returns:
            (output, aux_losses_dict, new_kv_cache)
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Apply positional encoding
        if not self.use_cope:
            q = apply_rotary_emb(q, freqs_cis, position_ids)
            k = apply_rotary_emb(k, freqs_cis, position_ids)

        # Transpose to (B, n_heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV cache for generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # CoPE position bias
        if self.use_cope and self.cope is not None:
            attn_weights = self.cope(q, attn_weights)

        # Apply causal mask
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = self._softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, n_heads, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.o_proj(out)

        return out, {}, new_kv_cache
