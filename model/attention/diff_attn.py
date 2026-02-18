import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from model.attention.rope import apply_rotary_emb
from model.norm.rmsnorm import RMSNorm


class DiffAttn(nn.Module):
    """Differential Attention: compute two attention maps and subtract.

    out = (attn1 - lambda * attn2) @ V, with SubLN normalization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int = 0,
        dropout: float = 0.0,
        use_cope: bool = False,
        cope_module: Optional[nn.Module] = None,
        softmax_fn: Optional[Callable] = None,
        lambda_init_base: float = 0.8,
        lambda_init_decay: float = 0.6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sub_dim = self.head_dim // 2  # Each sub-attention uses half head_dim
        self.use_cope = use_cope
        self.cope = cope_module
        self.softmax_fn = softmax_fn
        self.layer_idx = layer_idx

        # Q/K projections: split into Q1,Q2,K1,K2 (each sub_dim per head)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # projects to n_heads * head_dim
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Lambda reparameterization: 4 learnable vectors per layer
        self.lambda_q1 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)

        # Lambda init based on layer index
        self.lambda_init = lambda_init_base - lambda_init_decay * math.exp(-0.3 * layer_idx)

        # SubLN: RMSNorm applied per head after subtraction
        self.sub_ln = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)

        # Output scaling
        self.output_scale = 1 - self.lambda_init

    def _compute_lambda(self) -> torch.Tensor:
        """Compute lambda from learnable vectors."""
        return (
            torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
            - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
            + self.lambda_init
        )

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
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Split Q and K into two halves: Q1, Q2, K1, K2
        q1, q2 = q[..., : self.sub_dim], q[..., self.sub_dim :]
        k1, k2 = k[..., : self.sub_dim], k[..., self.sub_dim :]

        # Apply RoPE to all 4 sub-projections (or skip if CoPE)
        if not self.use_cope:
            # RoPE expects (B, S, n_heads, dim) -- sub_dim needs matching freqs
            q1 = apply_rotary_emb(q1, freqs_cis, position_ids)
            q2 = apply_rotary_emb(q2, freqs_cis, position_ids)
            k1 = apply_rotary_emb(k1, freqs_cis, position_ids)
            k2 = apply_rotary_emb(k2, freqs_cis, position_ids)

        # Transpose to (B, n_heads, S, dim)
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # cached_k contains concatenated k1,k2
            cached_k1, cached_k2 = cached_k.chunk(2, dim=-1)
            k1 = torch.cat([cached_k1, k1], dim=2)
            k2 = torch.cat([cached_k2, k2], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (torch.cat([k1, k2], dim=-1), v)

        scale = 1.0 / math.sqrt(self.sub_dim)

        # Attention map 1
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        # Attention map 2
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # CoPE position bias (applied to both attention maps)
        if self.use_cope and self.cope is not None:
            attn1 = self.cope(q1, attn1)
            attn2 = self.cope(q2, attn2)

        # Causal mask
        if mask is not None:
            attn1 = attn1 + mask
            attn2 = attn2 + mask

        attn1 = self._softmax(attn1, dim=-1)
        attn2 = self._softmax(attn2, dim=-1)
        attn1 = self.dropout(attn1)
        attn2 = self.dropout(attn2)

        # Differential attention
        lam = self._compute_lambda()
        # Both attend to the same V
        out1 = torch.matmul(attn1, v)  # (B, n_heads, S, head_dim)
        out2 = torch.matmul(attn2, v)
        out = out1 - lam * out2

        # SubLN per head
        # Reshape: (B, n_heads, S, head_dim) -> (B*n_heads*S, head_dim) -> norm -> reshape
        B_n, H_n, S_n, D_n = out.shape
        out = self.sub_ln(out.reshape(-1, D_n)).reshape(B_n, H_n, S_n, D_n)

        # Scale by (1 - lambda_init)
        out = out * self.output_scale

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.o_proj(out)

        return out, {}, new_kv_cache
