import torch
import torch.nn as nn
from typing import Optional

from model.norm.ngpt import l2_normalize


class TransformerBlock(nn.Module):
    """Standard pre-LN transformer block."""

    def __init__(
        self,
        attn: nn.Module,
        ffn: nn.Module,
        norm_class: type,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, dict, Optional[tuple]]:
        # Attention
        h = self.norm1(x)
        attn_out, attn_aux, new_kv = self.attn(
            h, freqs_cis, mask=mask, position_ids=position_ids, kv_cache=kv_cache
        )
        x = x + self.dropout(attn_out)

        # FFN
        h = self.norm2(x)
        ffn_out, ffn_aux = self.ffn(h)
        x = x + self.dropout(ffn_out)

        aux = {**attn_aux, **ffn_aux}
        return x, aux, new_kv


class NGPTBlock(nn.Module):
    """nGPT block: no layer norms, LERP residual on hypersphere."""

    def __init__(
        self,
        attn: nn.Module,
        ffn: nn.Module,
        d_model: int,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.d_model = d_model

        # Learnable interpolation strengths (per dimension)
        self.alpha_attn = nn.Parameter(torch.ones(d_model) * 0.05)
        self.alpha_ffn = nn.Parameter(torch.ones(d_model) * 0.05)

        # Scaling vectors for QK and value paths
        self.s_qk = nn.Parameter(torch.ones(d_model))
        self.s_uv = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, dict, Optional[tuple]]:
        # x should already be on the unit hypersphere
        # Attention: no norm needed since x is normalized
        attn_out, attn_aux, new_kv = self.attn(
            x, freqs_cis, mask=mask, position_ids=position_ids, kv_cache=kv_cache
        )
        # LERP residual for attention
        alpha_a = torch.sigmoid(self.alpha_attn)
        h_attn = l2_normalize(attn_out)
        x = l2_normalize(x + alpha_a * (h_attn - x))

        # FFN
        ffn_out, ffn_aux = self.ffn(x)
        alpha_f = torch.sigmoid(self.alpha_ffn)
        h_ffn = l2_normalize(ffn_out)
        x = l2_normalize(x + alpha_f * (h_ffn - x))

        aux = {**attn_aux, **ffn_aux}
        return x, aux, new_kv
