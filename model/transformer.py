import torch
import torch.nn as nn
from typing import Optional

from model.config import ModelConfig
from model.norm.rmsnorm import RMSNorm
from model.norm.ngpt import NormLinear, l2_normalize
from model.attention.rope import precompute_freqs_cis
from model.attention.standard import StandardMHA
from model.attention.diff_attn import DiffAttn
from model.attention.cope import CoPE
from model.attention.ssmax import ScalableSoftmax
from model.ffn.swiglu import SwiGLU
from model.ffn.moe_pp import MoEPP
from model.block import TransformerBlock, NGPTBlock
from model.routing.mod import MoDWrapper
from model.memory.titans import TitansWrapper


def build_layer(config: ModelConfig, layer_idx: int, freqs_cis: torch.Tensor) -> nn.Module:
    """Factory: build a single transformer layer based on config toggles."""
    d_model = config.d_model
    n_heads = config.n_heads
    head_dim = d_model // n_heads

    # --- Softmax function ---
    softmax_fn = None
    if config.use_ssmax:
        softmax_fn = ScalableSoftmax(n_heads=n_heads, head_dim=head_dim)

    # --- CoPE module ---
    cope_module = None
    if config.use_cope:
        # DiffAttn uses sub_dim queries (head_dim//2), so CoPE must match
        cope_dim = head_dim // 2 if config.use_diff_attn else head_dim
        cope_module = CoPE(n_heads=n_heads, head_dim=cope_dim, npos_max=config.cope_npos_max)

    # --- Attention ---
    if config.use_diff_attn:
        attn = DiffAttn(
            d_model=d_model,
            n_heads=n_heads,
            layer_idx=layer_idx,
            dropout=config.dropout,
            use_cope=config.use_cope,
            cope_module=cope_module,
            softmax_fn=softmax_fn,
            lambda_init_base=config.diff_attn_lambda_init_base,
            lambda_init_decay=config.diff_attn_lambda_init_decay,
        )
    else:
        attn = StandardMHA(
            d_model=d_model,
            n_heads=n_heads,
            dropout=config.dropout,
            use_cope=config.use_cope,
            cope_module=cope_module,
            softmax_fn=softmax_fn,
        )

    # --- FFN ---
    if config.use_moe_pp:
        ffn = MoEPP(
            d_model=d_model,
            d_ff=config.d_ff,
            n_experts=config.moe_n_experts,
            n_zero_experts=config.moe_n_zero_experts,
            n_copy_experts=config.moe_n_copy_experts,
            n_const_experts=config.moe_n_const_experts,
            top_k=config.moe_top_k,
            aux_loss_weight=config.moe_aux_loss_weight,
            dropout=config.dropout,
        )
    else:
        ffn = SwiGLU(d_model=d_model, d_ff=config.d_ff, dropout=config.dropout)

    # --- Block ---
    if config.use_ngpt:
        block = NGPTBlock(
            attn=attn, ffn=ffn, d_model=d_model, layer_idx=layer_idx
        )
    else:
        block = TransformerBlock(
            attn=attn, ffn=ffn, norm_class=RMSNorm, d_model=d_model, dropout=config.dropout
        )

    # --- MoD wrapper (inside Titans) ---
    if config.use_mod:
        block = MoDWrapper(
            block=block,
            d_model=d_model,
            capacity_ratio=config.mod_capacity_ratio,
            aux_loss_weight=config.mod_aux_loss_weight,
        )

    # --- Titans wrapper (outermost) ---
    if config.use_titans and (layer_idx % config.titans_every_n_layers == 0):
        block = TitansWrapper(
            block=block,
            d_model=d_model,
            mem_size=config.titans_mem_size,
            n_persistent_tokens=config.titans_n_persistent_tokens,
            chunk_size=config.titans_chunk_size,
            mem_layers=config.titans_mem_layers,
        )

    return block


class Transformer(nn.Module):
    """Top-level transformer model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no learned positional -- RoPE/CoPE handles positions)
        if config.use_ngpt:
            self.tok_emb = NormLinear(config.vocab_size, config.d_model, bias=False)
        else:
            self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Precompute RoPE frequencies
        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len * 2),
            persistent=False,
        )

        # Build layers
        self.layers = nn.ModuleList([
            build_layer(config, i, self.freqs_cis) for i in range(config.n_layers)
        ])

        # Final norm
        if config.use_ngpt:
            self.final_norm = nn.Identity()  # nGPT: already on hypersphere
        else:
            self.final_norm = RMSNorm(config.d_model)

        # LM head
        if config.use_ngpt:
            self.lm_head = NormLinear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_embeddings and not config.use_ngpt:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> tuple[torch.Tensor, list[dict], Optional[list]]:
        """
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) optional for loss computation
            kv_caches: list of kv cache tuples per layer

        Returns:
            (logits, aux_losses_list, new_kv_caches)
        """
        B, S = input_ids.shape

        if self.config.use_ngpt:
            # NormLinear embedding: one-hot -> project -> normalize
            x = self.tok_emb(
                torch.nn.functional.one_hot(input_ids, self.config.vocab_size).float()
            )
            x = l2_normalize(x)
        else:
            x = self.tok_emb(input_ids)

        # Causal mask
        mask = torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # If using kv_cache, adjust mask for the cached prefix
        if kv_caches is not None and kv_caches[0] is not None:
            cached_len = kv_caches[0][0].shape[2]
            # Only need mask for new tokens attending to all previous
            mask = torch.zeros(1, 1, S, cached_len + S, device=x.device, dtype=x.dtype)
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            mask[:, :, :, cached_len:] = causal

        aux_losses = []
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            x, aux, new_kv = layer(x, self.freqs_cis, mask=mask, kv_cache=kv)
            aux_losses.append(aux)
            new_kv_caches.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # nGPT: scale logits
        if self.config.use_ngpt:
            logits = logits * (self.config.d_model ** 0.5)

        return logits, aux_losses, new_kv_caches

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_ngpt:
            n_params -= self.tok_emb.weight.numel()
        return n_params
