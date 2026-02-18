import torch
from einops import rearrange


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE.

    Returns: (max_seq_len, dim//2) complex tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs_cis: (max_seq_len, head_dim//2) complex
        position_ids: (batch, seq_len) optional explicit positions for MoD
    """
    b, s, h, d = x.shape

    # Select frequencies based on position_ids or sequential positions
    if position_ids is not None:
        # position_ids: (batch, seq_len) -> gather from freqs_cis
        freqs = freqs_cis[position_ids]  # (batch, seq_len, head_dim//2)
    else:
        freqs = freqs_cis[:s]  # (seq_len, head_dim//2)
        freqs = freqs.unsqueeze(0)  # (1, seq_len, head_dim//2)

    # Reshape x to complex: (batch, seq_len, n_heads, head_dim//2, 2) -> complex
    x_complex = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))

    # Broadcast freqs over heads; slice to match x dimension (for DiffAttn sub_dim)
    freqs = freqs[..., : d // 2].unsqueeze(2)  # (..., 1, d//2)

    # Apply rotation
    x_rotated = x_complex * freqs.to(x_complex.device)

    # Back to real
    x_out = torch.view_as_real(x_rotated).reshape(b, s, h, d)
    return x_out.type_as(x)
