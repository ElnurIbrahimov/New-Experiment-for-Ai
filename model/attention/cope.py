import torch
import torch.nn as nn
import torch.nn.functional as F


class CoPE(nn.Module):
    """Contextual Position Encoding (CoPE).

    Computes fractional positions via gated cumsum and interpolates
    learned position embeddings to add as bias to attention logits.
    Mutually exclusive with RoPE.
    """

    def __init__(self, n_heads: int, head_dim: int, npos_max: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.npos_max = npos_max

        # Learnable position embedding table: (npos_max, head_dim)
        self.pos_emb = nn.Embedding(npos_max, head_dim)

    def forward(self, q: torch.Tensor, attn_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (batch, n_heads, seq_len, head_dim) -- query vectors
            attn_logits: (batch, n_heads, seq_len, kv_len) -- raw attention scores

        Returns:
            attn_logits with position bias added
        """
        # Compute gates from attention logits
        gates = torch.sigmoid(attn_logits)  # (B, H, S, KV)

        # Cumsum right-to-left: flip, cumsum, flip back
        # This gives fractional positions: how many tokens each query "attends to"
        pos = gates.flip(-1).cumsum(-1).flip(-1)  # (B, H, S, KV)

        # Clamp to valid embedding range
        pos = pos.clamp(0, self.npos_max - 1.0)

        # Interpolate position embeddings
        pos_floor = pos.long().clamp(0, self.npos_max - 1)
        pos_ceil = (pos_floor + 1).clamp(0, self.npos_max - 1)
        frac = pos - pos_floor.float()  # fractional part

        # Gather embeddings: (npos_max, head_dim)
        emb_floor = self.pos_emb(pos_floor)  # (B, H, S, KV, head_dim)
        emb_ceil = self.pos_emb(pos_ceil)

        # Interpolate
        pos_emb = emb_floor + frac.unsqueeze(-1) * (emb_ceil - emb_floor)

        # Compute position logits: dot product of q with interpolated pos embeddings
        # q: (B, H, S, D) -> (B, H, S, 1, D)
        # pos_emb: (B, H, S, KV, D)
        pos_logits = (q.unsqueeze(-2) * pos_emb).sum(-1)  # (B, H, S, KV)

        return attn_logits + pos_logits
