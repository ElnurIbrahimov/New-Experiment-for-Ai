import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MoDWrapper(nn.Module):
    """Mixture-of-Depths: route top-k tokens through the inner block,
    bypass the rest with a residual connection.
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        capacity_ratio: float = 0.5,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.block = block
        self.d_model = d_model
        self.capacity_ratio = capacity_ratio
        self.aux_loss_weight = aux_loss_weight

        # Router: projects to scalar routing score
        self.router = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, dict, Optional[tuple]]:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        k = max(1, int(S * self.capacity_ratio))

        # Router scores
        router_logits = self.router(x).squeeze(-1)  # (B, S)
        router_weights = torch.sigmoid(router_logits)

        # Top-k selection per batch
        topk_vals, topk_indices = torch.topk(router_weights, k, dim=-1)  # (B, k)
        # Sort indices so selected tokens maintain causal ordering
        topk_indices, sort_order = topk_indices.sort(dim=-1)
        topk_vals = topk_vals.gather(-1, sort_order)

        # Gather selected tokens
        # (B, k, D)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_x = x.gather(1, expanded_indices)

        # Position IDs for selected tokens (for correct RoPE)
        if position_ids is not None:
            selected_pos = position_ids.gather(1, topk_indices)
        else:
            # Default sequential positions, select the ones we're routing
            default_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            selected_pos = default_pos.gather(1, topk_indices)

        # Build causal mask for selected tokens
        # selected tokens attend to all tokens with lower original position
        # For simplicity, we build a (B, 1, k, k) mask among selected tokens
        selected_mask = None
        if k > 1:
            # selected_pos: (B, k) -- positions in original sequence
            pos_i = selected_pos.unsqueeze(-1)  # (B, k, 1)
            pos_j = selected_pos.unsqueeze(-2)  # (B, 1, k)
            selected_mask = torch.where(
                pos_i >= pos_j,
                torch.tensor(0.0, device=x.device, dtype=x.dtype),
                torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
            )
            selected_mask = selected_mask.unsqueeze(1)  # (B, 1, k, k)

        # Run block on selected tokens
        block_out, block_aux, new_kv = self.block(
            selected_x, freqs_cis, mask=selected_mask, position_ids=selected_pos, kv_cache=kv_cache
        )

        # Weight by router scores
        block_out = block_out * topk_vals.unsqueeze(-1)

        # Scatter back: output = residual for skipped, block_out for selected
        output = x.clone()
        output.scatter_(1, expanded_indices, block_out)

        # Auxiliary loss: BCE between router logits and top-k binary targets
        with torch.no_grad():
            targets = torch.zeros(B, S, device=x.device)
            targets.scatter_(1, topk_indices, 1.0)
        aux_loss = F.binary_cross_entropy_with_logits(router_logits, targets) * self.aux_loss_weight

        routed_frac = k / S

        aux = {
            "mod_aux_loss": aux_loss,
            "mod_routed_frac": torch.tensor(routed_frac),
            **block_aux,
        }

        return output, aux, new_kv
