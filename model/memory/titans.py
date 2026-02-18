import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy
import math


class NeuralMemory(nn.Module):
    """Titans Neural Long-Term Memory.

    A small MLP whose weights are updated online via gradient descent
    based on surprise (prediction error). Acts as an associative memory.
    """

    def __init__(self, d_model: int, mem_size: int = 256, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.mem_size = mem_size

        # Memory MLP: maps d_model -> mem_size -> d_model
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(d_model, d_model, bias=False))
        else:
            layers.append(nn.Linear(d_model, mem_size, bias=False))
            layers.append(nn.SiLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(mem_size, mem_size, bias=False))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(mem_size, d_model, bias=False))
        self.memory_mlp = nn.Sequential(*layers)

        # Learnable meta-parameters
        self.theta = nn.Parameter(torch.tensor(0.01))  # Memory learning rate
        self.eta = nn.Parameter(torch.tensor(0.9))  # Momentum
        self.alpha = nn.Parameter(torch.tensor(0.0))  # Forget gate (sigmoid applied)

        # Momentum buffer (not a parameter, persists across forward calls)
        self._momentum_buffer = None

    def _get_memory_params(self):
        """Get the list of memory MLP parameters."""
        return list(self.memory_mlp.parameters())

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve from memory given queries.

        Args:
            queries: (batch, n_tokens, d_model)
        Returns:
            retrieved: (batch, n_tokens, d_model)
        """
        return self.memory_mlp(queries)

    def update(self, keys: torch.Tensor, values: torch.Tensor):
        """Update memory weights via gradient descent on prediction error.

        Args:
            keys: (batch, n_tokens, d_model) -- inputs to predict from
            values: (batch, n_tokens, d_model) -- target outputs
        """
        # Compute surprise: prediction error
        predictions = self.memory_mlp(keys.detach())
        surprise = F.mse_loss(predictions, values.detach(), reduction="mean")

        # Compute gradients w.r.t. memory MLP parameters
        mem_params = self._get_memory_params()
        grads = torch.autograd.grad(
            surprise, mem_params, create_graph=False, allow_unused=True
        )

        # Apply forget gate
        forget = torch.sigmoid(self.alpha)
        lr = torch.sigmoid(self.theta) * 0.1  # Scale to reasonable range
        momentum = torch.sigmoid(self.eta)

        # Update memory parameters with momentum
        with torch.no_grad():
            if self._momentum_buffer is None:
                self._momentum_buffer = [torch.zeros_like(p) for p in mem_params]

            for i, (param, grad) in enumerate(zip(mem_params, grads)):
                if grad is None:
                    continue
                # Forget gate: decay existing weights
                param.data.mul_(forget)
                # Momentum update
                self._momentum_buffer[i] = momentum * self._momentum_buffer[i] + grad
                # Update
                param.data.sub_(lr * self._momentum_buffer[i])


class TitansWrapper(nn.Module):
    """Wraps a transformer block with Titans neural memory.

    Retrieves from memory, prepends persistent + retrieved tokens,
    runs the block on the augmented sequence, trims output.
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        mem_size: int = 256,
        n_persistent_tokens: int = 4,
        chunk_size: int = 128,
        mem_layers: int = 2,
    ):
        super().__init__()
        self.block = block
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.n_persistent = n_persistent_tokens

        # Neural memory module
        self.memory = NeuralMemory(d_model, mem_size=mem_size, n_layers=mem_layers)

        # Persistent memory tokens (learnable)
        self.persistent_tokens = nn.Parameter(torch.randn(n_persistent_tokens, d_model) * 0.02)

        # Projections for memory key/value
        self.mem_key_proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_value_proj = nn.Linear(d_model, d_model, bias=False)

        # Gate for blending memory output
        self.mem_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, dict, Optional[tuple]]:
        """
        Process with memory augmentation.

        Args:
            x: (batch, seq_len, d_model)
        """
        B, S, D = x.shape

        # Retrieve from memory
        mem_queries = x  # Use input as queries
        retrieved = self.memory.retrieve(mem_queries)  # (B, S, D)

        # Persistent tokens: expand to batch
        persistent = self.persistent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_persistent, D)

        # Prepend persistent tokens + blend retrieved context
        gate = torch.sigmoid(self.mem_gate(x))
        x_augmented = x + gate * retrieved

        # Concatenate persistent tokens
        augmented = torch.cat([persistent, x_augmented], dim=1)  # (B, n_persistent + S, D)

        # Build extended causal mask
        total_len = self.n_persistent + S
        ext_mask = torch.full(
            (total_len, total_len), float("-inf"), device=x.device, dtype=x.dtype
        )
        ext_mask = torch.triu(ext_mask, diagonal=1)
        # Persistent tokens can attend to each other and are attended by all
        ext_mask[:self.n_persistent, :self.n_persistent] = 0
        ext_mask = ext_mask.unsqueeze(0).unsqueeze(0)

        # Extended position IDs
        if position_ids is not None:
            # Persistent tokens get positions 0..n_persistent-1
            persistent_pos = torch.arange(self.n_persistent, device=x.device).unsqueeze(0).expand(B, -1)
            ext_position_ids = torch.cat([persistent_pos, position_ids + self.n_persistent], dim=1)
        else:
            ext_position_ids = None

        # Run block on augmented sequence
        block_out, block_aux, new_kv = self.block(
            augmented, freqs_cis, mask=ext_mask, position_ids=ext_position_ids, kv_cache=kv_cache
        )

        # Trim: remove persistent token outputs
        output = block_out[:, self.n_persistent:, :]

        # Update memory with current chunk
        mem_keys = self.mem_key_proj(x.detach())
        mem_values = self.mem_value_proj(x.detach())
        self.memory.update(mem_keys, mem_values)

        aux = {**block_aux}
        return output, aux, new_kv
