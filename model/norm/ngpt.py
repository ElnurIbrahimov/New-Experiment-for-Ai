import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along given dimension."""
    return F.normalize(x.float(), p=2, dim=dim, eps=eps).type_as(x)


class NormLinear(nn.Module):
    """Linear layer that normalizes its weights at each forward pass (nGPT).

    Ensures weight matrix rows lie on the unit hypersphere.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize weight rows to unit norm at each forward pass
        w = l2_normalize(self.weight, dim=-1)
        out = F.linear(x, w, self.bias)
        return out
