import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        elif step >= self.max_steps:
            scale = self.min_lr / self.base_lrs[0] if self.base_lrs[0] > 0 else 0
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = self.min_lr / self.base_lrs[0] + (1 - self.min_lr / self.base_lrs[0]) * 0.5 * (
                1 + math.cos(math.pi * progress)
            ) if self.base_lrs[0] > 0 else 0

        return [scale * base_lr for base_lr in self.base_lrs]
