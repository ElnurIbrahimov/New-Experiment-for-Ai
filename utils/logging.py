import os
import time
import torch
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class MetricsLogger:
    """TensorBoard + console metrics logger."""

    def __init__(self, log_dir: str, experiment_name: str, enabled: bool = True):
        self.enabled = enabled
        self.experiment_name = experiment_name
        if enabled and SummaryWriter is not None:
            full_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(full_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=full_dir)
        else:
            self.writer = None

        self._step_start = None
        self._tokens_processed = 0

    def start_step(self):
        self._step_start = time.time()
        self._tokens_processed = 0

    def add_tokens(self, n: int):
        self._tokens_processed += n

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        aux_losses: Optional[dict] = None,
    ):
        if not self.enabled:
            return

        elapsed = time.time() - self._step_start if self._step_start else 0
        tokens_per_sec = self._tokens_processed / max(elapsed, 1e-6)
        perplexity = min(torch.exp(torch.tensor(loss)).item(), 1e6)

        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/perplexity", perplexity, step)
            self.writer.add_scalar("train/lr", lr, step)
            self.writer.add_scalar("train/grad_norm", grad_norm, step)
            self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                self.writer.add_scalar("system/peak_vram_gb", peak_vram, step)

            if aux_losses:
                for k, v in aux_losses.items():
                    self.writer.add_scalar(f"aux/{k}", v, step)

        print(
            f"step {step:6d} | loss {loss:.4f} | ppl {perplexity:.1f} | "
            f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
            f"tok/s {tokens_per_sec:.0f}"
        )

    def log_eval(self, step: int, eval_loss: float):
        if not self.enabled:
            return
        eval_ppl = min(torch.exp(torch.tensor(eval_loss)).item(), 1e6)
        if self.writer is not None:
            self.writer.add_scalar("eval/loss", eval_loss, step)
            self.writer.add_scalar("eval/perplexity", eval_ppl, step)
        print(f"  [eval] step {step:6d} | loss {eval_loss:.4f} | ppl {eval_ppl:.1f}")

    def close(self):
        if self.writer is not None:
            self.writer.close()
