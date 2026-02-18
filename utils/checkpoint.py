import os
import torch
import torch.nn as nn
from typing import Optional
from model.config import ModelConfig, TrainConfig, save_config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_dir: str,
    prefix: str = "ckpt",
):
    """Save a training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{prefix}_step{step}.pt")

    # Get model state dict (unwrap DDP if needed)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(checkpoint, path)

    # Also save config alongside
    config_path = os.path.join(checkpoint_dir, f"{prefix}_step{step}_config.yaml")
    save_config(model_config, train_config, config_path)

    # Symlink to latest
    latest_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pt")
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.abspath(path), latest_path)
    except OSError:
        import shutil
        if os.path.exists(latest_path):
            os.remove(latest_path)
        shutil.copy2(path, latest_path)

    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> int:
    """Load a checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Unwrap DDP if needed
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

    return checkpoint["step"]
