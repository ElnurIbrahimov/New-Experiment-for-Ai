#!/usr/bin/env python3
"""Main training script for NovelFormer."""

import argparse
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.config import ModelConfig, TrainConfig, load_config
from model.transformer import Transformer
from model.norm.ngpt import l2_normalize
from data.dataset import get_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.distributed import setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process
from utils.lr_schedule import CosineWarmupScheduler
from utils.logging import MetricsLogger


def get_param_groups(model: Transformer, train_config: TrainConfig, model_config: ModelConfig):
    """Separate parameters into decay and no-decay groups."""
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay for biases, norms, embeddings, and all params when nGPT
        if model_config.use_ngpt:
            no_decay.add(name)
        elif name.endswith(".weight") and ("norm" not in name and "emb" not in name):
            decay.add(name)
        else:
            no_decay.add(name)

    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    groups = [
        {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]
    # Filter empty groups
    groups = [g for g in groups if len(g["params"]) > 0]
    return groups


def ngpt_normalize_weights(model: Transformer):
    """Post-optimizer-step: normalize all weight matrices for nGPT."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                param.data = l2_normalize(param.data)


@torch.no_grad()
def evaluate(model: Transformer, eval_loader, eval_steps: int, device: torch.device):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(eval_loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(model_config: ModelConfig, train_config: TrainConfig):
    # DDP setup
    if train_config.ddp:
        rank, world_size, local_rank = setup_ddp(train_config.backend)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_process = is_main_process() if train_config.ddp else True

    # Logging
    logger = MetricsLogger(
        log_dir=train_config.log_dir,
        experiment_name=train_config.experiment_name or "run",
        enabled=main_process,
    )

    if main_process:
        print(f"Model config: {model_config}")
        print(f"Train config: {train_config}")
        print(f"Device: {device}")

    # Dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map.get(train_config.dtype, torch.bfloat16)

    # Model
    model = Transformer(model_config).to(device)
    if main_process:
        n_params = model.get_num_params()
        print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    if train_config.compile:
        model = torch.compile(model)

    if train_config.ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    raw_model = model.module if train_config.ddp else model
    param_groups = get_param_groups(raw_model, train_config, model_config)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        fused=torch.cuda.is_available(),
    )

    # Scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        min_lr=train_config.min_lr,
    )

    # Resume
    start_step = 0
    if train_config.resume_from:
        start_step = load_checkpoint(
            train_config.resume_from, model, optimizer, scheduler, device=str(device)
        )
        if main_process:
            print(f"Resumed from step {start_step}")

    # Data
    train_loader = get_dataloader(
        seq_len=train_config.seq_len,
        batch_size=train_config.batch_size,
        split="train",
        dataset_name=train_config.dataset_name,
        dataset_subset=train_config.dataset_subset,
        tokenizer_name=train_config.tokenizer,
        rank=rank,
        world_size=world_size,
    )
    eval_loader = get_dataloader(
        seq_len=train_config.seq_len,
        batch_size=train_config.batch_size,
        split="train",  # Use same split, different shard for eval
        dataset_name=train_config.dataset_name,
        dataset_subset=train_config.dataset_subset,
        tokenizer_name=train_config.tokenizer,
        rank=rank,
        world_size=world_size,
    )

    # Training loop
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    train_iter = iter(train_loader)

    grad_accum_steps = train_config.gradient_accumulation_steps
    tokens_per_step = train_config.batch_size * train_config.seq_len * grad_accum_steps * world_size

    if main_process:
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Starting training from step {start_step}...")

    for step in range(start_step, train_config.max_steps):
        logger.start_step()
        optimizer.zero_grad()
        total_loss = 0.0
        total_aux = {}

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logger.add_tokens(x.numel())

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits, aux_losses_list, _ = model(x)
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                # Aggregate aux losses (only scalar tensors with gradients)
                aux_sum = 0.0
                for aux_dict in aux_losses_list:
                    for k, v in aux_dict.items():
                        if isinstance(v, torch.Tensor) and v.requires_grad and v.dim() == 0:
                            aux_sum = aux_sum + v
                            total_aux[k] = total_aux.get(k, 0.0) + v.item() / grad_accum_steps
                        elif isinstance(v, torch.Tensor) and v.dim() == 0:
                            total_aux[k] = total_aux.get(k, 0.0) + v.item() / grad_accum_steps

                loss = (ce_loss + aux_sum) / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += ce_loss.item() / grad_accum_steps

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # nGPT: normalize weights after optimizer step
        if model_config.use_ngpt:
            ngpt_normalize_weights(raw_model)

        # Logging
        if main_process and (step + 1) % train_config.log_interval == 0:
            logger.log_step(
                step=step + 1,
                loss=total_loss,
                lr=scheduler.get_last_lr()[0],
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                aux_losses=total_aux if total_aux else None,
            )

        # Eval
        if (step + 1) % train_config.eval_interval == 0:
            eval_loss = evaluate(raw_model, eval_loader, train_config.eval_steps, device)
            if main_process:
                logger.log_eval(step + 1, eval_loss)

        # Checkpoint
        if main_process and (step + 1) % train_config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, step + 1,
                model_config, train_config,
                train_config.checkpoint_dir,
                prefix=train_config.experiment_name or "ckpt",
            )

    # Final save
    if main_process:
        save_checkpoint(
            model, optimizer, scheduler, train_config.max_steps,
            model_config, train_config,
            train_config.checkpoint_dir,
            prefix=train_config.experiment_name or "ckpt",
        )

    logger.close()
    if train_config.ddp:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train NovelFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    model_config, train_config = load_config(args.config)
    if args.resume:
        train_config.resume_from = args.resume

    # Auto-detect DDP
    if "RANK" in os.environ:
        train_config.ddp = True

    train(model_config, train_config)


if __name__ == "__main__":
    main()
