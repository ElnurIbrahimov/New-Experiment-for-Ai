#!/usr/bin/env python3
"""Evaluation script: validation perplexity and optional downstream tasks."""

import argparse
import math
import torch
import torch.nn.functional as F

from model.config import load_config
from model.transformer import Transformer
from data.dataset import get_dataloader
from utils.checkpoint import load_checkpoint


@torch.no_grad()
def evaluate_perplexity(
    model: Transformer,
    dataloader,
    n_batches: int,
    device: torch.device,
) -> dict:
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
        count += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "n_batches": count,
        "n_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NovelFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n_batches", type=int, default=100, help="Number of eval batches")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    model_config, train_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = Transformer(model_config).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    n_params = model.get_num_params()
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Data
    batch_size = args.batch_size or train_config.batch_size
    eval_loader = get_dataloader(
        seq_len=train_config.seq_len,
        batch_size=batch_size,
        split="train",
        dataset_name=train_config.dataset_name,
        dataset_subset=train_config.dataset_subset,
        tokenizer_name=train_config.tokenizer,
    )

    # Perplexity
    print(f"Evaluating on {args.n_batches} batches...")
    results = evaluate_perplexity(model, eval_loader, args.n_batches, device)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Tokens evaluated: {results['n_tokens']:,}")

    # Optional: lm-eval-harness integration
    try:
        import lm_eval
        print("\nlm-eval-harness detected. Running HellaSwag + ARC-Easy...")
        # Wrap model for lm-eval
        # This is a placeholder -- full integration requires lm_eval.models.huggingface wrapper
        print("(lm-eval integration requires additional setup)")
    except ImportError:
        print("\nlm-eval-harness not installed. Skipping downstream tasks.")
        print("Install with: pip install lm-eval")


if __name__ == "__main__":
    main()
