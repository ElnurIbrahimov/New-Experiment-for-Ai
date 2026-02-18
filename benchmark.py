#!/usr/bin/env python3
"""Benchmarking script: throughput, VRAM, parameter breakdown."""

import argparse
import time
import torch
import torch.nn.functional as F

from model.config import load_config
from model.transformer import Transformer


def count_params_by_module(model: Transformer) -> dict:
    """Get parameter count breakdown by top-level module."""
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
    breakdown["total"] = sum(p.numel() for p in model.parameters())
    return breakdown


def benchmark_throughput(
    model: Transformer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """Measure forward + backward throughput in tokens/sec."""
    model.train()
    tokens_per_iter = batch_size * seq_len

    # Warmup
    for _ in range(n_warmup):
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, aux, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            for a in aux:
                for v in a.values():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        loss = loss + v
        loss.backward()
        model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(n_iters):
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, aux, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            for a in aux:
                for v in a.values():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        loss = loss + v
        loss.backward()
        model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = tokens_per_iter * n_iters
    tokens_per_sec = total_tokens / elapsed

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9

    return {
        "tokens_per_sec": tokens_per_sec,
        "elapsed_sec": elapsed,
        "n_iters": n_iters,
        "peak_vram_gb": peak_vram,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NovelFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for benchmarking")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--n_iters", type=int, default=20, help="Number of benchmark iterations")
    args = parser.parse_args()

    model_config, train_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = args.seq_len or model_config.max_seq_len
    dtype = torch.bfloat16 if train_config.dtype == "bfloat16" else torch.float16

    # Build model
    model = Transformer(model_config).to(device)

    # Parameter breakdown
    print("=" * 60)
    print("Parameter Breakdown")
    print("=" * 60)
    breakdown = count_params_by_module(model)
    for name, count in breakdown.items():
        print(f"  {name:30s}: {count:>12,} ({count / 1e6:.2f}M)")
    print()

    # Active vs total params (relevant for MoE++)
    total = breakdown["total"]
    print(f"Total parameters: {total:,} ({total / 1e6:.1f}M)")

    if model_config.use_moe_pp:
        # Estimate active params: only top-k experts active
        active_ratio = model_config.moe_top_k / (
            model_config.moe_n_experts + model_config.moe_n_zero_experts
            + model_config.moe_n_copy_experts + model_config.moe_n_const_experts
        )
        print(f"MoE++ active expert ratio: {active_ratio:.2f}")

    # Throughput benchmark
    print()
    print("=" * 60)
    print(f"Throughput Benchmark (batch={args.batch_size}, seq_len={seq_len})")
    print("=" * 60)

    results = benchmark_throughput(
        model, args.batch_size, seq_len, device, dtype, n_iters=args.n_iters
    )
    print(f"  Tokens/sec:  {results['tokens_per_sec']:>12,.0f}")
    print(f"  Peak VRAM:   {results['peak_vram_gb']:>12.2f} GB")
    print(f"  Time ({results['n_iters']} iters): {results['elapsed_sec']:>8.2f}s")

    if model_config.use_mod:
        effective = results["tokens_per_sec"] / model_config.mod_capacity_ratio
        print(f"  MoD effective tokens/sec (adjusted): {effective:,.0f}")


if __name__ == "__main__":
    main()
