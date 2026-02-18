#!/bin/bash
# RunPod environment setup script for NovelFormer
set -e

echo "=== NovelFormer RunPod Setup ==="

# System info
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install datasets>=2.14.0 tiktoken>=0.5.0 pyyaml>=6.0 tensorboard>=2.14.0 einops>=0.7.0

# Optional: lm-eval for downstream evaluation
pip install lm-eval 2>/dev/null || echo "lm-eval install skipped (optional)"

# Verify imports
echo ""
echo "Verifying installations..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import datasets; print(f'datasets: {datasets.__version__}')
import tiktoken; print('tiktoken: OK')
import yaml; print('pyyaml: OK')
import tensorboard; print(f'tensorboard: {tensorboard.__version__}')
import einops; print(f'einops: {einops.__version__}')
"

# Pre-download tokenizer
echo ""
echo "Pre-downloading tokenizer..."
python -c "import tiktoken; tiktoken.get_encoding('gpt2'); print('Tokenizer cached.')"

# Verify model builds
echo ""
echo "Verifying model construction..."
python -c "
import sys; sys.path.insert(0, '.')
from model.config import load_config
from model.transformer import Transformer
mc, tc = load_config('configs/base.yaml')
m = Transformer(mc)
n = m.get_num_params()
print(f'Baseline model: {n:,} params ({n/1e6:.1f}M)')
print('Model construction OK')
"

echo ""
echo "=== Setup complete ==="
echo "To train: python train.py --config configs/base.yaml"
echo "With DDP: torchrun --nproc_per_node=N train.py --config configs/base.yaml"
