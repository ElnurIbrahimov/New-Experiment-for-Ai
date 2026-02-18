from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    # Base architecture
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 2048  # SwiGLU intermediate size
    vocab_size: int = 50304  # Padded to multiple of 64 for efficiency
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True

    # Mod toggles
    use_diff_attn: bool = False
    use_ssmax: bool = False
    use_mod: bool = False
    use_moe_pp: bool = False
    use_ngpt: bool = False
    use_titans: bool = False
    use_cope: bool = False

    # DiffAttn params
    diff_attn_lambda_init_base: float = 0.8
    diff_attn_lambda_init_decay: float = 0.6

    # SSMax params (no extra params -- learnable per-head s)

    # MoD params
    mod_capacity_ratio: float = 0.5
    mod_aux_loss_weight: float = 0.01

    # MoE++ params
    moe_n_experts: int = 8
    moe_n_zero_experts: int = 2
    moe_n_copy_experts: int = 1
    moe_n_const_experts: int = 1
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01

    # nGPT params (no extra -- uses NormLinear + LERP residual)

    # Titans params
    titans_mem_size: int = 256
    titans_n_persistent_tokens: int = 4
    titans_chunk_size: int = 128
    titans_every_n_layers: int = 3
    titans_mem_layers: int = 2

    # CoPE params
    cope_npos_max: int = 2048


@dataclass
class TrainConfig:
    # Optimization
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 50000

    # Batch
    batch_size: int = 32  # Per GPU
    seq_len: int = 1024
    gradient_accumulation_steps: int = 4

    # Mixed precision
    dtype: str = "bfloat16"

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    tokenizer: str = "gpt2"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50
    log_dir: str = "logs"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    resume_from: Optional[str] = None

    # DDP
    ddp: bool = False
    backend: str = "nccl"

    # Compile
    compile: bool = False

    # Experiment name (auto-generated from config if not set)
    experiment_name: str = ""


def _coerce_type(value, field_type):
    """Coerce a YAML value to the expected dataclass field type."""
    if field_type is float and isinstance(value, str):
        return float(value)
    if field_type is int and isinstance(value, str):
        return int(value)
    if field_type is float and isinstance(value, int):
        return float(value)
    return value


def load_config(path: str) -> tuple[ModelConfig, TrainConfig]:
    """Load ModelConfig and TrainConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_fields = {f.name: f for f in ModelConfig.__dataclass_fields__.values()}
    train_fields = {f.name: f for f in TrainConfig.__dataclass_fields__.values()}

    model_kwargs = {}
    train_kwargs = {}
    for k, v in raw.items():
        if k in model_fields:
            model_kwargs[k] = _coerce_type(v, model_fields[k].type)
        elif k in train_fields:
            train_kwargs[k] = _coerce_type(v, train_fields[k].type)

    return ModelConfig(**model_kwargs), TrainConfig(**train_kwargs)


def save_config(model_config: ModelConfig, train_config: TrainConfig, path: str):
    """Save both configs to a single YAML file."""
    merged = {**asdict(model_config), **asdict(train_config)}
    with open(path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
