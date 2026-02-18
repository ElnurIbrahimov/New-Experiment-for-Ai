import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from datasets import load_dataset
from typing import Optional
import itertools


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    """Get a tiktoken tokenizer."""
    return tiktoken.get_encoding(name)


class FineWebEduDataset(IterableDataset):
    """Streaming dataset from FineWeb-Edu with token packing."""

    def __init__(
        self,
        seq_len: int,
        split: str = "train",
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_subset: str = "sample-10BT",
        tokenizer_name: str = "gpt2",
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.split = split
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.eot_token = self.tokenizer.eot_token

    def _get_stream(self):
        ds = load_dataset(
            self.dataset_name,
            self.dataset_subset,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        # Shard across DDP workers
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        return ds

    def __iter__(self):
        buffer = []
        stream = self._get_stream()

        for example in stream:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, allowed_special=set())
            tokens.append(self.eot_token)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def get_dataloader(
    seq_len: int,
    batch_size: int,
    split: str = "train",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
    tokenizer_name: str = "gpt2",
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    dataset = FineWebEduDataset(
        seq_len=seq_len,
        split=split,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        tokenizer_name=tokenizer_name,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
