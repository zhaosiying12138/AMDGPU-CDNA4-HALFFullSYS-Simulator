"""Generic validation helpers for the RowParallelLinear OOT adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


MIN_TP_SIZE = 1
MAX_TP_SIZE = 16


@dataclass(frozen=True)
class RowParallelShard:
    tp_size: int
    tp_rank: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def row_parallel_shard(
    input_size: int,
    *,
    tp_size: int,
    tp_rank: int,
) -> RowParallelShard:
    """Return the contiguous input/weight-column range owned by one TP rank."""

    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise TypeError("row-parallel input_size must be an integer")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        raise TypeError("row-parallel tp_size must be an integer")
    if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
        raise TypeError("row-parallel tp_rank must be an integer")
    if not MIN_TP_SIZE <= tp_size <= MAX_TP_SIZE:
        raise ValueError("row-parallel tp_size must be in 1..16")
    if input_size <= 0 or input_size % tp_size != 0:
        raise ValueError("row-parallel input_size must divide exactly by tp_size")
    if not 0 <= tp_rank < tp_size:
        raise ValueError("row-parallel tp_rank is outside the TP group")
    shard_size = input_size // tp_size
    start = tp_rank * shard_size
    return RowParallelShard(tp_size, tp_rank, start, start + shard_size)


def validate_row_parallel_contract(
    *,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    tp_size: int,
    tp_rank: int,
    weight: torch.Tensor,
) -> RowParallelShard:
    """Validate backend capacity without imposing a model-specific contract."""

    if not isinstance(weight, torch.Tensor):
        raise TypeError("row-parallel weight must be a torch.Tensor")
    if isinstance(output_size, bool) or not isinstance(output_size, int):
        raise TypeError("row-parallel output_size must be an integer")
    if (
        isinstance(input_size_per_partition, bool)
        or not isinstance(input_size_per_partition, int)
    ):
        raise TypeError("row-parallel partition size must be an integer")
    if output_size <= 0:
        raise ValueError("row-parallel output_size must be positive")
    shard = row_parallel_shard(input_size, tp_size=tp_size, tp_rank=tp_rank)
    if input_size_per_partition != shard.size:
        raise RuntimeError("vLLM row-parallel partition metadata is inconsistent")
    if tuple(weight.shape) != (output_size, shard.size):
        raise ValueError("row-parallel local weight shape is invalid")
    return shard


__all__ = [
    "MAX_TP_SIZE",
    "MIN_TP_SIZE",
    "RowParallelShard",
    "row_parallel_shard",
    "validate_row_parallel_contract",
]
