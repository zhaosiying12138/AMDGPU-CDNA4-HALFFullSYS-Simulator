"""Generic PyTorch ProcessGroup backed by the GemSim CCL engine.

This module is framework-neutral.  Framework adapters may register the
backend through PyTorch's public third-party ProcessGroup API, but no framework
model, layer, tensor shape, or operator policy belongs here.
"""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _create_work_from_future

from .bootstrap import BootstrapError, build_engine, claim_group
from .engine import AllReduceEngine


BACKEND_NAME = "gemsim_ccl"
_GROUP_NAME = re.compile(r"[a-zA-Z0-9_.-]{1,63}")
_registration_lock = threading.Lock()
_registered = False


class ProcessGroupError(RuntimeError):
    pass


def _completed_work(result: Any):
    future = torch.futures.Future()
    future.set_result(result)
    return _create_work_from_future(future)


def _timeout_ns(timeout: timedelta) -> int:
    if not isinstance(timeout, timedelta):
        raise TypeError("ProcessGroup timeout must be datetime.timedelta")
    value = int(timeout.total_seconds() * 1_000_000_000)
    if value <= 0 or value > (1 << 63) - 1:
        raise ValueError("ProcessGroup timeout is outside the CCL range")
    return value


def _binding_name(group_id: str) -> str:
    if not isinstance(group_id, str) or _GROUP_NAME.fullmatch(group_id) is None:
        raise ProcessGroupError("c10d group_id is not canonical")
    return f"c10d:{group_id}"


def _stage_for_engine(tensor: torch.Tensor) -> torch.Tensor:
    """Return an immutable contiguous CPU view for the transport engine."""
    return tensor.detach().to(device="cpu", copy=True).contiguous()


class GemsimProcessGroup(dist.ProcessGroup):
    """SUM-only process group with failure-atomic public tensor commit.

    CPU tensors are useful for protocol diagnostics.  CUDA-typed tensors are
    accepted only from a HIP-enabled PyTorch build; PyTorch ROCm intentionally
    exposes HIP devices through the ``cuda`` device type.  Device data is
    staged as bytes for the transport and committed back only after every
    private collective succeeds.  The engine performs no host reduction.
    """

    def __init__(
        self,
        store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        *,
        group_id: str,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        if type(rank) is not int or type(world_size) is not int:
            raise TypeError("rank and world_size must be exact integers")
        if not 0 <= rank < world_size or not 1 <= world_size <= 16:
            raise ValueError("GemSim CCL rank/world_size must be in 1..16")
        super().__init__(rank, world_size)
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._group_id = group_id
        self._timeout_ns = _timeout_ns(timeout)
        self._barrier_sequence = 0
        self._closed = False
        if engine_factory is not None:
            self._engine = engine_factory()
        elif world_size == 1:
            self._engine = AllReduceEngine.singleton()
        else:
            prefix = os.environ.get("ROCM_SIM_ROOT")
            if not prefix:
                raise ProcessGroupError("ROCM_SIM_ROOT is required")
            try:
                binding = claim_group(
                    _binding_name(group_id),
                    expected_rank=rank,
                    expected_world_size=world_size,
                    expected_product_prefix=Path(prefix),
                )
            except (BootstrapError, OSError) as error:
                raise ProcessGroupError(str(error)) from error
            self._timeout_ns = min(
                self._timeout_ns,
                binding["group"]["rank"]["collective_timeout_ns"],
            )
            self._engine = build_engine(binding)
        if self._engine.rank != rank or self._engine.world_size != world_size:
            self._engine.destroy()
            raise ProcessGroupError("CCL engine differs from the c10d group")

    def getBackendName(self) -> str:
        return BACKEND_NAME

    @staticmethod
    def _validate_tensors(tensors: Sequence[torch.Tensor], opts) -> None:
        if not isinstance(tensors, (list, tuple)) or not tensors:
            raise TypeError("allreduce requires a nonempty tensor list")
        if opts.reduceOp != dist.ReduceOp.SUM:
            raise NotImplementedError("GemSim CCL supports SUM only")
        device = tensors[0].device
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("allreduce entries must be torch.Tensor")
            if tensor.device != device:
                raise ValueError("allreduce tensors must share one device")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError("GemSim CCL supports CPU diagnostics and HIP tensors")
            if tensor.device.type == "cuda" and torch.version.hip is None:
                raise ValueError("CUDA tensors are accepted only from a HIP PyTorch build")
            if tensor.dtype not in (torch.bfloat16, torch.float32):
                raise TypeError("GemSim CCL supports bfloat16 and float32")
            if tensor.numel() == 0 or not tensor.is_contiguous():
                raise ValueError("allreduce tensors must be nonempty and contiguous")

    def allreduce(self, tensors, opts):
        if self._closed:
            raise ProcessGroupError("ProcessGroup is closed")
        self._validate_tensors(tensors, opts)
        staged = [_stage_for_engine(tensor) for tensor in tensors]
        before = [tensor.view(torch.uint8).clone() for tensor in staged]
        outputs = [
            self._engine.all_reduce(tensor, timeout_ns=self._timeout_ns)
            for tensor in staged
        ]
        current = [_stage_for_engine(tensor) for tensor in tensors]
        for private_input, original, current_input, output in zip(
            staged, before, current, outputs
        ):
            if not torch.equal(current_input.view(torch.uint8), original):
                raise ProcessGroupError("CCL engine modified its public input")
            if (
                not isinstance(output, torch.Tensor)
                or output.shape != private_input.shape
                or output.dtype != private_input.dtype
                or output.device.type != "cpu"
                or not output.is_contiguous()
            ):
                raise ProcessGroupError("CCL engine returned an invalid tensor")
        for public, private in zip(tensors, outputs):
            if private.shape != public.shape or private.dtype != public.dtype:
                raise ProcessGroupError("CCL engine output differs from public tensor")
        # One caller-visible commit phase after every private collective passes.
        for public, private in zip(tensors, outputs):
            public.copy_(private, non_blocking=False)
        return _completed_work(tensors)

    def allreduce_coalesced(self, tensors, opts):
        return self.allreduce(tensors, opts)

    def barrier(self, opts):
        if self._closed:
            raise ProcessGroupError("ProcessGroup is closed")
        sequence = self._barrier_sequence
        self._barrier_sequence += 1
        prefix = f"gemsim-ccl-barrier/{self._group_id}/{sequence}"
        if self._store.add(prefix + "/arrived", 1) == self._world_size:
            self._store.set(prefix + "/complete", b"1")
        self._store.wait([prefix + "/complete"])
        return _completed_work(None)

    def _unsupported(self, operation: str):
        raise NotImplementedError(
            f"GemSim CCL {operation} has no accepted implementation"
        )

    def broadcast(self, tensors, opts):
        return self._unsupported("broadcast")

    def allgather(self, output_tensors, input_tensors, opts):
        return self._unsupported("allgather")

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        return self._unsupported("reduce_scatter")

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            self._engine.close(timeout_ns=self._timeout_ns)
        except Exception:
            self._engine.destroy()
            self._closed = True
            raise
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._engine.destroy()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.abort()
        except Exception:
            pass


def _creator(options, backend_options):
    if backend_options is not None:
        raise ProcessGroupError("custom ProcessGroup options are not supported")
    global_ranks = list(options.global_ranks_in_group)
    if len(global_ranks) != int(options.group_size):
        raise ProcessGroupError("c10d global rank map differs from group_size")
    return GemsimProcessGroup(
        options.store,
        int(options.group_rank),
        int(options.group_size),
        options.timeout,
        group_id=str(options.group_id),
    )


def register_backend() -> None:
    """Register exactly one official PyTorch third-party backend."""
    global _registered
    with _registration_lock:
        if _registered:
            return
        if BACKEND_NAME in dist.Backend.backend_list:
            raise ProcessGroupError(f"c10d backend name is already owned: {BACKEND_NAME}")
        dist.Backend.register_backend(
            BACKEND_NAME,
            _creator,
            extended_api=True,
            devices=["cpu", "cuda"],
        )
        _registered = True


__all__ = [
    "BACKEND_NAME",
    "GemsimProcessGroup",
    "ProcessGroupError",
    "register_backend",
]
