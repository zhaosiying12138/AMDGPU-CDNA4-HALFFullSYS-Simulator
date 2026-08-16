"""Formal out-of-tree vLLM communicator over the accepted CCL engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from .ccl_bootstrap import BootstrapError, claim_group
from gemsim_ccl.bootstrap import build_engine as _shared_build_engine


class CommunicatorError(RuntimeError):
    pass


def _build_engine(binding: dict[str, Any]):
    return _shared_build_engine(binding)


class GemsimDeviceCommunicator(DeviceCommunicatorBase):
    """SUM-only communicator whose all-reduce payload never enters Gloo.

    Gloo remains available through ``cpu_group`` for vLLM control metadata and
    barriers. Every tensor method not backed by accepted device-live evidence
    is explicitly rejected instead of inheriting ``torch.distributed``.
    """

    supports_tensor_dict = False

    def __init__(
        self,
        cpu_group,
        device: torch.device | None = None,
        device_group=None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ) -> None:
        if global_ranks is not None or global_world_size is not None:
            raise CommunicatorError("stateless or elastic groups are not supported")
        if use_all2all:
            raise CommunicatorError("all-to-all and expert parallelism are not supported")
        if not isinstance(unique_name, str) or not unique_name.startswith("tp:"):
            raise CommunicatorError("only named tp:* groups are supported")
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
            use_all2all=use_all2all,
        )
        if self.device.type != "cpu":
            raise CommunicatorError("GemSim tensors must remain on CPU staging")
        if not 2 <= self.world_size <= 16:
            raise CommunicatorError("GemSim CCL world_size must be in 2..16")
        if self.device.index not in (None, self.rank):
            raise CommunicatorError("device index differs from the CCL control rank")
        configured_prefix = os.environ.get("ROCM_SIM_ROOT")
        if not configured_prefix:
            raise CommunicatorError("ROCM_SIM_ROOT is required")
        try:
            binding = claim_group(
                unique_name,
                expected_rank=self.rank,
                expected_world_size=self.world_size,
                expected_product_prefix=Path(configured_prefix),
            )
        except (BootstrapError, OSError) as error:
            raise CommunicatorError(str(error)) from error
        self._collective_timeout_ns = binding["group"]["rank"][
            "collective_timeout_ns"
        ]
        self._engine = _build_engine(binding)
        if (
            self._engine.rank != self.rank
            or self._engine.world_size != self.world_size
        ):
            self._engine.destroy()
            self._engine = None
            raise CommunicatorError("CCL engine differs from the vLLM control group")

    @staticmethod
    def _unsupported(name: str):
        raise NotImplementedError(
            f"GemSim CCL {name} has no accepted device-live implementation"
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self._engine is None:
            raise CommunicatorError("communicator is destroyed")
        if not isinstance(input_, torch.Tensor):
            raise TypeError("all_reduce input must be a torch.Tensor")
        if input_.device.type != "cpu":
            raise ValueError("all_reduce input must use CPU staging")
        # The formal live topology matrix currently accepts BF16. FP32 remains
        # inside the primitive boundary until its live communicator matrix runs.
        if input_.dtype != torch.bfloat16:
            raise TypeError("the accepted vLLM all_reduce dtype is bfloat16")
        if not input_.is_contiguous() or input_.numel() == 0:
            raise ValueError("all_reduce input must be nonempty and contiguous")
        output = self._engine.all_reduce(
            input_, timeout_ns=self._collective_timeout_ns
        )
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != input_.shape
            or output.dtype != input_.dtype
            or output.device != input_.device
            or not output.is_contiguous()
        ):
            raise CommunicatorError("all_reduce returned an invalid tensor contract")
        # Input immutability and fresh-storage are enforced by the CCL engine
        # and its standalone contract tests.  Python storage introspection is
        # deliberately kept out of the compiled vLLM graph.
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self._unsupported("all_gather")

    def all_gatherv(self, input_, dim: int = 0, sizes=None):
        return self._unsupported("all_gatherv")

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self._unsupported("reduce_scatter")

    def reduce_scatterv(self, input_, dim: int = -1, sizes=None):
        return self._unsupported("reduce_scatterv")

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1):
        return self._unsupported("gather")

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        self._unsupported("send")

    def recv(self, size, dtype, src: int | None = None) -> torch.Tensor:
        return self._unsupported("recv")

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        return self._unsupported("broadcast")

    def batch_isend_irecv(self, p2p_ops: list):
        return self._unsupported("batch_isend_irecv")

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        return self._unsupported("dispatch_router_logits")

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        return self._unsupported("dispatch")

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        return self._unsupported("combine")

    def checkpoint_prepare(self) -> None:
        self._unsupported("checkpoint_prepare")

    def checkpoint_restore(self) -> None:
        self._unsupported("checkpoint_restore")

    def destroy(self) -> None:
        engine = getattr(self, "_engine", None)
        self._engine = None
        if engine is not None:
            try:
                engine.close(timeout_ns=5_000_000_000)
            except Exception:
                engine.destroy()


__all__ = ["CommunicatorError", "GemsimDeviceCommunicator"]
