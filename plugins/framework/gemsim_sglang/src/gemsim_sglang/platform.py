"""Diagnostic SGLang platform over the standard PyTorch ROCm device surface."""

from __future__ import annotations

import torch

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.platforms.rocm import RocmDeviceMixin

from .process_group import BACKEND_NAME, register_backend


class GemsimSRTPlatform(RocmDeviceMixin, SRTPlatform):
    """Bounded OOT diagnostic that preserves upstream ROCm device semantics.

    Production model acceptance uses SGLang's in-tree ``RocmSRTPlatform`` and
    the upstream HIP/Triton/RCCL surfaces.  This class exists only to exercise
    the public platform and c10d hooks while the transparent device facade is
    brought up.  It deliberately advertises no graph capability yet.
    """

    _enum = PlatformEnum.OOT
    device_name = "rocm"
    device_type = "cuda"

    def apply_server_args_defaults(self, server_args) -> None:
        for name, value in (
            ("disable_custom_all_reduce", True),
            ("enable_mscclpp", False),
            ("enable_torch_symm_mem", False),
            ("pre_warm_nccl", False),
            ("disable_cuda_graph", True),
        ):
            if hasattr(server_args, name):
                setattr(server_args, name, value)
        if (
            hasattr(server_args, "attention_backend")
            and server_args.attention_backend is None
        ):
            server_args.attention_backend = "triton"

    def get_default_attention_backend(self) -> str:
        return "triton"

    def get_torch_distributed_backend_str(self) -> str:
        return BACKEND_NAME

    def get_compile_backend(self, mode: str | None = None) -> str:
        return "inductor"

    def get_dispatch_key_name(self) -> str:
        return "hip"

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

        return DecodeCudaGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return DSATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator.paged import (
            PagedTokenToKVPoolAllocator,
        )

        return PagedTokenToKVPoolAllocator

    def get_piecewise_backend_cls(self) -> type:
        raise NotImplementedError(
            "piecewise graph compilation is not accepted for GemSim"
        )

    def get_torch_profiler_activity_str(self) -> str:
        return "cuda"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.CUDA

    def init_backend(self) -> None:
        register_backend()


__all__ = ["GemsimSRTPlatform"]
