"""vLLM out-of-tree platform metadata for the CPU-staging simulator path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class GemsimPlatform(Platform):
    """A gfx950 simulator whose PyTorch-visible storage remains on the CPU."""

    _enum = PlatformEnum.OOT
    # vLLM uses this field to construct torch.device objects.  GemSim keeps
    # tensors on CPU; the Triton target identity remains separate.
    device_name = "cpu"
    device_type = "cpu"
    dispatch_key = "CPU"
    # Gloo is control-plane only. Tensor collectives are handled by the
    # out-of-tree communicator and must never fall back to this ProcessGroup.
    dist_backend = "gloo"
    # Keep vLLM/PyTorch's upstream compiler selection.  The simulator driver
    # is the execution backend; it must not silently turn a user's
    # torch.compile request into eager mode.
    simple_compile_backend = "inductor"
    supported_quantization: list[str] = []

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        if type(device_id) is not int or not 0 <= device_id < 16:
            raise ValueError("GemSim device_id must be an integer in 0..15")
        return DeviceCapability(9, 5)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        cls.get_device_capability(device_id)
        return f"GemSim gfx950 rank {device_id}"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        cls.get_device_capability(device_id)
        return f"gemsim-gfx950-rank{device_id}"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "gemsim_vllm.communicator.GemsimDeviceCommunicator"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config import CUDAGraphMode

        parallel = vllm_config.parallel_config
        if (
            getattr(parallel, "nnodes", 1) != 1
            or getattr(parallel, "node_rank", 0) != 0
        ):
            raise ValueError("GemSim currently accepts a single node only")
        unsupported = {
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "prefill_context_parallel_size": parallel.prefill_context_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "data_parallel_size_local": parallel.data_parallel_size_local,
            "decode_context_parallel_size": parallel.decode_context_parallel_size,
        }
        if any(value != 1 for value in unsupported.values()):
            raise ValueError(
                "GemSim currently accepts single-node pure tensor parallelism only"
            )
        if not 1 <= parallel.tensor_parallel_size <= 16:
            raise ValueError("GemSim tensor_parallel_size must be in 1..16")
        if any(
            bool(getattr(parallel, field, False))
            for field in (
                "enable_expert_parallel",
                "enable_elastic_ep",
                "enable_eplb",
                "enable_dbo",
            )
        ):
            raise ValueError("GemSim does not accept EP, elastic groups, EPLB, or DBO")
        model = getattr(vllm_config, "model_config", None)
        if model is not None and bool(getattr(model, "is_moe", False)):
            raise ValueError("GemSim tensor parallelism does not accept MoE models")
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("GemSim tensor parallelism does not accept speculation")

        compilation = getattr(vllm_config, "compilation_config", None)
        if compilation is not None:
            # GemSim uses synchronous CPU staging and has no CUDA graph
            # capture ABI.  Reject an explicit graph request instead of
            # silently changing it; all other upstream compile settings remain
            # authoritative and are handled by vLLM's normal config path.
            if compilation.cudagraph_mode != CUDAGraphMode.NONE:
                raise ValueError(
                    "GemSim does not support CUDA Graph capture; use "
                    "torch.compile without cudagraphs"
                )

    @classmethod
    def import_kernels(cls) -> None:
        """All accepted kernels are Python/Triton operators from this package."""

    @classmethod
    def import_ir_kernels(cls) -> None:
        from . import ops as _ops  # noqa: F401

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ) -> str:
        """Select the simulator attention facade for an upstream backend.

        The facade preserves vLLM's metadata/implementation contract while
        dispatching supported kernels through the OOT Triton driver.  Backend
        selection remains capability-driven: an explicitly requested backend
        is accepted only when the generic facade can represent it, and no
        hidden CPU attention fallback is introduced.
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if attn_selector_config.use_mla or attn_selector_config.use_sparse:
            raise ValueError("GemSim construction does not support MLA/sparse attention")
        if attn_selector_config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("GemSim construction requires FP16 or BF16 attention")
        if selected_backend not in (
            None,
            AttentionBackendEnum.CPU_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
        ):
            raise ValueError(
                f"unsupported construction attention backend: {selected_backend}"
            )
        return "gemsim_vllm.attention.GemsimAttentionBackend"
