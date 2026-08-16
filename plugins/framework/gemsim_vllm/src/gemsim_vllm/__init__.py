"""Formal out-of-tree entry points for the GemSim vLLM integration."""

from __future__ import annotations

import os


def _enabled() -> bool:
    return (
        os.environ.get("TRITON_DEFAULT_BACKEND") == "gemsim_amd"
        and bool(os.environ.get("ROCM_SIM_ROOT"))
    )


def platform_plugin() -> str | None:
    """Activate only inside the explicit private GemSim runtime environment."""
    if not _enabled():
        return None
    return "gemsim_vllm.platform.GemsimPlatform"


def register_ops() -> None:
    """Register project-owned operators, adapters, and model architecture."""
    if not _enabled():
        return
    from . import ops as _ops  # noqa: F401
    _ops.register_upstream_compile_symbols()
    _ops.validate_target()

    try:
        from . import adapters as _adapters  # noqa: F401
        from vllm import ModelRegistry
    except ModuleNotFoundError as error:
        if error.name != "vllm":
            raise
        return

    architecture = "GemsimQwen3_5ForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture,
            "gemsim_vllm.model:GemsimQwen3_5ForCausalLM",
        )


__all__ = ["platform_plugin", "register_ops"]
