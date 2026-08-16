"""Official SGLang SRT platform entry point for GemSim."""

from __future__ import annotations

import os


def _hip_facade_ready() -> bool:
    """Return whether the standard PyTorch ROCm device contract is live."""
    if os.environ.get("GEMSIM_HIP_DEVICE_FACADE") != "1":
        return False
    try:
        import torch

        return bool(
            torch.version.hip is not None
            and torch.cuda.is_available()
            and 1 <= torch.cuda.device_count() <= 16
        )
    except (ImportError, RuntimeError):
        return False


def activate() -> str | None:
    """Activate the diagnostic OOT platform only over a real HIP facade.

    Formal model acceptance leaves ``SGLANG_PLATFORM`` unset so unchanged
    SGLang discovers its own in-tree ``RocmSRTPlatform`` from PyTorch's HIP
    identity.  This explicit plugin remains a bounded diagnostic for the
    third-party c10d backend and must never masquerade as a CPU device.
    """
    if (
        os.environ.get("SGLANG_PLATFORM") != "gemsim"
        or not os.environ.get("ROCM_SIM_ROOT")
        or not _hip_facade_ready()
    ):
        return None
    return "gemsim_sglang.platform:GemsimSRTPlatform"


__all__ = ["activate"]
