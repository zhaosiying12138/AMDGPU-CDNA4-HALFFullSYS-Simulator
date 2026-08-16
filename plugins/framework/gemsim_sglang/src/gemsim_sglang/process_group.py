"""Compatibility import for the framework-neutral GemSim ProcessGroup.

The implementation is owned by ``gemsim_ccl`` so SGLang is not a second
collective architecture.  New consumers should import the shared module
directly.
"""

from gemsim_ccl.torch_process_group import (
    BACKEND_NAME,
    GemsimProcessGroup,
    ProcessGroupError,
    register_backend,
)

__all__ = [
    "BACKEND_NAME",
    "GemsimProcessGroup",
    "ProcessGroupError",
    "register_backend",
]
