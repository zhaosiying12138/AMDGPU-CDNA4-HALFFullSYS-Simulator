"""Upstream HIP compiler exposed under a simulator-specific target name."""

from triton.backends.amd import compiler as amd_compiler

from . import BACKEND_NAME


class GemsimHIPBackend(amd_compiler.HIPBackend):

    @staticmethod
    def supports_target(target):
        return target.backend == BACKEND_NAME
