# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.

"""Upstream HIP compiler exposed under a simulator-specific target name."""

from triton.backends.amd import compiler as amd_compiler

from . import BACKEND_NAME


class GemsimHIPBackend(amd_compiler.HIPBackend):

    @staticmethod
    def supports_target(target):
        return target.backend == BACKEND_NAME
