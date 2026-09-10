# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.

"""Compatibility facade for the framework-neutral GemSim CCL bootstrap."""

from gemsim_ccl.bootstrap import (
    BootstrapError,
    LEGACY_VLLM_SCHEMA as SCHEMA,
    _reset_claims_for_tests,
    claim_group,
)

__all__ = ["BootstrapError", "SCHEMA", "claim_group"]
