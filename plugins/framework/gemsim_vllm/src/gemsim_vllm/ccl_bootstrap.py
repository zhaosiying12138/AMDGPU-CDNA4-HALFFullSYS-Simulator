"""Compatibility facade for the framework-neutral GemSim CCL bootstrap."""

from gemsim_ccl.bootstrap import (
    BootstrapError,
    LEGACY_VLLM_SCHEMA as SCHEMA,
    _reset_claims_for_tests,
    claim_group,
)

__all__ = ["BootstrapError", "SCHEMA", "claim_group"]
