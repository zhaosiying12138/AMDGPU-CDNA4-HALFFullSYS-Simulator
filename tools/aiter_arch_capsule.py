#!/usr/bin/env python3
"""Check the architecture discovery boundary used by aiter_meta."""

import json
import os

from aiter_meta.csrc.cpp_itfs import utils


validated = utils.validate_and_update_archs()
payload = {
    "default_gpu_arch": utils.DEFAULT_GPU_ARCH,
    "gpu_arch": utils.GPU_ARCH,
    "gpu_archs_env": os.environ.get("GPU_ARCHS", ""),
    "validated_archs": validated,
}
print(json.dumps(payload, sort_keys=True))

if not payload["gpu_archs_env"] or not validated:
    raise SystemExit("aiter_meta did not receive a usable GPU architecture")
