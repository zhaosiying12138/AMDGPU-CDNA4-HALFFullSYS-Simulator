#!/usr/bin/env python3
"""Run the existing layer-0 GDN chain capsule with Qwen3.5-9B geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/qwen35_gdn_layer0_decode_chain_capsule.py"
SHARDS = (
    ROOT / "models/Qwen3.5-9B/model.safetensors-00003-of-00004.safetensors",
    ROOT / "models/Qwen3.5-9B/model.safetensors-00004-of-00004.safetensors",
)


def _load_source():
    spec = importlib.util.spec_from_file_location("qwen35_gdn_layer0_chain", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_source()
    from safetensors.torch import load_file as load_shard

    module.HIDDEN = 4096
    module.H = 16
    module.HV = 32
    module.K = 128
    module.V = 128
    module.DIM = 8192
    module.ZDIM = 4096
    module.WIDTH = 4
    module.SCALE = float(module.K) ** -0.5
    module.MODEL_FILE = SHARDS[0]

    def load_9b_weights(_path, device="cpu"):
        merged = {}
        for shard in SHARDS:
            merged.update(load_shard(str(shard), device=device))
        return merged

    module.load_file = load_9b_weights
    original_diff_loader = module._load_decode_diff_module

    def load_9b_diff():
        diff = original_diff_loader()
        diff.H = 16
        diff.HV = 32
        diff.K = 128
        diff.V = 128
        diff.DIM = 8192
        diff.STATE_LEN = 3
        diff.SCALE = float(diff.K) ** -0.5
        diff.MODEL_FILE = SHARDS[0]
        return diff

    module._load_decode_diff_module = load_9b_diff
    sys.argv = [
        str(SOURCE),
        "--output-dir",
        str(ROOT / "artifacts/qwen35-gdn-layer0-9b-capsule/v1"),
    ]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
