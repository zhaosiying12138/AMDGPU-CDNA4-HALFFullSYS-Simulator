"""Opt-in Python startup hook for the SGLang layer differential."""

from __future__ import annotations

import os
from pathlib import Path
import runpy


if os.environ.get("SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT"):
    if os.environ.get("SAGR_TRITON_LAUNCH_LOG"):
        runpy.run_path(
            str(
                Path(__file__).resolve().parents[1]
                / "triton_launch_probe/sitecustomize.py"
            ),
            run_name="_sagr_triton_launch_probe",
        )
    from qwen35_sglang_layer_gate import install

    install()
