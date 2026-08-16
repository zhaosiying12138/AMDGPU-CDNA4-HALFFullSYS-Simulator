#!/usr/bin/env python3
"""Run an upstream vLLM layer through the registered GemSim OOT implementation."""

from pathlib import Path
import runpy


_BOOTSTRAP = Path(__file__).resolve().parents[1] / "triton/_gemsim_bootstrap.py"
runpy.run_path(str(_BOOTSTRAP))["bootstrap"](__file__, "quickstart-vllm-silu")

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.plugins import load_general_plugins


load_general_plugins()
from vllm.model_executor.layers.activation import SiluAndMul


with set_current_vllm_config(VllmConfig()):
    layer = SiluAndMul()
    if type(layer).__module__ != "gemsim_vllm.adapters":
        raise RuntimeError(f"vLLM OOT replacement was not selected: {type(layer)}")
    torch.manual_seed(13)
    value = (torch.randn((2, 256), dtype=torch.float32) * 0.125).to(torch.bfloat16)
    actual = layer(value)
left, right = value.chunk(2, dim=-1)
expected = (torch.nn.functional.silu(left.float()) * right.float()).to(torch.bfloat16)
torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03125)
print("vLLM SiluAndMul passed through the GemSim OOT plugin")
