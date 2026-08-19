"""Import-only probe for the opt-in SGLang layer-gate startup contract."""

from qwen35_sglang_layer_gate import assert_installed

assert_installed()
print("qwen35 layer gate import probe: PASS", flush=True)
