#!/usr/bin/env python3
"""Minimal hybrid-crash reproducer: construct the sglang Engine (Qwen3.5-0.8B,
TP1, load_format=dummy) far enough to reach the RoPE inv-freq kernels that
segfault under --hybrid-cta in the model lane, without weight loading.

The crash fires during model __init__ (rotary _compute_inv_freq), which runs
BEFORE any weight I/O, so dummy weights reach it equally.  Printed marker
ENGINE_CONSTRUCTED means this capsule survived the crash window.
"""
import ctypes
import faulthandler

faulthandler.enable()
# Native crash backtracer: installs SIGSEGV/SIGABRT handlers that dump a
# native stack to stderr. Loaded at import time so the spawn-reimported
# scheduler child installs it too.
ctypes.CDLL("/home/zhaosiying/zcode-lane/tools/crashbt/crashbt.so",
            mode=ctypes.RTLD_GLOBAL)

def main():
    from sglang.srt.entrypoints.engine import Engine
    engine = Engine(
        model_path="/home/zhaosiying/zcode-lane/models/Qwen3.5-0.8B",
        tp_size=1,
        dtype="bfloat16",
        attention_backend="aiter",
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        max_total_tokens=16,
        max_running_requests=1,
        max_mamba_cache_size=5,
        random_seed=0,
        watchdog_timeout=86400,
        dist_timeout=300,
        context_length=16,
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        load_format="dummy",
        log_level="info",
)
    print("ENGINE_CONSTRUCTED", flush=True)
    engine.shutdown()
    print("MINI_ENGINE_PASS", flush=True)


if __name__ == "__main__":
    main()
