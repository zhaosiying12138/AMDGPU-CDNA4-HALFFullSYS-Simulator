"""Log every Triton kernel launch with its name and grid.

Attributing simulator wall-clock to a *specific* kernel is otherwise guesswork:
gem5's dispatch trace records launch geometry but no kernel identity, and
reasoning backwards from grid arithmetic to a source kernel has repeatedly been
wrong here. Ask Triton directly instead.

This is a diagnostic shim, not a product change. It installs through the
ordinary ``sitecustomize`` mechanism, so no edit to SGLang, vLLM, aiter or
Triton is required: put this directory first on PYTHONPATH and set
SAGR_TRITON_LAUNCH_LOG to a writable path.

It wraps ``CompiledKernel.launch_metadata``. That is the one call on the actual
launch path -- ``JITFunction.run`` invokes it immediately before
``kernel.run`` -- and it receives both the kernel (``self.name``) and the grid.
``CompiledKernel.__getitem__`` is not on this path, and
``knobs.runtime.launch_enter_hook`` is a ``HookChain`` that only carries the
name, not the grid that identifies the expensive launch geometry.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time


_LOG_PATH = os.environ.get("SAGR_TRITON_LAUNCH_LOG")


def _install() -> None:
    if not _LOG_PATH:
        return
    try:
        from triton.compiler.compiler import CompiledKernel
    except Exception:  # pragma: no cover - Triton absent
        return
    if getattr(CompiledKernel, "_sagr_probe", False):
        return

    try:
        stream = open(_LOG_PATH, "a", buffering=1)
    except OSError as error:
        # An unopenable log must not take the fast-autotune shim down with
        # it: sitecustomize has no caller to catch, and one missing log
        # directory once silently reverted a whole lane to real autotuning.
        print(f"[sagr] triton launch probe DISABLED ({error})", file=sys.stderr)
        return
    lock = threading.Lock()
    counter = {"n": 0}
    original = CompiledKernel.launch_metadata

    def probed_launch_metadata(self, grid, stream_arg, *args):
        with lock:
            counter["n"] += 1
            record = {
                "seq": counter["n"],
                "monotonic": round(time.monotonic(), 4),
                "name": getattr(self, "name", None),
                "grid": list(grid) if isinstance(grid, (list, tuple)) else str(grid),
                "num_warps": getattr(getattr(self, "metadata", None), "num_warps", None),
            }
            try:
                stream.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass
        return original(self, grid, stream_arg, *args)

    CompiledKernel.launch_metadata = probed_launch_metadata
    CompiledKernel._sagr_probe = True
    print(f"[sagr] triton launch probe active -> {_LOG_PATH}", file=sys.stderr)


def _install_fast_autotune() -> None:
    """Skip triton.testing.do_bench's L2-flush zero under the simulator.

    Autotuning is the vLLM bring-up wall: every do_bench rep clears a
    256 MB scratch buffer (64M int32 / vec4 = the 16.7M-workitem fills
    that dominated whole 12 h lanes), and a config sweep pays it ~8x per
    config x 27 configs per kernel.  On gem5 the flush buys nothing --
    the "timing" that picks the config is simulator time, already not
    hardware-representative, and every candidate config computes the
    same math.  Gated by SAGR_TRITON_FAST_AUTOTUNE so standard lanes are
    untouched.  The driver patch is late-bound (installed at the first
    do_bench call) because the triton driver cannot instantiate at
    interpreter startup.
    """
    if os.environ.get("SAGR_TRITON_FAST_AUTOTUNE") != "1":
        return
    try:
        import triton.testing
    except Exception:  # pragma: no cover - Triton absent
        return
    if getattr(triton.testing.do_bench, "_sagr_fast_autotune", False):
        return

    def fast_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                      quantiles=None, return_mode="mean"):
        # Run the candidate once so it compiles and its launch is proven,
        # then hand the autotuner a constant: under the simulator the
        # wall time of a kernel says nothing about hardware, and letting
        # do_bench measure it backfires twice -- the 256 MB L2-flush zero
        # per rep costs more than everything else in the sweep, and once
        # the flush is gone, gem5's near-zero event timings inflate
        # n_repeat = rep/estimate into tens of thousands of relaunches of
        # a single config (observed: 11,221 launches of one kernel).
        # A constant makes the tuner pick deterministically; every
        # candidate computes the same math, so any pick is correct.
        fn()
        if quantiles:
            return tuple(1.0 for _ in quantiles)
        return 1.0

    fast_do_bench._sagr_fast_autotune = True
    triton.testing.do_bench = fast_do_bench
    print("[sagr] triton fast-autotune active (do_bench returns a constant)",
          file=sys.stderr)


_install()
_install_fast_autotune()
