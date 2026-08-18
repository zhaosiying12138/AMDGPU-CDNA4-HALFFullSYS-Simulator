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

    stream = open(_LOG_PATH, "a", buffering=1)
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


_install()
