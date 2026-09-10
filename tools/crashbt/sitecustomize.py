# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.

# Loaded by every CPython interpreter (including sglang's spawn children):
# installs the native SIGSEGV/SIGABRT backtracer whose output goes to a
# fixed file, immune to child stderr routing.
try:
    import ctypes
    ctypes.CDLL("/home/zhaosiying/zcode-lane/tools/crashbt/crashbt.so",
                mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
