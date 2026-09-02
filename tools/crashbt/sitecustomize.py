# Loaded by every CPython interpreter (including sglang's spawn children):
# installs the native SIGSEGV/SIGABRT backtracer whose output goes to a
# fixed file, immune to child stderr routing.
try:
    import ctypes
    ctypes.CDLL("/home/zhaosiying/zcode-lane/tools/crashbt/crashbt.so",
                mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
