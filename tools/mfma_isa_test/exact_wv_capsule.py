#!/usr/bin/env python3
"""Exact-kernel reproducer for the wvSplitKrc m%16==0 NaN defect.

Loads the gfx950 fatbin ELF carved from vllm's _rocm_C.abi3.so and
launches the very kernel the engine dispatches (THRDS=64, YTILE=16,
WvPrGrp=4, A_CHUNK=8, UNRL=1, N=16, GrpsShrB=1, CHUNKK=1, DTRM=1)
with fp16 inputs, against a CPU matmul reference.  Also mirrors the
host-side glbl/cntr workspaces.
"""
import ctypes
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import torch

# The engine's ctx16 dispatch resolves to wvSplitK (not rc): the
# m>8 && n<=5 branch of rocm_unquantized_gemm_impl with x=(16,1024)
# fp16 -> wvSplitK_hf_sml_<fp16,32,4,16,8,2,2> (group=165504, wg=(64,16)).
SYM = ("_Z11wvSplitKrc_I6__halfLi64ELi16ELi4ELi8ELi1ELi16ELi1ELi1ELi1EE"
       "viiiiiiPKT_S3_S3_PfPiPS1_i")


def main() -> int:
    out_dir = Path(os.environ.get("EXACT_WV_OUTPUT", "/tmp/exactwv-art"))
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = Path("/tmp/seg_gfxcandidate.elf").read_bytes()

    hip_path = None
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        cand = os.path.join(d, "libamdhip64.so.7")
        if os.path.exists(cand):
            hip_path = cand
            break
    lib = ctypes.CDLL(hip_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]

    dev = "cuda"
    # Match the engine's real dispatch: wvSplitKrc fp16 N=16 family,
    # block=(64,4), grid.x = CuCount from torch device props.
    import os as _os
    M = int(_os.environ.get("EXACT_WV_M", "16"))
    N = int(_os.environ.get("EXACT_WV_N", "256"))
    K = int(_os.environ.get("EXACT_WV_K", "1024"))
    CuCount = int(os.environ.get("EXACT_WV_CUCOUNT", "0")) or \
        torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(0)
    import os as _os
    if _os.environ.get("EXACT_WV_MAP_HALF"):
        # K-half discriminator: A nonzero only on the chosen half of K.
        # Correct C = (K/2)*0.001*m*n = 0.512*m*n; per row parity a value of
        # 0.512mn / 0.256mn / 0 says which k-window that row consumed.
        half = _os.environ["EXACT_WV_MAP_HALF"]
        kk = torch.arange(K)
        mask = (kk < K // 2) if half == "first" else (kk >= K // 2)
        A = (torch.arange(M, dtype=torch.float32)[:, None] * 0.001 *
             mask[None, :].float()).half().to(dev)
        B = (torch.arange(N, dtype=torch.float32)[:, None] *
             torch.ones(1, K)).half().to(dev)
    elif _os.environ.get("EXACT_WV_MAP_DIAG"):
        # Mapping diagnostic: C[i][n] should be exactly 1.024 * i * n.
        # Any mis-gather shows up as which (i', n') each element really
        # consumed: got/1.024 == i' * n'.
        A = (torch.arange(M, dtype=torch.float32)[:, None] * 0.001 *
             torch.ones(1, K)).half().to(dev)
        B = (torch.arange(N, dtype=torch.float32)[:, None] *
             torch.ones(1, K)).half().to(dev)
    else:
        A = torch.randn(M, K, dtype=torch.float16).to(dev)
        B = torch.randn(N, K, dtype=torch.float16).to(dev)
    # wvSplitKrc uses one counter per CU and one 16-byte partial slot per
    # workgroup; the real vllm caller sizes these from grid geometry. Give
    # generous headroom so small-buffer overflow cannot masquerade as a
    # simulator defect.
    glbl = torch.zeros(1 << 20, dtype=torch.float32).to(dev)
    cntr = torch.zeros(1 << 16, dtype=torch.int32).to(dev)
    C = torch.full((N, M), 777.0, dtype=torch.float16).to(dev)

    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(elf)
    rc = lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p))
    assert rc == 0, f"load {rc}"
    fn = ctypes.c_void_p()
    rc = lib.hipModuleGetFunction(ctypes.byref(fn), module, SYM.encode())
    assert rc == 0, f"getfn {rc}"

    i_actlN = ctypes.c_int(N)          # wvSplitKrc: actlN
    i_K = ctypes.c_int(K)
    i_Kap = ctypes.c_int(A.stride(0))  # Kap
    i_M = ctypes.c_int(M)
    i_Bx = ctypes.c_int(1)
    i_By = ctypes.c_int(1)
    p_A = ctypes.c_ulonglong(A.data_ptr())
    p_B = ctypes.c_ulonglong(B.data_ptr())
    p_BIAS = ctypes.c_ulonglong(0)
    p_glbl = ctypes.c_ulonglong(glbl.data_ptr())
    p_cntr = ctypes.c_ulonglong(cntr.data_ptr())
    p_C = ctypes.c_ulonglong(C.data_ptr())
    i_cu = ctypes.c_int(CuCount)
    import sys
    print("PTRDBG A=%#x B=%#x glbl=%#x cntr=%#x C=%#x" %
          (A.data_ptr(), B.data_ptr(), glbl.data_ptr(), cntr.data_ptr(),
           C.data_ptr()), file=sys.stderr, flush=True)
    fields = [i_actlN, i_K, i_Kap, i_M, i_Bx, i_By, p_A, p_B, p_BIAS,
              p_glbl, p_cntr, p_C, i_cu]
    holders = []
    params = []
    for f in fields:
        holders.append(ctypes.cast(ctypes.byref(f), ctypes.c_void_p))
    params_arr = (ctypes.c_void_p * len(holders))(*holders)

    stream = torch.cuda.current_stream().cuda_stream
    # wvSplitKrc's descriptor declares group_segment_fixed_size = 163840;
    # the LDS tile staging (global_load_lds/ds_write_b128/ds_read) drops
    # every access without it. The real runtime fills this field from the
    # kernel descriptor; a hand-rolled launch must pass it explicitly.
    # Use the real vllm wrapper: parameters, persistent glbl/cntr and the
    # variant selection are the engine's own, eliminating hand-rolled
    # launch guesswork.
    from vllm import _custom_ops as vops
    got_t = vops.wvSplitKrc(A, B, CuCount)
    torch.cuda.synchronize()
    C.copy_(got_t.to(C.dtype).T.reshape(N, M))

    ref = (A.float() @ B.float().T)  # (M,N)
    got = C.float().T                # kernel writes out {M? N,M}
    nan_count = int(torch.isnan(got).sum())
    maxdiff = (got - ref).abs().max().item() if nan_count == 0 else float("nan")
    result = {
        "schema": "amdgpu-sim.exact-wvsplitkrc.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "M": M, "N": N, "K": K, "CuCount": CuCount,
        "nan_count": nan_count,
        "maxdiff": maxdiff,
        "match": nan_count == 0 and torch.allclose(got, ref, atol=0.5, rtol=0.5),
        "sample_got": [round(v, 2) for v in got.flatten()[:6].tolist()],
        "sample_ref": [round(v, 2) for v in ref.flatten()[:6].tolist()],
        "gem5": os.environ.get("SAGR_MANAGED_GEM5", "<unset>"),
    }
    torch.save({"got": got.cpu(), "ref": ref.cpu(), "A": A.cpu(), "B": B.cpu()},
               str(out_dir / "full.pt"))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
