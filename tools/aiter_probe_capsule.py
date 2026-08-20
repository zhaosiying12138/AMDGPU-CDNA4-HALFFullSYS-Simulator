import torch
import aiter
from aiter.tuned_gemm import get_GEMM_A16W16_config
from aiter.ops.flydsl import gemm_kernels as gk
cfg = get_GEMM_A16W16_config(1, 8192, 1024, False, "torch.bfloat16", "torch.bfloat16")
params = gk.get_flydsl_splitk_hgemm_kernel_params(cfg["kernelName"])
print("kernel params:", params, flush=True)
print("get_rocm_arch:", gk.get_rocm_arch(), flush=True)
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP
print("smem cap:", SMEM_CAPACITY_MAP.get(gk.get_rocm_arch()), "map keys:", list(SMEM_CAPACITY_MAP), flush=True)
tiled = {
    "TILE_M": params["tile_m"], "TILE_N": params["tile_n"], "TILE_K": params["tile_k"],
    "STAGES": params.get("stages", params.get("stage", 2)),
    "SPLIT_K": params["split_k"],
    "BLOCK_M_WARPS": params["block_m_warps"], "BLOCK_N_WARPS": params["block_n_warps"],
    "BLOCK_K_WARPS": params["block_k_warps"], "B_TO_LDS": params.get("b_to_lds", True),
}
print("filter verdict:", gk.selection_filter(1, 8192, 1024, tiled), flush=True)
