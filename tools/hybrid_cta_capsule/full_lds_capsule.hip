#include <hip/hip_runtime.h>
// Full-LDS serialization capsule.  Slots are addressed with the absolute
// work-item id so any block size works; the value round-trips through LDS
// across an s_barrier, which is the semantics the vLLM 0.8B chunked-GDN
// kernel depends on.
//
//   full_lds_a: __shared__ 152 KB -- two WGs never co-fit a 160 KB CU LDS,
//               forcing serialized occupancy (the vLLM chunked-GDN shape).
//   full_lds_b: __shared__ 64 KB  -- comfortable co-residency, the control.
// Oracle: out[absid] = wg*77777 + (absid - wg*blockDim.x).

__global__ void full_lds_a(float* out) {
    __shared__ float tile[155648 / 4];
    const unsigned wg = __builtin_amdgcn_workgroup_id_x();
    const unsigned absid = (__builtin_amdgcn_workgroup_id_x() * blockDim.x + __builtin_amdgcn_workitem_id_x());
    const unsigned rel = absid - wg * blockDim.x;
    tile[rel] = (float)(wg * 77777 + rel);
    __builtin_amdgcn_s_barrier();
    out[absid] = tile[rel];
}

__global__ void full_lds_b(float* out) {
    __shared__ float tile[65536 / 4];
    const unsigned wg = __builtin_amdgcn_workgroup_id_x();
    const unsigned absid = (__builtin_amdgcn_workgroup_id_x() * blockDim.x + __builtin_amdgcn_workitem_id_x());
    const unsigned rel = absid - wg * blockDim.x;
    tile[rel] = (float)(wg * 77777 + rel);
    __builtin_amdgcn_s_barrier();
    out[absid] = tile[rel];
}

// C: exact failing geometry of the vLLM chunked-GDN kernel family:
//    __shared__ == 163840 B (the CU LDS capacity, exactly full) and a 2D
//    block (64,4).  Occupancy is one WG per CU by construction.
__global__ void full_lds_c(float* out) {
    __shared__ float tile[163840 / 4];
    const unsigned wg = blockIdx.x;
    const unsigned rel = threadIdx.y * blockDim.x + threadIdx.x;
    tile[rel] = (float)(wg * 77777 + rel);
    __builtin_amdgcn_s_barrier();
    out[wg * (blockDim.x * blockDim.y) + rel] = tile[rel];
}

// D: >64KB LDS offset addressing (the wvSplitKrc s[] staging reaches
// ~97KB into the workgroup's group segment; LDS offsets beyond 16 bits
// must still resolve exactly).
__global__ void full_lds_d(float* out) {
    __shared__ float stg[66560 / 4];      // low partition, like wvSplitKrc stg
    __shared__ float s[97280 / 4];        // high partition, base ~66.5KB
    const unsigned wg = blockIdx.x;
    const unsigned rel = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned hi_base = 66560 / 4 + 80000 / 4;  // ~ +80KB into segment
    s[hi_base - 66560 / 4 + rel] = (float)(wg * 77777 + rel);  // ~97KB offset
    stg[rel] = (float)(wg * 77777 + rel);                       // low offset
    __builtin_amdgcn_s_barrier();
    out[wg * (blockDim.x * blockDim.y) + rel] =
        s[hi_base - 66560 / 4 + rel] + stg[rel] - (float)(wg * 77777 + rel);
}
