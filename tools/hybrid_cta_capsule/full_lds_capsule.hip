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
