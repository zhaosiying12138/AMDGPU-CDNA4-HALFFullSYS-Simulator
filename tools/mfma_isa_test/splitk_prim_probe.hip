#include <hip/hip_runtime.h>

/* Primitive bisect probe for the wvSplitKrc (vLLM skinny split-K) NaN.
 * buf layout (floats):
 *   [0]    S1 atomicAdd float sum (64 lanes x +1.0f onto 0.0f) -> 64.0f
 *   [1]    S2 __hip_atomic_store(float2) roundtrip -> .x, next word .y
 *   [2]    S2 y
 *   [3]    S3 __builtin_nontemporal_load(float) of pattern -> value
 *   [4..7] S4 __builtin_amdgcn_global_load_lds readback (lane dword)
 *   [8]    S5 signal-fence + plain store ordering marker
 */
__global__ void splitk_prim_probe(float* buf) {
    __shared__ float lds_sink[256];
    const int lane = threadIdx.x & 63;
    const float* pat = buf + 1024;   /* pattern area, prefilled by host */

    /* S1: relaxed atomic float add */
    if (lane == 0)
        __hip_atomic_store(buf, 0.0f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __builtin_amdgcn_sched_barrier(0);
    if (lane == 0)
        __hip_atomic_store(buf, 0.0f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    float one = 1.0f;
    atomicAdd(buf, one);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    /* S2: 64-bit hip atomic store of a float2 */
    if (lane == 0) {
        float2 v = {11.0f, 22.0f};
        __hip_atomic_store((float2*)&buf[1], v, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    }

    /* S3: non-temporal scalar load */
    float nt = __builtin_nontemporal_load(&pat[lane]);

    /* S4: global load directly into LDS, then read back */
    __builtin_amdgcn_global_load_lds((int*)&pat[lane], (int*)&lds_sink[lane],
                                     4, 0, 0);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    float from_lds = lds_sink[lane];

    /* S5: marker */
    float marker = 77.0f;

    __syncthreads();
    /* lane 0 collects: only lane 0 stores verdicts to keep it simple;
     * per-lane values go to the per-lane area. */
    buf[512 + lane] = nt;          /* S3 per-lane */
    buf[640 + lane] = from_lds;    /* S4 per-lane */
    if (lane == 0)
        buf[8] = marker;           /* S5 */
}
