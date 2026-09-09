#include <hip/hip_runtime.h>
/* global_atomic_add return-value semantics probe: one lane performs eight
 * sequential RMW adds on one address and stores every returned old value.
 * Expected: out[0..7] == 0..7, out[8] == out[9] == 8. */
__global__ void atomic_ret(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 8; ++i)
            out[i] = atomicAdd(&out[8], 1.0f);
        __threadfence();
        out[9] = out[8];
    }
}
