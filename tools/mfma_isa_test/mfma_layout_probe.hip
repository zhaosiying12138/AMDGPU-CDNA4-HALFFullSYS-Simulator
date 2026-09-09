#include <hip/hip_runtime.h>
/* MFMA 16x16x32_f16 element-layout probe: structured A/B, one MFMA,
 * every lane stores its four accumulator elements. The host computes the
 * expected per-lane elements from the ISA's documented output layout. */
__global__ void mfma_layout_probe(const float* abuf, const float* bbuf,
                                  float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    // A operand: 4 VGPRs = 16x32 f16 packed; loaded as 16 dwordx4? minimal:
    // host pre-packs per-lane A/B operand registers in abuf/bbuf so the
    // kernel only copies them in (avoids replicating the wv gather chain).
    // abuf: 64 lanes x 4 dwords (A), bbuf: 64 lanes x 4 dwords (B)
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile(
        "global_load_dwordx4 v[16:19], v[4:5], off\n\t"
        "global_load_dwordx4 v[20:23], v[6:7], off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x32_f16 %%0, v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_accvgpr_read_b32 v40, %%0\n\t"
        "s_endpgm\n\t"
        : "=a"(acc)
        :
        : "memory");
    // placeholder body; operand address setup TODO next session
    out[0] = 0.f;
}
