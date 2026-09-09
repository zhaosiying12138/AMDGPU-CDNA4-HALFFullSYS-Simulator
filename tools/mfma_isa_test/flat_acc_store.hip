#include <hip/hip_runtime.h>
/* FLAT/global store with AGPR data-source probe: MFMA result lands in
 * a[0:3], then one global_store_dwordx4 reads the accumulator directly as
 * its data operand (the wvSplitKrc epilogue form). All-ones f16 operands
 * make every element 32.0f. */
__global__ void flat_acc_store(float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    const float* p = out;
    unsigned long long addr = (unsigned long long)p;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile(
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"
        "v_mov_b32_e32 v16, 0x3c003c00\n\t"
        "v_mov_b32_e32 v17, 0x3c003c00\n\t"
        "v_mov_b32_e32 v18, 0x3c003c00\n\t"
        "v_mov_b32_e32 v19, 0x3c003c00\n\t"
        "v_mov_b32_e32 v20, 0x3c003c00\n\t"
        "v_mov_b32_e32 v21, 0x3c003c00\n\t"
        "v_mov_b32_e32 v22, 0x3c003c00\n\t"
        "v_mov_b32_e32 v23, 0x3c003c00\n\t"
        "v_mfma_f32_16x16x32_f16 %0, v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_lshlrev_b32_e32 v4, 4, v11\n\t"
        "v_add_co_u32_e32 v2, vcc, %2, v4\n\t"
        "v_addc_co_u32_e32 v3, vcc, 0, %3, vcc\n\t"
        "global_store_dwordx4 v[2:3], %0, off\n\t"
        "s_endpgm\n\t"
        : "=a"(acc)
        : "s"(p), "v"((unsigned)addr), "v"((unsigned)(addr >> 32))
        : "v2","v3","v4","v11","v16","v17","v18","v19",
          "v20","v21","v22","v23","memory");
}
