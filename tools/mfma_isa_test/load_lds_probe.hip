#include <hip/hip_runtime.h>
/* global_load_lds_dwordx4 -> LDS -> ds_read_b128 round-trip probe.
 * Lane L loads in[L*4..L*4+3] through the load-to-LDS path, then reads
 * back LDS[L*16 .. +15] and stores it. If the LDS destination semantics
 * differ by a base/stride the host sees the shifted pattern; if the data
 * never reaches LDS the output stays zero. */
__global__ void load_lds_probe(const float* in, float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    unsigned long long addr = (unsigned long long)in;
    unsigned long long oaddr = (unsigned long long)out;
    asm volatile(
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"
        "v_lshlrev_b32_e32 v4, 4, v11\n\t"          // LDS byte offset = lane*16
        "v_add_co_u32_e32 v2, vcc, %0, v4\n\t"      // in + lane*16 bytes
        "v_addc_co_u32_e32 v3, vcc, 0, %1, vcc\n\t"
        "s_mov_b32 m0, 0\n\t"
        "global_load_lds_dwordx4 v[2:3], off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        "ds_read_b128 v[8:11], v4\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_lshlrev_b32_e32 v5, 4, v11\n\t"
        "v_add_co_u32_e32 v6, vcc, %2, v5\n\t"
        "v_addc_co_u32_e32 v7, vcc, 0, %3, vcc\n\t"
        "global_store_dwordx4 v[6:7], v[8:11], off\n\t"
        "s_endpgm\n\t"
        :
        : "v"((unsigned)addr), "v"((unsigned)(addr >> 32)),
          "v"((unsigned)oaddr), "v"((unsigned)(oaddr >> 32))
        : "v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","vcc","memory");
    out[0] = 0.f; // unreachable side effect to keep the kernel non-empty
}
