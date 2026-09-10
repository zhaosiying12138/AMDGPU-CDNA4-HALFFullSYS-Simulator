/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
 * Full license text: LICENSE at the repository root.
 */

#include <hip/hip_runtime.h>
/* Two-batch load->wait->MFMA chain probe (immediate-offset loads so the
 * asm needs a single base pointer). All-ones f16 tiles: every element of
 * a[0] must be 64.0f (32 + 32); 32.0f means the second batch was lost
 * between the load writeback and the MFMA consumer. */
__global__ void two_batch_mfma(const float* in, float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    unsigned long long addr = (unsigned long long)in;
    unsigned long long oaddr = (unsigned long long)out;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile(
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"
        "v_lshlrev_b32_e32 v4, 4, v11\n\t"
        "v_add_co_u32_e32 v2, vcc, %1, v4\n\t"
        "v_addc_co_u32_e32 v3, vcc, 0, %2, vcc\n\t"
        "global_load_dwordx4 v[16:19], v[2:3], off\n\t"
        "global_load_dwordx4 v[20:23], v[2:3], off offset:1024\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x32_f16 %0, v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "global_load_dwordx4 v[16:19], v[2:3], off offset:2048\n\t"
        "global_load_dwordx4 v[20:23], v[2:3], off offset:3072\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x32_f16 %0, v[16:19], v[20:23], %0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_accvgpr_read_b32 v40, a0\n\t"
        "v_add_co_u32_e32 v6, vcc, %3, v4\n\t"
        "v_addc_co_u32_e32 v7, vcc, 0, %4, vcc\n\t"
        "global_store_dword v[6:7], v40, off\n\t"
        "s_endpgm\n\t"
        : "=a"(acc)
        : "v"((unsigned)addr), "v"((unsigned)(addr >> 32)),
          "v"((unsigned)oaddr), "v"((unsigned)(oaddr >> 32))
        : "v2","v3","v4","v6","v7","v11","v16","v17","v18","v19",
          "v20","v21","v22","v23","v40","vcc","memory");
}
