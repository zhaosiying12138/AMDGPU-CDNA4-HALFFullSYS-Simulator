/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
 * Full license text: LICENSE at the repository root.
 */

#include <hip/hip_runtime.h>
/* Minimal v_mfma_f32_16x16x32 probe pair (the wvSplitKrc inner op family).
 *
 * The accumulator operands are declared with the "a" register class so the
 * compiler allocates real AGPRs and the code object carries a truthful
 * num_agpr symbol -- the raw-asm variant hard-coding a[0:3] compiled with
 * num_agpr == 0, which no allocator can serve.  Accumulator results are
 * read back with v_accvgpr_read of the group's first register (MFMA C/D
 * operands must share a register file, so a VGPR-destination MFMA cannot
 * source an AGPR accumulator directly).  The build asserts the group is
 * allocated at a[0:3]; a future shift fails the capsule loudly instead of
 * reading a stale accumulator.
 *
 * mfma1632_probe (bf16): A=B=1.0 -> every VGPR-dst and AGPR-dst lane is
 *   exactly 32.0f; stores buf[lane] (VGPR result) and buf[256+lane]
 *   (AGPR result).
 * mfma_chain (f16): two chained MFMAs accumulate into the same AGPR group
 *   (first C=0, second C=acc) -> 64.0f; stores buf[lane]. */

__global__ void mfma_chain(float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    const float* p = out;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile(
        "s_mov_b64 s[12:13], s[0:1]\n\t"
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
        "v_mfma_f32_16x16x32_f16 %0, v[16:19], v[20:23], %0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_accvgpr_read_b32 v44, a0\n\t"
        "v_mov_b32_e32 v48, s12\n\t"
        "v_mov_b32_e32 v49, s13\n\t"
        "v_lshlrev_b32_e32 v50, 2, v11\n\t"
        "v_add_u32_e32 v50, v50, v48\n\t"
        "v_mov_b32_e32 v51, v49\n\t"
        "global_store_dword v[50:51], v44, off\n\t"
        "s_endpgm\n\t"
        : "=a"(acc)
        : "s"(p)
        : "v11","v16","v17","v18","v19",
          "v20","v21","v22","v23","v44",
          "v48","v49","v50","v51","memory");
}

__global__ void mfma1632_probe(float* out) {
    __attribute__((address_space(3))) static volatile unsigned lds_pad[64];
    lds_pad[0] = 0;
    const float* p = out;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    asm volatile(
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"
        "v_mov_b32_e32 v16, 0x3f803f80\n\t"
        "v_mov_b32_e32 v17, 0x3f803f80\n\t"
        "v_mov_b32_e32 v18, 0x3f803f80\n\t"
        "v_mov_b32_e32 v19, 0x3f803f80\n\t"
        "v_mov_b32_e32 v20, 0x3f803f80\n\t"
        "v_mov_b32_e32 v21, 0x3f803f80\n\t"
        "v_mov_b32_e32 v22, 0x3f803f80\n\t"
        "v_mov_b32_e32 v23, 0x3f803f80\n\t"
        "v_mfma_f32_16x16x32_bf16 v[0:3], v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_mfma_f32_16x16x32_bf16 %0, v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_accvgpr_read_b32 v44, a0\n\t"
        "v_lshlrev_b32_e32 v50, 2, v11\n\t"
        "global_store_dword v50, v0, %1\n\t"
        "global_store_dword v50, v44, %1 offset:1024\n\t"
        "s_endpgm\n\t"
        : "=a"(acc)
        : "s"(p)
        : "v0","v1","v2","v3","v11",
          "v16","v17","v18","v19","v20","v21","v22","v23",
          "v44","v50","memory");
}
