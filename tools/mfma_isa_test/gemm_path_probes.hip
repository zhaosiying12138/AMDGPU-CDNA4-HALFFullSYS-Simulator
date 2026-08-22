/* ISA-level discriminators for the flydsl gemm4 zero-output defect.
 *
 * One wavefront (64 lanes), ONE pointer argument (buf) holding:
 *   dwords [0,2048)     : results out (see per-section layout below)
 *   dwords [2048,6144)  : T2 pattern (1.0 + i*0.25f)
 *   dwords [6144,6146)  : T4 atomic target (1.0f, 1.0f)
 * Sections:
 *  T1 mfma     v_mfma_f32_16x16x32_bf16 (A=B=1.0 bf16, C=0) -> 32.0f
 *              everywhere; dst in VGPRs and in AGPRs (accvgpr_read back).
 *  T2 mubufdma buffer_load_dwordx4 ... offen sc0 lds (rsrc word2=-1,
 *              word3=0x27000 exactly like the hgemm kernel) into LDS at
 *              m0=0x1000, lane slot = lane*16; read back via ds_read_b128.
 *  T3 ds       ds_write_b32/ds_read_b32 roundtrip at 0x3000 + lane*4.
 *  T4 atom     global_atomic_pk_add_bf16 of (2.0,2.0)bf16 onto the
 *              (1.0,1.0)bf16x2 target from all 64 lanes -> 129.0 x2.
 * out layout per lane: [0..3]=mfma VGPR dst, [4]=AGPR dst0, [5]=ds value;
 * out[512+lane*4..+3] = mubuf dma readback.
 */
typedef unsigned u32;

__attribute__((amdgpu_kernel)) void gemm_path_probe(float* buf)
{
    __attribute__((address_space(3))) static volatile unsigned lds_pad[8192];
    lds_pad[0] = 0;
        asm volatile(
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"

        /* s[12:13] = buf (out base) */
        "s_mov_b64 s[12:13], s[4:5]\n\t"

        /* ---------------- T1: MFMA 16x16x32_bf16 all-ones --------- */
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
        "v_mfma_f32_16x16x32_bf16 a[0:3], v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_accvgpr_read_b32 v44, a0\n\t"
        "v_accvgpr_read_b32 v45, a1\n\t"

        /* ---------------- T3: ds write/read roundtrip -------------- */
        "v_lshlrev_b32_e32 v53, 2, v11\n\t"
        "v_mov_b32_e32 v36, 0x42f6e979\n\t"          /* 123.456f */
        "ds_write_b32 v53, v36 offset:0x3000\n\t"
        "ds_read_b32 v38, v53 offset:0x3000\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"

        /* ---------------- T2: MUBUF LDS-DMA ------------------------ */
        /* rsrc s[8:11] = {pat_lo, pat_hi, -1, 0x27000}; pat = buf+0x2000 */
        "s_mov_b64 s[8:9], s[4:5]\n\t"
        "s_add_u32 s8, s8, 0x2000\n\t"
        "s_addc_u32 s9, s9, 0\n\t"
        "s_mov_b32 s10, -1\n\t"
        "s_mov_b32 s11, 0x27000\n\t"
        "s_mov_b32 m0, 0x1000\n\t"
        "v_lshlrev_b32_e32 v24, 4, v11\n\t"      /* lane*16 = voff  */
        "buffer_load_dwordx4 v24, s[8:11], 0 offen sc0 lds\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "ds_read_b128 v[32:35], v24 offset:0x1000\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"

        /* ---------------- T4: atomic pk_add_bf16 ------------------- */
        /* atom = buf + 0x6000 */
        "s_add_u32 s14, s12, 0x6000\n\t"
        "s_addc_u32 s15, s13, 0\n\t"
        "v_mov_b64_e32 v[2:3], s[14:15]\n\t"
        "v_mov_b32_e32 v40, 0x40004000\n\t"          /* (2.0,2.0) bf16 */
        "global_atomic_pk_add_bf16 v[2:3], v40, off\n\t"
        "s_waitcnt vmcnt(0)\n\t"

        /* ---------------- stores (saddr base s[12:13]) ------------- */
        "v_lshlrev_b32_e32 v48, 3, v11\n\t"
        "global_store_dwordx4 v48, v[0:3], s[12:13]\n\t"
        "v_add_u32_e32 v51, 4, v48\n\t"
        "global_store_dword v51, v44, s[12:13]\n\t"
        "v_add_u32_e32 v52, 5, v48\n\t"
        "global_store_dword v52, v38, s[12:13]\n\t"
        "v_lshlrev_b32_e32 v54, 2, v11\n\t"
        "v_add_u32_e32 v55, 2048, v54\n\t"
        "global_store_dwordx4 v55, v[32:35], s[12:13]\n\t"
        :
        :
        : "v0", "v1", "v2", "v3", "v11", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v32", "v33", "v34", "v35",
          "v36", "v38", "v40", "v44", "v45", "v48", "v51", "v52", "v53",
          "v54", "v55", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
          "s15", "vcc", "a0", "a1", "a2", "a3", "memory");
}
