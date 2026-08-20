__global__ void mfma_agpr_test(float* out) {
    const float* p = out;
    __asm__ volatile(
        "v_mov_b32_e32 v16, 0x3f803f80\n\t"
        "v_mov_b32_e32 v17, 0x3f803f80\n\t"
        "v_mov_b32_e32 v18, 0x3f803f80\n\t"
        "v_mov_b32_e32 v19, 0x3f803f80\n\t"
        "v_mov_b32_e32 v20, 0x3f803f80\n\t"
        "v_mov_b32_e32 v21, 0x3f803f80\n\t"
        "v_mov_b32_e32 v22, 0x3f803f80\n\t"
        "v_mov_b32_e32 v23, 0x3f803f80\n\t"
        "v_mov_b32_e32 v4, 0x3f800000\n\t"
        "v_mov_b32_e32 v5, 0x40000000\n\t"
        "v_mov_b32_e32 v6, 0x40400000\n\t"
        "v_mov_b32_e32 v7, 0x3fc00000\n\t"
        "v_mfma_f32_32x32x16_bf16 a[0:15], v[16:19], v[20:23], 0\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_mfma_f32_32x32x16_bf16 a[0:15], v[16:19], v[20:23], a[0:15]\n\t"
        "s_nop 0\n\t"
        "s_nop 0\n\t"
        "v_mov_b32_e32 v9, 0x42c80000\n\t"
        "v_accvgpr_write_b32 a8, v9\n\t"
        "v_accvgpr_read_b32 v10, a8\n\t"
        "v_pk_mul_f32 v[8:9], v[4:5], v[6:7]\n\t"
        "v_accvgpr_read_b32 v0, a0\n\t"
        "v_accvgpr_read_b32 v1, a1\n\t"
        "v_mbcnt_lo_u32_b32 v11, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v11, -1, v11\n\t"
        // ticket: v30 = atomic_add(out[1024], 1) -> unique lane id, order random
        // ticket: every lane atomically adds 1 to the same LDS word; the
        // returned old value is a unique per-lane id independent of v_mbcnt
        "v_mov_b32_e32 v2, 1\n\t"
        "v_mov_b32_e32 v31, 60\n\t"
        "ds_add_rtn_u32 v30, v31, v2\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        // per-lane block byte base: ticket*32
        "v_lshlrev_b32_e32 v31, 5, v30\n\t"
        "global_store_dword v31, v11, %0\n\t"
        "v_add_u32_e32 v13, 4, v31\n\t"
        "global_store_dword v13, v0, %0\n\t"
        "v_add_u32_e32 v14, 8, v31\n\t"
        "global_store_dword v14, v1, %0\n\t"
        "v_add_u32_e32 v15, 12, v31\n\t"
        "global_store_dword v15, v10, %0\n\t"
        "v_add_u32_e32 v17, 16, v31\n\t"
        "global_store_dword v17, v8, %0\n\t"
        :
        : "s"(p)
        : "v0","v1","v4","v5","v6","v7","v8","v9","v10","v11","v13","v14","v15","v16",
          "v17","v18","v19","v20","v21","v22","v23","v28","v29","v30","v31",
          "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","a12","a13","a14","a15",
          "memory");
}
