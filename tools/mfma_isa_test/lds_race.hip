__shared__ char lds_pad[4096];

__global__ void mfma_agpr_test(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    // Cross-wave LDS visibility: wave-0 lanes (mbcnt id <= 63) write
    // LDS[id*4] = id+60; s_barrier; then every lane reads LDS[(id&63)*4]
    // and stores it at its ticket slot.  Expected value = (id&63)+60.
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"
        "v_mov_b32_e32 v2, 1\n\t"
        "v_cmp_le_u32_e64 vcc, v20, 63\n\t"
        "s_and_saveexec_b64 s[4:5], vcc\n\t"
        "v_lshlrev_b32_e32 v23, 2, v20\n\t"
        "v_add_u32_e32 v24, 60, v20\n\t"
        "ds_write_b32 v23, v24\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_or_b64 exec, exec, s[4:5]\n\t"
        "s_barrier\n\t"
        "v_mov_b32_e32 v25, 63\n\t"
        "v_lshlrev_b32_e32 v25, 3, v2\n\t"
        "v_lshlrev_b32_e32 v25, 6, v25\n\t"
        "ds_add_rtn_u32 v30, v25, v2\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_and_b32_e64 v26, v20, 63\n\t"
        "v_lshlrev_b32_e32 v27, 2, v26\n\t"
        "ds_read_b32 v28, v27\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_add_u32_e32 v22, 60, v26\n\t"
        "v_lshlrev_b32_e32 v31, 5, v30\n\t"
        "global_store_dword v31, v28, %0\n\t"
        "v_add_u32_e32 v13, 4, v31\n\t"
        "global_store_dword v13, v22, %0\n\t"
        :
        : "s"(p)
        : "v2","v13","v20","v22","v23","v24","v25","v26","v27","v28","v30","v31",
          "vcc","s4","s5","memory");
}
