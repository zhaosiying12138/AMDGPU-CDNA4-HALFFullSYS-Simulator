__shared__ char lds_pad[1024];

__global__ void mfma_agpr_test(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"
        "v_mov_b32_e32 v2, 3\n\t"
        "v_and_b32_e64 v41, v20, v2\n\t"   // reg & reg
        "v_and_b32_e64 v42, 3, v20\n\t"    // imm in src0 slot
        "v_and_b32_e32 v43, 3, v20\n\t"    // e32 form, imm in src0
        "v_mov_b32_e32 v31, 0\n\t"
        "global_store_dword v31, v41, %0\n\t"
        "v_mov_b32_e32 v32, 4\n\t"
        "global_store_dword v32, v42, %0\n\t"
        "v_mov_b32_e32 v33, 8\n\t"
        "global_store_dword v33, v43, %0\n\t"
        :
        : "s"(p)
        : "v2","v20","v31","v32","v33","v41","v42","v43","memory");
}
