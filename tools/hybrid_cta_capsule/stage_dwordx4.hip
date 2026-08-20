__shared__ char lds_pad[4096];

// Global dwordx4 staging probe shaped like the CK double-buffer loop:
// four k-block tiles are fetched with buffer_load_dwordx4 from global
// memory (interleaved so earlier tiles are NOT hot), consumed by ALU adds
// after s_waitcnt vmcnt, accumulated per lane, stored via ticket layout.
// If early k-blocks' fragments arrive stale, their contributions vanish
// exactly like the CK d256 loss.
__global__ void mfma_agpr_test(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"
        // src pointer = p + 4096 (skip the output area) via s0 pair from %0
        "s_load_dwordx2 s[6:7], %0, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_mov_b32_e32 v2, s6\n\t"
        "v_mov_b32_e32 v3, s7\n\t"
        "v_mov_b32_e32 v4, 1\n\t"
        // ticket for output addressing
        "v_mov_b32_e32 v25, 8\n\t"
        "v_lshlrev_b32_e32 v25, 6, v25\n\t"
        "ds_add_rtn_u32 v30, v25, v4\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        // per-lane global byte offset: lane*16, four tiles at +0/+1024/+2048/+3072
        "v_lshlrev_b32_e32 v5, 4, v20\n\t"
        // four dwordx4 loads into v[8:11], v[12:15], v[16:19], v[24:27]
        "v_add_u32_e32 v6, v5, v2\n\t"
        "global_load_dwordx4 v[8:11], v[6:7], off\n\t"
        "v_mov_b32_e32 v28, 1024\n\t"
        "v_add_u32_e64 v6, v6, v28\n\t"
        "global_load_dwordx4 v[12:15], v[6:7], off\n\t"
        "v_mov_b32_e32 v28, 2048\n\t"
        "v_add_u32_e32 v6, v5, v2\n\t"
        "v_add_u32_e64 v6, v6, v28\n\t"
        "global_load_dwordx4 v[16:19], v[6:7], off\n\t"
        "v_mov_b32_e32 v28, 3072\n\t"
        "v_add_u32_e64 v6, v6, v28\n\t"
        "global_load_dwordx4 v[24:27], v[6:7], off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        // accumulate: tile0 through tile3 sums (each dword)
        "v_add_u32_e32 v8, v8, v12\n\t"
        "v_add_u32_e32 v9, v9, v13\n\t"
        "v_add_u32_e32 v10, v10, v14\n\t"
        "v_add_u32_e32 v11, v11, v15\n\t"
        "v_add_u32_e32 v8, v8, v16\n\t"
        "v_add_u32_e32 v9, v9, v17\n\t"
        "v_add_u32_e32 v10, v10, v18\n\t"
        "v_add_u32_e32 v11, v11, v19\n\t"
        "v_add_u32_e32 v8, v8, v24\n\t"
        "v_add_u32_e32 v9, v9, v25\n\t"
        "v_add_u32_e32 v10, v10, v26\n\t"
        "v_add_u32_e32 v11, v11, v27\n\t"
        // store four dwords at ticket slots 0..3
        "v_lshlrev_b32_e32 v31, 5, v30\n\t"
        "global_store_dword v31, v8, %0\n\t"
        "v_add_u32_e32 v13, 4, v31\n\t"
        "global_store_dword v13, v9, %0\n\t"
        "v_add_u32_e32 v14, 8, v31\n\t"
        "global_store_dword v14, v10, %0\n\t"
        "v_add_u32_e32 v15, 12, v31\n\t"
        "global_store_dword v15, v11, %0\n\t"
        :
        : "s"(p)
        : "v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",
          "v14","v15","v16","v17","v18","v19","v20","v24","v25","v26","v27",
          "v28","v30","v31","memory");
}
