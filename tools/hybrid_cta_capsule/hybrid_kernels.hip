__shared__ char lds_pad[1024];

// A: plain data-parallel. out[wg*256 + lane] = wg*4096 + lane (raw bits).
__global__ void plain_dp(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    const unsigned wg = __builtin_amdgcn_workgroup_id_x();
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n\t"
        "v_mov_b32_e32 v1, 256\n\t"
        "v_mul_u32_u24_e32 v2, %1, v1\n\t"
        "v_add_u32_e32 v3, v2, v0\n\t"
        "v_mov_b32_e32 v4, 4096\n\t"
        "v_mul_u32_u24_e32 v5, %1, v4\n\t"
        "v_add_u32_e32 v6, v5, v0\n\t"
        "v_lshlrev_b32_e32 v3, 2, v3\n\t"
        "global_store_dword v3, v6, %0\n\t"
        :
        : "s"(p), "s"(wg)
        : "v0","v1","v2","v3","v4","v5","v6","memory");
}

// B: barrier + cross-wave LDS. WG = 128 threads (2 waves). Every lane
// writes LDS[lane]=wg*100000+lane, s_barrier, then reads LDS[(lane+64)%128]
// and stores the read value at its own slot.
__global__ void barrier_lds(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    const unsigned wg = __builtin_amdgcn_workgroup_id_x();
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n\t"
        "v_and_b32_e64 v1, v0, 63\n\t"           // lane in wave
        "v_mov_b32_e32 v2, 100000\n\t"
        "v_mul_u32_u24_e32 v3, %1, v2\n\t"       // wg*100000
        // global slot = wg*256 + (wave*64 + lane) = wg*256 + raw lane id
        "v_mov_b32_e32 v4, 256\n\t"
        "v_mul_u32_u24_e32 v5, %1, v4\n\t"
        "v_add_u32_e32 v5, v5, v0\n\t"
        // LDS addr: raw lane id * 4
        "v_lshlrev_b32_e32 v6, 2, v0\n\t"
        "v_add_u32_e32 v7, v3, v0\n\t"
        "ds_write_b32 v6, v7\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
        // read sibling: (rawid + 64) & 127
        "v_and_b32_e64 v8, v0, 63\n\t"
        "v_xor_b32_e64 v8, v8, 64\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "ds_read_b32 v9, v8\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_lshlrev_b32_e32 v5, 2, v5\n\t"
        "global_store_dword v5, v9, %0\n\t"
        :
        : "s"(p), "s"(wg)
        : "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","memory");
}

// C: decline trigger. Contains buffer_wbl2 (isMemSync, not endpgm): the
// hybrid screen must decline the whole dispatch to full timing. Output is
// identical in both modes by construction (plain writes).
__global__ void atomic_decline(float* out) {
    *(volatile char*)lds_pad = 0;
    const float* p = out;
    const unsigned wg = __builtin_amdgcn_workgroup_id_x();
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n\t"
        "v_mov_b32_e32 v1, 256\n\t"
        "v_mul_u32_u24_e32 v2, %1, v1\n\t"
        "v_add_u32_e32 v3, v2, v0\n\t"
        "v_mov_b32_e32 v6, 7\n\t"
        "v_lshlrev_b32_e32 v6, 20, v6\n\t"
        "v_add_u32_e32 v6, v6, v0\n\t"
        "v_lshlrev_b32_e32 v3, 2, v3\n\t"
        "buffer_wbl2\n\t"
        "global_store_dword v3, v6, %0\n\t"
        :
        : "s"(p), "s"(wg)
        : "v0","v1","v2","v3","v6","memory");
}
