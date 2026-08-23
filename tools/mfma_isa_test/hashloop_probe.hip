/* Discriminator: hash-probe loop -> s_waitcnt vmcnt -> 16-bit multiply chain
 *                  -> pointer select -> final global loads.
 *
 * The vLLM TP1 generate-phase fault (lane v11, dispatch ~1870, kernel 32)
 * is a JIT hash-table probe kernel whose decoded window is:
 *
 *     global_load_dword v1, v7, s[24:25] offset:14   // entry, UNALIGNED (+14)
 *     global_load_ushort v2, v7, s[2:3]              // header ushort at +12
 *     v_lshlrev_b32_e32 v6, 2, v0
 *     s_waitcnt vmcnt(1)
 *     v_lshrrev_b32_e32 v3, 16, v1
 *     v_and_b32_e32   v1, 0xffff, v1
 *     v_mul_lo_u32    v1, v1, v3
 *     s_waitcnt vmcnt(0)
 *     v_mul_lo_u32    v1, v1, v2
 *     v_add_u32_e32   v8, v1, v0
 *     s_cbranch_scc1  <back edge>
 *     ... exit path (cap = s15 = 50 here):
 *     v_subrev_u32_e32   v4, s15, v0        // mirror = idx - cap
 *     v_lshlrev_b32_e32  v4, 2, v4
 *     v_mov_b32_e32      v5, v7             // 0
 *     v_lshl_add_u64 v[2:3], s[0:1], 0, v[6:7]      // A + idx*4
 *     v_lshl_add_u64 v[4:5], s[4:5], 0, v[4:5]      // B + (idx-cap)*4
 *     v_cndmask_b32_e32  v3, v5, v3, vcc
 *     v_cndmask_b32_e32  v2, v4, v2, vcc
 *     ...
 *     v_lshl_add_u64 v[4:5], v[8:9], 2, s[0:1]      // A + v8*4 (64-bit form)
 *     v_subrev_co_u32_e32 vN, vcc, s15, v8
 *     v_lshl_add_u64 v[N:N+1], v[N:N+1], 2, s[4:5]  // B + (v8-cap)*4
 *     v_cndmask_b32_e32  v4, vN, v4, vcc
 *     global_load_dword  v2, v[2:3], off            // <<< faulted in the lane
 *     global_load_dword  v3, v[4:5], off
 *
 * This kernel mirrors that window instruction-for-instruction with a KNOWN
 * table (3-iteration chain), one wave of 64 lanes, per-lane initial index =
 * lane id, and stores per-lane results:
 *
 *   out[lane*8 + 0] = final primary index v0       (expect lane + 45)
 *   out[lane*8 + 1] = final secondary index v8     (expect lane + 48)
 *   out[lane*8 + 2] = dword through pair-1 pointer (A/B select by v0 vs cap)
 *   out[lane*8 + 3] = dword through pair-2 pointer (A/B select by v8 vs cap)
 *   out[lane*8 + 4] = last raw entry dword         (expect 0x00010001)
 *   out[lane*8 + 5] = header ushort M              (expect 3)
 *   out[lane*8 + 6] = final hash product v1        (expect 3)
 *   out[lane*8 + 7] = iteration count              (expect 3)
 *
 * Table contract (host-initialized): ushort M = 3 at table+12; entries E0 at
 * +14 (0x00050002 -> (2*5)*3 = 30), E1 at +18 (0x00040001 -> (1*4)*3 = 12),
 * E2 at +22 (0x00010001 -> (1*1)*3 = 3); the probe pointer advances 4 bytes
 * per iteration.  Chain: v0 = lane -> +30 -> +42 -> +45; v8 = v0 + 3.
 * cap = 50: index < 50 -> slice A[index]; index >= 50 -> slice B[index-50].
 * A[i] = 0xA0000000 + i, B[i] = 0xB0000000 + i, both 128 entries.
 *
 * hdr_p = table + 12 (the ushort's fixed base, mirroring the lane kernel's
 * precomputed s[2:3] = s[24:25] + 12).
 */

__global__ void hashloop_probe(unsigned long* table_p,
                               unsigned long* hdr_p,
                               unsigned long* bufA_p,
                               unsigned long* bufB_p,
                               unsigned int* out_p) {
    __asm__ volatile(
        /* v20 = lane id */
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"
        "v_mov_b32_e32 v0, v20\n\t"          /* initial probe index = lane */
        "v_mov_b32_e32 v7, 0\n\t"            /* zero vaddr for header loads */
        "v_mov_b32_e32 v10, 0\n\t"           /* iteration counter */
        "s_mov_b32 s16, 3\n\t"               /* trip count */
        "s_mov_b64 s[4:5], %0\n\t"           /* advancing entry pointer */
        "s_mov_b64 s[2:3], %1\n\t"           /* fixed ushort header base */

        /* loop head */
        "0:\n\t"
        "global_load_dword v1, v7, s[4:5] offset:14\n\t"
        "global_load_ushort v2, v7, s[2:3]\n\t"
        "v_lshlrev_b32_e32 v6, 2, v0\n\t"
        "s_waitcnt vmcnt(1)\n\t"
        "v_lshrrev_b32_e32 v3, 16, v1\n\t"
        "v_mov_b32_e32 v18, v1\n\t"          /* park raw entry dword */
        "v_and_b32_e32 v1, 0xffff, v1\n\t"
        "v_mul_lo_u32 v1, v1, v3\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "v_mul_lo_u32 v1, v1, v2\n\t"
        "v_add_u32_e32 v8, v1, v0\n\t"
        "v_mov_b32_e32 v0, v8\n\t"
        "v_add_u32_e32 v10, 1, v10\n\t"
        "s_add_u32 s4, s4, 4\n\t"            /* advance entry pointer lo */
        "s_addc_u32 s5, s5, 0\n\t"           /* advance entry pointer hi */
        "s_sub_i32 s16, s16, 1\n\t"
        "s_cmpk_gt_i32 s16, 0\n\t"
        "s_cbranch_scc1 0b\n\t"

        /* park loop-carried observables before the exit path eats them */
        "v_mov_b32_e32 v19, v2\n\t"          /* header ushort */
        "v_mov_b32_e32 v21, v1\n\t"          /* final hash product */
        /* the real kernel's back edge lands after the v0 = v8 update, so
           v6 (= v0 << 2) is fresh w.r.t. the final index at the head; my
           visible in-loop compute precedes the update, so refresh it here */
        "v_lshlrev_b32_e32 v6, 2, v0\n\t"

        /* exit path: pair 1 (primary index v0) */
        "v_subrev_u32_e32 v4, 50, v0\n\t"
        "v_lshlrev_b32_e32 v4, 2, v4\n\t"
        "v_mov_b32_e32 v5, v7\n\t"
        "v_lshl_add_u64 v[2:3], %2, 0, v[6:7]\n\t"
        "v_lshl_add_u64 v[4:5], %3, 0, v[4:5]\n\t"
        "v_cmp_gt_u32_e32 vcc, 50, v0\n\t"
        "v_cndmask_b32_e32 v3, v5, v3, vcc\n\t"
        "v_cndmask_b32_e32 v2, v4, v2, vcc\n\t"

        /* exit path: pair 2 (secondary index v8, the 64-bit fault form) */
        "v_mov_b32_e32 v9, v7\n\t"
        "v_lshl_add_u64 v[4:5], v[8:9], 2, %2\n\t"
        "v_mov_b32_e32 v13, v7\n\t"
        "v_subrev_co_u32_e32 v12, vcc, 50, v8\n\t"
        "v_lshl_add_u64 v[12:13], v[12:13], 2, %3\n\t"
        "v_cndmask_b32_e32 v5, v13, v5, vcc\n\t"
        "v_cndmask_b32_e32 v4, v12, v4, vcc\n\t"

        "global_load_dword v14, v[2:3], off\n\t"
        "global_load_dword v15, v[4:5], off\n\t"
        "s_waitcnt vmcnt(0)\n\t"

        /* out[lane*10 + k] stores, byte offset = lane*40 + k*4 */
        "v_mul_u32_u24_e32 v17, 40, v20\n\t"
        "global_store_dword v17, v0, %4\n\t"
        "global_store_dword v17, v8, %4 offset:4\n\t"
        "global_store_dword v17, v14, %4 offset:8\n\t"
        "global_store_dword v17, v15, %4 offset:12\n\t"
        "global_store_dword v17, v18, %4 offset:16\n\t"
        "global_store_dword v17, v19, %4 offset:20\n\t"
        "global_store_dword v17, v21, %4 offset:24\n\t"
        "global_store_dword v17, v10, %4 offset:28\n\t"
        /* direct VOP3 integer CLAMP regression checks: the lane's table
           corruption came from v_sub_u32_e64 ... clamp wrapping instead of
           saturating to 0.  Immediates ride in registers: VOP3 has no
           literal field for 2048/-1. */
        "v_mov_b32_e32 v24, 2048\n\t"
        "v_mov_b32_e32 v25, -1\n\t"
        "v_sub_u32_e64 v22, v20, v24 clamp\n\t"       /* lane - 2048 */
        "v_add_u32_e64 v23, v25, v20 clamp\n\t"       /* 0xffffffff + lane */
        "global_store_dword v17, v22, %4 offset:32\n\t"
        "global_store_dword v17, v23, %4 offset:36\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "s"(table_p), "s"(hdr_p), "s"(bufA_p), "s"(bufB_p), "s"(out_p)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v12", "v13", "v14", "v15", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "s2", "s3", "s4",
          "s5", "s16", "vcc", "memory");
}
