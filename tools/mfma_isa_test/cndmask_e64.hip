/* Discriminator: v_cndmask_b32_e64 with an explicit SCALAR PAIR condition.
 *
 * The vLLM warmup crash kernel (chunk_gated_delta_rule_fwd_kernel_h) computes
 * its buffer_store_dwordx4 offsets through exactly this form:
 *
 *     v_cndmask_b32_e64 v94, v111, v14, s[2:3]
 *     buffer_store_dwordx4 v[2:5], v94, s[12:15], 0 offen
 *
 * with the crash dump showing s2=s3=0xffffffff.  One wavefront; out layout:
 *   out[pattern * 64 + lane] (patterns 0..3) plus out[256+2*lane..] = raw s[2:3] after saveexec; patterns:
 *     0: condition s2=0x0000FFFF (lanes 0..15 -> src1)
 *     1: condition all-ones (the crash state; every lane -> src1)
 *     2: condition s2=0xFFFF0000 (lanes 16..31 -> src1)
 *     3: s_and_saveexec_b64 s[2:3], vcc with old exec all-ones
 *   src1 = 1000 + lane*10 (v5), src2 = 2000 + lane (v6).
 */

__global__ void cndmask_e64_test(unsigned int* out) {
    const unsigned int* p = out;
    __asm__ volatile(
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"
        "v_mul_u32_u24_e32 v4, 10, v20\n\t"
        "v_add_u32_e32 v5, 1000, v4\n\t"
        "v_add_u32_e32 v6, 2000, v20\n\t"

        "s_mov_b32 s2, 0xffff\n\t"
        "s_mov_b32 s3, 0\n\t"
        ".long 0xd1000007, 0x000a0d05\n\t"
        "v_lshlrev_b32_e32 v8, 2, v20\n\t"
        "global_store_dword v8, v7, %0\n\t"

        "s_mov_b32 s2, -1\n\t"
        "s_mov_b32 s3, -1\n\t"
        ".long 0xd1000007, 0x000a0d05\n\t"
        "v_add_u32_e32 v8, 64, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dword v8, v7, %0\n\t"

        "s_mov_b32 s2, -65536\n\t"
        "s_mov_b32 s3, 0\n\t"
        ".long 0xd1000007, 0x000a0d05\n\t"
        "v_add_u32_e32 v8, 128, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dword v8, v7, %0\n\t"

        "v_cmp_gt_u32_e32 vcc, 32, v20\n\t"
        "s_and_saveexec_b64 s[2:3], vcc\n\t"
        ".long 0xd1000007, 0x000a0d05\n\t"
        "s_mov_b64 exec, s[2:3]\n\t"
        /* pattern 4: dump raw s[2:3] (the saveexec destination) so the
           destination semantics are directly observable, not inferred
           through the cndmask. */
        "v_mov_b32_e32 v12, 0\n\t"
        "v_mov_b32_e32 v13, 0\n\t"
        "s_mov_b32 s6, s2\n\t"
        "s_mov_b32 s7, s3\n\t"
        "v_mov_b32_e32 v12, s6\n\t"
        "v_mov_b32_e32 v13, s7\n\t"
        "v_add_u32_e32 v8, 256, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dwordx2 v8, v[12:13], %0\n\t"
        "v_add_u32_e32 v8, 192, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dword v8, v7, %0\n\t"
        :
        : "s"(p)
        : "v4", "v5", "v6", "v7", "v8", "v20",
          "s2", "s3", "vcc", "memory");
}
