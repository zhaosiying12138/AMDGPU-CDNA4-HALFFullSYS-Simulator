/* Corrected discriminator: v_cndmask_b32_e64 with a saveexec-written
 * scalar-pair condition, with exec predication modeled exactly.
 *
 * tools/qwen35_cndmask_e64_capsule.py claims that after
 *     s_and_saveexec_b64 s[2:3], vcc
 * the immediately following v_cndmask_b32_e64 reads a vcc-shaped condition.
 * gem5 instrumentation disproves that: the condition read is the old exec
 * (all ones).  The observable p3 mismatch in that capsule has two other
 * causes, both in the capsule itself:
 *   (1) it expects lanes masked off by saveexec's new exec to update dst,
 *       which hardware never does (VALU writes only active lanes), and
 *   (2) its raw-dump stores an overlapping qword per lane at dword stride,
 *       so 63 of the 128 words it checks are never written.
 *
 * This probe re-runs the same patterns with hardware-exact expectations and
 * a correctly addressed raw dump:
 *   out[pattern * 64 + lane] for patterns 0..4, plus out[320+2*lane..+1] raw.
 *     0: condition s[2:3]=0x0000FFFF           -> lanes 0..15  src1
 *     1: condition s[2:3]=all-ones             -> every lane    src1
 *     2: condition s[2:3]=0xFFFF0000           -> lanes 16..31 src1
 *     3: s_and_saveexec_b64 s[2:3], vcc (vcc=lanes<32, old exec all-ones);
 *        cndmask runs under the NEW exec (lanes<32): active lanes take the
 *        true path (old-exec condition, all ones); the store itself is
 *        exec-masked, so lanes 32..63 do not store and the buffer keeps 0
 *     4: s_mov_b64 exec, s[2:3] restores exec to the old exec (all ones);
 *        the SAME cndmask e64 with the SAME s[2:3] now runs on every lane
 *        -> every lane src1 (this is the capsule author's intended claim)
 *   raw: s[2:3] moved to v[12:13] and stored as compact qwords at
 *        out+4*(320+2*lane) -> all 128 words 0xffffffff.
 *
 * src0 = 1000 + lane*10 (v5), src1 = 2000 + lane (v6); dst = cond ? src1 : src0.
 */

#ifndef __GLOBAL_DEFINED
#define __GLOBAL_DEFINED
#define __global__ __attribute__((amdgpu_kernel))
#endif

__global__ void cndmask_probe2_test(unsigned int* out) {
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
        "v_add_u32_e32 v8, 192, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dword v8, v7, %0\n\t"

        "s_mov_b64 exec, s[2:3]\n\t"
        ".long 0xd1000007, 0x000a0d05\n\t"
        "v_add_u32_e32 v8, 256, v20\n\t"
        "v_lshlrev_b32_e32 v8, 2, v8\n\t"
        "global_store_dword v8, v7, %0\n\t"

        /* raw s[2:3] (the saveexec destination): compact qword per lane at
           word offset 320 + 2*lane so every checked word is actually
           written exactly once. */
        "v_mov_b32_e32 v12, 0\n\t"
        "v_mov_b32_e32 v13, 0\n\t"
        "s_mov_b32 s6, s2\n\t"
        "s_mov_b32 s7, s3\n\t"
        "v_mov_b32_e32 v12, s6\n\t"
        "v_mov_b32_e32 v13, s7\n\t"
        "v_add_u32_e32 v9, v20, v20\n\t"
        "v_add_u32_e32 v9, 320, v9\n\t"
        "v_lshlrev_b32_e32 v9, 2, v9\n\t"
        "global_store_dwordx2 v9, v[12:13], %0\n\t"
        :
        : "s"(p)
        : "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v20",
          "s2", "s3", "s6", "s7", "vcc", "memory");
}
