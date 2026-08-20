// Global dwordx4 staging-path probe, CK double-buffer fetch shape.
//
// Suspect: GLOBAL wide loads (global_load_dwordx4) -> s_waitcnt vmcnt(0)
// -> nearby ALU consumers lose early tiles in the gem5 AMD GPU model,
// which would explain the aiter CK attention kernel losing head dims
// [0:128) (early k-blocks vanish, final k-block exact).
//
// Shape: one workgroup of 256 threads (4x wave64).  Each wave fetches
// four 16B/lane k-block tiles (1024 B apart) with the address math
// interleaved between the wide loads so earlier tiles are not hot,
// waits on vmcnt(0), then consumes the four tiles with 12 v_add_u32
// accumulation adds and stores four dwords at ticket slots taken from
// an LDS ds_add_rtn counter.
//
// Pointer plumbing avoids ALL SGPR inline-asm operands (the previous
// s_load-from-out[0:1] scheme read a garbage base): the C preamble
// writes the kernel-argument pointers into __shared__ memory before
// __syncthreads(), and the asm reads them back with ds_read_b64 at
// fixed LDS offsets.  gfx950 global wide loads have no sbase form, so
// the flat v[addr:addr+1] form is used with addresses built in VGPRs.
//
// Two variants of the same body:
//   stage_imm   - consumers immediately after s_waitcnt vmcnt(0)
//   stage_delay - a 64-deep dependent v_add chain between the waitcnt
//                 and the consumers (tests whether consumer immediacy
//                 vs in-flight data loss is the discriminator)
//
// Input dword at tile t, dword index i (i = lane*4 + k):
//   (0x100000 << t) + i
// so the expected full sum is 0xF00000 + 4*i (i in [0,256)), seen once
// per wave (4 waves -> each valid value exactly 4 times over 256
// tickets x 4 dwords).  A vanished tile subset leaves the bit-mask of
// surviving tiles decodable from the high nibble of the sum.

#include <hip/hip_runtime.h>

struct StageShared {
    volatile unsigned long long base_in;   // LDS +0
    volatile unsigned long long base_out;  // LDS +8
    volatile unsigned ticket;              // LDS +16
    volatile char pad[4096 - 24];
};

__shared__ StageShared sstage;

#define REP16(X) X X X X X X X X X X X X X X X X
#define REP64(X) REP16(REP16(X))

#define STAGE_DELAY_NONE ""

#define STAGE_DELAY_CHAIN \
    "v_mov_b32_e32 v40, 0\n\t" REP64("v_add_u32_e32 v40, 1, v40\n\t")

#define STAGE_CLOBBERS \
    "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12", \
    "v13","v14","v15","v16","v17","v18","v19","v20","v24","v25","v26","v27", \
    "v28","v30","v31","v32","v33","v34","v35","v40","memory"

#define STAGE_KERNEL(NAME, DELAY)                                          \
__global__ void NAME(float* out, const int* inputs) {                      \
    if (threadIdx.x == 0) {                                                \
        sstage.base_in = (unsigned long long)inputs;                       \
        sstage.base_out = (unsigned long long)out;                         \
        sstage.ticket = 0;                                                 \
    }                                                                      \
    __syncthreads();                                                       \
    __asm__ volatile(                                                      \
        "v_mbcnt_lo_u32_b32 v20, -1, 0\n\t"                                \
        "v_mbcnt_hi_u32_b32 v20, -1, v20\n\t"                              \
        /* base pointers from LDS (written by the C preamble) */           \
        "v_mov_b32_e32 v0, 0\n\t"                                          \
        "ds_read_b64 v[6:7], v0\n\t"                                       \
        "ds_read_b64 v[2:3], v0 offset:8\n\t"                              \
        "s_waitcnt lgkmcnt(0)\n\t"                                         \
        /* ticket: in-bounds LDS atomic at +16 */                          \
        "v_mov_b32_e32 v4, 1\n\t"                                          \
        "v_mov_b32_e32 v28, 16\n\t"                                        \
        "ds_add_rtn_u32 v30, v28, v4\n\t"                                  \
        "s_waitcnt lgkmcnt(0)\n\t"                                         \
        /* per-lane byte offset within a tile: lane*16 */                  \
        "v_lshlrev_b32_e32 v5, 4, v20\n\t"                                 \
        /* four interleaved dwordx4 tile fetches; v[34:35] = base+lane*16, \
           tiles at +0/+1024/+2048/+3072 (address math interleaved between \
           the wide loads, CK double-buffer style) */                      \
        "v_mov_b32_e32 v35, v7\n\t"                                        \
        "v_add_u32_e64 v34, v5, v6\n\t"                                    \
        "global_load_dwordx4 v[8:11], v[34:35], off\n\t"                   \
        "v_mov_b32_e32 v28, 1024\n\t"                                      \
        "v_add_u32_e64 v34, v34, v28\n\t"                                  \
        "global_load_dwordx4 v[12:15], v[34:35], off\n\t"                  \
        "v_mov_b32_e32 v28, 2048\n\t"                                      \
        "v_add_u32_e64 v34, v5, v6\n\t"                                    \
        "v_add_u32_e64 v34, v34, v28\n\t"                                  \
        "global_load_dwordx4 v[16:19], v[34:35], off\n\t"                  \
        "v_mov_b32_e32 v28, 3072\n\t"                                      \
        "v_add_u32_e64 v34, v5, v6\n\t"                                    \
        "v_add_u32_e64 v34, v34, v28\n\t"                                  \
        "global_load_dwordx4 v[24:27], v[34:35], off\n\t"                  \
        "s_waitcnt vmcnt(0)\n\t"                                           \
        DELAY                                                              \
        /* consumers: accumulate tiles 1..3 into tile 0 */                 \
        "v_add_u32_e64 v8, v8, v12\n\t"                                    \
        "v_add_u32_e64 v9, v9, v13\n\t"                                    \
        "v_add_u32_e64 v10, v10, v14\n\t"                                  \
        "v_add_u32_e64 v11, v11, v15\n\t"                                  \
        "v_add_u32_e64 v8, v8, v16\n\t"                                    \
        "v_add_u32_e64 v9, v9, v17\n\t"                                    \
        "v_add_u32_e64 v10, v10, v18\n\t"                                  \
        "v_add_u32_e64 v11, v11, v19\n\t"                                  \
        "v_add_u32_e64 v8, v8, v24\n\t"                                    \
        "v_add_u32_e64 v9, v9, v25\n\t"                                    \
        "v_add_u32_e64 v10, v10, v26\n\t"                                  \
        "v_add_u32_e64 v11, v11, v27\n\t"                                  \
        /* store four dwords at ticket slots: out + ticket*32 + {0,4,8,12} */ \
        "v_lshlrev_b32_e32 v31, 5, v30\n\t"                                \
        "v_mov_b32_e32 v33, v3\n\t"                                        \
        "v_add_u32_e64 v32, v31, v2\n\t"                                   \
        "global_store_dword v[32:33], v8, off\n\t"                         \
        "global_store_dword v[32:33], v9, off offset:4\n\t"                \
        "global_store_dword v[32:33], v10, off offset:8\n\t"               \
        "global_store_dword v[32:33], v11, off offset:12\n\t"              \
        :                                                                  \
        :                                                                  \
        : STAGE_CLOBBERS);                                                 \
}

STAGE_KERNEL(stage_imm, STAGE_DELAY_NONE)
STAGE_KERNEL(stage_delay, STAGE_DELAY_CHAIN)
