// Device source for the gfx950 LDS-DMA capsule (tools/flat_lds_capsule).
//
// Every kernel launches ONE 64-lane wave per workgroup, fills its whole LDS
// tile with a sentinel, performs one global-to-LDS DMA, and copies the whole
// tile back out. The host compares the entire tile image against an oracle,
// so a wrong destination address, a wrong per-lane slot pitch, a wrong
// sub-dword extension, or lane compaction all show up as a mismatch rather
// than as "it ran".
//
// __builtin_amdgcn_global_load_lds(src_gptr, lds_ptr, size, offset, aux)
// lowers to global_load_lds_{ubyte,ushort,dword,dwordx3,dwordx4} for
// size = 1, 2, 4, 12, 16. The signed sub-dword opcodes
// (global_load_lds_sbyte / _sshort) have no builtin -- LLVM's intrinsic
// carries a size but no signedness -- so they are covered by the gem5 unit
// test only.

#include <hip/hip_runtime.h>

typedef unsigned int u32;

#define WAVE 64
#define SENT 0xC0DEC0DEu
#define TILE_U32 512
#define TILE_BYTES (TILE_U32 * 4)

#define GSRC(p) ((const __attribute__((address_space(1))) u32 *)(p))
#define GLDS(p) ((__attribute__((address_space(3))) u32 *)(p))

#define PROLOGUE                                                            \
  __shared__ u32 tile[TILE_U32];                                            \
  const int tid = threadIdx.x;                                              \
  for (int i = tid; i < TILE_U32; i += WAVE) tile[i] = SENT;                \
  __syncthreads();

#define EPILOGUE                                                            \
  __builtin_amdgcn_s_waitcnt(0);                                            \
  __syncthreads();                                                          \
  for (int i = tid; i < TILE_U32; i += WAVE)                                \
    out[blockIdx.x * TILE_U32 + i] = tile[i];

// Lane L reads 16 bytes at src + L*16 regardless of the transfer width, so
// the same host buffer feeds every case.
#define LANE_SRC (src + blockIdx.x * WAVE * 4 + tid * 4)

#define PROBE(NAME, SIZE, OFFSET, PRED)                                     \
  extern "C" __global__ void NAME(const u32 *__restrict src,                \
                                  u32 *__restrict out) {                    \
    PROLOGUE                                                                \
    if (PRED) {                                                             \
      __builtin_amdgcn_global_load_lds(GSRC(LANE_SRC), GLDS(tile),          \
                                       SIZE, OFFSET, 0);                    \
    }                                                                       \
    EPILOGUE                                                                \
  }

//     name          size  imm   predicate
PROBE(lds_dma_x4,      16,   0,  true)
PROBE(lds_dma_x3,      12,   0,  true)
PROBE(lds_dma_b32,      4,   0,  true)
PROBE(lds_dma_b8,       1,   0,  true)
PROBE(lds_dma_b16,      2,   0,  true)
PROBE(lds_dma_x4off,   16,  32,  true)
PROBE(lds_dma_x4msk,   16,   0,  ((tid & 1) == 0))

// Unaligned wave-uniform LDS base. An implementation that rounds M0 down to
// a dword boundary puts this at byte 1024 instead of 1026 -- a plausible
// image that is wrong by two bytes in every slot.
extern "C" __global__ void lds_dma_x4odd(const u32 *__restrict src,
                                         u32 *__restrict out) {
  PROLOGUE
  __builtin_amdgcn_global_load_lds(
      GSRC(LANE_SRC), GLDS((__attribute__((address_space(3))) char *)tile + 2),
      16, 0, 0);
  EPILOGUE
}

// Out-of-range LDS destination. Hardware discards the write; gem5's
// LdsChunk::write() must do the same and must not corrupt anything. The
// whole tile has to come back as the sentinel.
extern "C" __global__ void lds_dma_x4oob(const u32 *__restrict src,
                                         u32 *__restrict out) {
  PROLOGUE
  const u32 *lane_src = LANE_SRC;
  unsigned long long saved;
  asm volatile("s_mov_b64 %[sv], exec\n\t"
               "s_mov_b32 m0, 0x40000000\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "s_waitcnt vmcnt(0)"
               : [sv] "=&s"(saved)
               : [a] "v"(lane_src)
               : "memory");
  (void)saved;
  EPILOGUE
}

// Coalescer-token probe. Eight LDS DMAs are issued with EXEC = 0, twice
// gem5's max_cu_tokens = 4. Each acquires a token in
// ScheduleStage::fillDispatchList(); if the zero-lane path does not return
// them the CU runs out and the following real DMA never issues, so the tile
// comes back as the sentinel and the case fails (or the run hangs).
//
// The EXEC manipulation is written by hand because LLVM keeps an
// s_cbranch_execz around any block this expensive, which would skip the
// instruction instead of issuing it with an empty mask.
extern "C" __global__ void lds_dma_zeroexec(const u32 *__restrict src,
                                            u32 *__restrict out) {
  PROLOGUE
  const u32 *lane_src = LANE_SRC;
  unsigned long long saved;
  asm volatile("s_mov_b64 %[sv], exec\n\t"
               "s_mov_b64 exec, 0\n\t"
               "s_mov_b32 m0, 0\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "global_load_lds_dwordx4 %[a], off\n\t"
               "s_mov_b64 exec, %[sv]\n\t"
               "s_waitcnt vmcnt(0)"
               : [sv] "=&s"(saved)
               : [a] "v"(lane_src)
               : "memory");
  (void)saved;
  __syncthreads();
  __builtin_amdgcn_global_load_lds(GSRC(lane_src), GLDS(tile), 16, 0, 0);
  EPILOGUE
}
