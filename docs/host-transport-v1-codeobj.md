# Host Transport v1 Code-Object Authority

CODE_OBJECT_ABI_V1 (capability bit 6, mask 0x40) is frozen as a
source-authority extension at P3-CODEOBJ-01. The normative manifest is
protocol/host-transport-v1-codeobj.json. This revision has no code-object
request, acknowledgement, upload, or execution message. transport.message_types
is intentionally empty. It gives the runtime stable names and validation rules
before a loader wire API is reviewed.

The authority is tied to SOURCE_LOCK.json SHA-256
8c0a6d1d04d5a73ab036ddc56d09ff59f7513077931868bdfac5a763f85a73d4
(SL-0001) and PROJECT_LANES.json SHA-256
a5a7a5a3fa0abb0aa9c7dffd11d8823293de7bba7c5f97a7fd0b83d8d69260f2.
The complete source blob and tree identities are recorded in the JSON; this
authority lane modifies only the three named root files and does not touch
any child repository.

## Source boundary

The ABI conclusions are drawn from the locked baseline commits, not from
generic ROCm documentation:

- ROCr ELF/version and loader flow:
  projects/rocm-systems at
  92115a2941982a384de161be3f78cf9bff547027 (tree
  28bf42b65f7aad25167180543dda69b5fc6caf58). The relevant
  amd_hsa_code.cpp, amd_elf_image.cpp, loader/executable.cpp,
  AMDHSAKernelDescriptor.h, and amd_hsa_elf.h blob hashes are in the
  manifest.
- LLVM's ELF V4-V6, metadata verifier/streamer, descriptor, relocation, and
  gfx950 definitions are from commit
  73f2a21fe16b34e35fd0e149564b8664e59da392 (tree
  d589480097e8a30fd1df38435ccc9a9fca71f489).
- The same locked LLVM tree's amd/device-libs subtree is anchored at tree
  9da20ec22ab6c7fe8b81693db14dfa17c901f483. Its OCKL kernel-code header and
  OCLC ABI/ISA templates define device-side resource bits and generated
  version constants; README.md makes the bitcode build/link path explicit.
  The selected blob hashes are README.md
  f8b6547c387a69deca6b20bcd7e8875781dbdb56,
  ockl/inc/amd_hsa_kernel_code.h
  6c2742a68a3d4d51d01602208aac921483bb6c47,
  oclc/src/abi_version.cl.in
  699f8cb64ae80d50790261f2df75469d519ebceb, and
  oclc/src/isa_version.cl.in
  654a1d45d709247b1963366350c7669521ec5f25.
- Triton's locked AMD backend is commit
  cd513e2798db0f4675b3d1205c8e76eb3381a0b (tree
  944754ed44b5414f2b72fed267455abc9f6fc8c1).
- gem5's untouched decoder baseline is commit
  cbf0eae213c5e39c727172b546434287d47b5bbe (tree
  d6527ec47acf018cb89afab496b974bda79eaa36).

ROCr is the ELF image loader and relocation consumer. It does not parse
LLVM's MsgPack metadata; LLVM emits the NT_AMDGPU_METADATA note and validates
its map. Keeping those roles separate avoids treating a hand-decoded fixture
as a loader implementation.

The device-libs sources are templates and source files only. No built bitcode
set is pinned here, so a future compiler/device-library build must freeze the
exact generated ABI and ISA constants before it can claim reproducible HSACO
emission.

## ELF contract

The accepted HSA code-object ABI values are:

| code-object ABI | EI_ABIVERSION | metadata version |
| --- | ---: | --- |
| V4 | 2 | [1, 1] |
| V5 | 3 | [1, 2] |
| V6 | 4 | [1, 2] |

An image must be ELF64, little-endian, ELFOSABI_AMDGPU_HSA (64),
EM_AMDGPU (224), ET_DYN (3), ELF version 1, and e_entry == 0. For
gfx950, e_flags carries machine 0x4f; XNACK and SRAMECC are independent
"any" bits in the observed images. An HSA loader must validate these values
before mapping segments.

V3 and later metadata is a SHT_NOTE named .note, with owner AMDGPU,
note type NT_AMDGPU_METADATA (32), and a MessagePack descriptor. The root
map requires amdhsa.version, amdhsa.target, and amdhsa.kernels. Each kernel
requires its name, descriptor symbol, kernarg size/alignment, LDS and private
fixed sizes, wavefront and register counts, and maximum flat workgroup size.
Every argument requires byte offset, byte size, and value_kind; hidden kinds
are ABI fields, not optional padding that a caller may compact.

The linker emits Elf64_Rela records. The manifest records the complete
AMDGPU relocation IDs used by the locked LLVM source and the static formulas
(S + A, S + A - P, and the REL16 word formula) plus dynamic
RELATIVE64 = B + A. A future loader must map PT_LOAD segments, load/zero-fill
PROGBITS/NOBITS, resolve definitions, apply relocations, and only then publish
an executable. Both current fixtures happen to have zero relocation sections.

## Descriptor, symbols, and kernargs

Every kernel has a 64-byte little-endian descriptor. Its fixed fields are
group/private/kernarg segment sizes at offsets 0/4/8, a signed entry byte
offset at 16, resource registers at 44/48/52, properties at 56, and kernarg
preload at 58. Property bit 1 enables the dispatch pointer and bit 3 enables
the kernarg segment pointer in the observed descriptors (0x000a). The AQL
kernel_object is the descriptor address, not the .text entry address.

Metadata symbol resolves to a protected global STT_OBJECT descriptor in
.rodata of size 64. Metadata name resolves independently to a protected
global STT_FUNC in .text. A loader must require both exact symbols and must
not guess a suffix or use e_entry.

Kernarg offsets are byte offsets. LLVM emits explicit arguments followed by
hidden block counts (u32), group sizes and remainders (u16), reserved ABI
space, global offsets (u64), and grid dimensions (u16), with later V5/V6
hidden fields retained when present. The metadata alignment is a byte
alignment; the LLVM emitter's lower bound is 4 bytes. The tracked images use
8-byte alignment. The 280- and 288-byte segment sizes include hidden fields and
reserved holes even where LDS and private fixed sizes are zero. A runtime must
construct those holes exactly, not pack only visible arguments.

group_segment_fixed_size is fixed LDS allocation in bytes. Dynamic LDS is
represented by hidden_dynamic_lds_size and requires corresponding dispatch
and loader support. private_segment_fixed_size is fixed scratch/private
allocation; dynamic stack and private/shared base hidden fields are coupled to
descriptor properties. No current fixture proves dynamic LDS, scratch, or
stack execution.

## Tracked fixtures

The two files are already tracked at the locked ROCm commit; they are not
generated by this change:

| fixture | bytes | SHA-256 | metadata desc SHA-256 | kernels |
| --- | ---: | --- | --- | --- |
| gpuReadWrite_kernels.hsaco | 5,528 | 7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56 | 13012b38a62029e0cb7798121f188db945bc6683a7a4ab60ee520bd981725b0a | gpuReadWrite |
| binary_search_kernels.hsaco | 11,296 | cb415571319569f4cdc28872faa2155dfe12be06a8f75fe173df3486fdabc053 | 48cca02a7b36a54363ef5a4eab01aaa49a5b5530ec8929f68e83ed57e96542e4 | binarySearch, binarySearch_mulkeys, binarySearch_mulkeysConcurrent |

Both are ELF V6/gfx950 images with wavefront 64 and kernarg alignment 8.
gpuReadWrite has kernarg size 280, SGPR/VGPR counts 22/6, and three
global-buffer arguments. binarySearch has size 280 and counts 19/8;
the two multi-key kernels have size 288 and counts 20/6 and 22/12. The full
argument maps, descriptor bytes, symbol values, section offsets, and Git blob
IDs are in the manifest. The descriptors are at .rodata offsets 0x800,
0x1400, 0x1440, and 0x1480; each is exactly 64 bytes.

## ISA audit and blockers

The audit used the available
/usr/lib/llvm-21/bin/llvm-objdump -d --mcpu=gfx950 only to observe the
tracked bytes. That executable is not pinned by SOURCE_LOCK; the fixture
.comment names AMD clang/LLD 22 roc-7.2.3 commit
f58b06dce1f9c15707c5f808fd002e18c2accf7e. Therefore the disassembly is not
a reproducible compiler claim.

All 13 unique mnemonic spellings in gpuReadWrite occur in the locked gem5
decoder source, but gem5's GPU.py gfx950 enum does not imply decoder feature
equivalence. binary_search has 76 unique spellings and contains
v_fmamk_f32 twice; that mnemonic is absent from the locked baseline decoder.
The general source-authority audit remains blocked, and binary_search remains
blocked. CP-0020 adds a separate, descendant execution record for the locked
`gpuReadWrite` bytes only: the 5,528-byte fixture crosses the no-x86 native
queue/CP core into the reused GPUDispatcher/Shader/ComputeUnit/Vega path,
executes four 256-item workgroups as sixteen wave64 waves, records the exact
19-PC instruction-start sequence per wave (304 total), and checks A/B/C output,
packet retirement, MQD read-index `0->1`, direct-u64 signal `1->0`, and pin
release. This is an accepted functional boundary for that one fixture, not a
generic gfx950 decoder/loader or arbitrary-HSACO claim. The existing CP-0008
bounded code-image fixture is not a substitute for a HSACO loader or a
reduction fixture.

The fixture's metadata describes hidden ABI fields, while its locked bytes read
the global-offset slots at kernarg offsets `+24`, `+32`, and `+40`. CP-0020
zeroes those three slots as fixture-specific compatibility data and preserves
the source bytes; this does not establish a generic V5/V6 metadata or hidden
argument loader. Atomics are rejected; only the demonstrated plain global
read/write path is in scope.

## CP-0021 Triton provenance boundary

CP-0021 records the first pinned Triton vecadd compile/provenance gate without
promoting it to a launcher result. The unmodified source
`projects/triton/python/tutorials/01-vector-add.py` is 5,644 bytes with SHA-256
`842430949e0ccde4fbce07606cce3ac4bac36bf21b2b12619a31b795ca4029b3`; the
retained 5,408-byte HSACO has SHA-256
`ee8b0f892da7ab1886f17ee66f88de5c23e05a48f7f361e02bd0707c9a11826e` and uses
the producer spelling `amdgcn-amd-amdhsa-unknown-gfx950`. Its `vecadd.kd`
descriptor is GLOBAL DEFAULT-visible while the code symbol remains GLOBAL
PROTECTED; the parser exception is target-scoped and metadata-only. CP-0021
hash-binds this one observed artifact; it does not make the parser a generic
hash or arbitrary-image acceptance path, and `isa_supported_by_gemsim` remains
false.

The raw descriptor encodes a kernarg preload value of **12 DWORD** (48 bytes),
not 12 bytes. Runtime CTest is 16/16 and the focused code-object set is 4/4;
caller-local code and zero-initialized kernarg materialization pass, while
`compiler_invoked`, `jit`, `launcher`, `transport`, `execution`, and `fallback`
are all false. EV-0050 contains the detailed child result and EV-0051 binds it
to the clean runtime child and CP-0021 transaction journal.

This gate does not alter the public A1 contract: mapping, descriptor, code, and
kernarg VAs remain zero and the transport remains fixture-only. CP-0022 is the
next generic wire-v2/allocator/AQL-linkage/normal-launcher gate. Triton E2E
remains 0/1.

## Verification

tests/test_host_transport_codeobj_spec.py checks:

- JSON schema, immutable lock/lanes hashes, repository commit/tree anchors, and
  every listed source Git blob and SHA-256;
- tracked HSACO Git blobs, complete-file hashes, ELF headers/sections/segments,
  note owner/type/MessagePack descriptor hashes, no relocation sections,
  descriptor bytes/fields, and exact symbol relations;
- metadata versions, targets, kernel resource values, hidden-argument offsets,
  alignment, and kernarg sizes;
- the explicit blocked general ISA status, the bounded CP-0020 functional
  addendum, the CP-0021 Triton provenance addendum, and the absence of any wire
  message IDs.

The test is a parser/authority check. Passing it records the one locked
CP-0020 functional case but does not turn the binary-search fixture or generic
gfx950/arbitrary HSACO into runnable gem5 kernels.
