# Host-native simulator architecture

## Decision

The physical machine is already the host. The project will expose a
host-native simulator daemon/library that reuses the existing GPU functional
core without building or launching the gem5 `VEGA_X86` target. The current
gem5 integration remains intact as a behavioral reference and regression
oracle.

This is an extraction of front-end responsibilities, not a rewrite of the GPU
model. The first implementation should reuse the existing packet processor,
GPU command processor, Vega decoder/instruction classes, dispatcher, compute
unit, sparse memory, queue, and signal state. New code should be limited to
the host event loop, page/memory adapter, daemon lifecycle, and build/link
boundary needed by those modules.

## Current boundary

The CP14 inventory found that the legacy AMDGPU device front-end depends on
`USE_X86_ISA`, `System`, CPU ports, and Ruby/SE memory context even though the
generic GPU/Vega layers are primarily guarded by `BUILD_GPU`. CP15-CP20 now
build standalone `HOSTGPU_NATIVE_CONTROL` executables with `BUILD_ISA=n`,
`USE_X86_ISA=n`, `BUILD_GPU=y`, and `VEGA_GPU_ISA=y`. B1 extracted the
host-owned address, queue, and command-processor admission responsibilities;
CP-0020-B2 now connects that admission to the reused GPUDispatcher, Shader,
ComputeUnit, and Vega instruction path for one locked `gpuReadWrite` fixture.
The legacy device/system front-end remains present as the reference path.
CP-0021 adds only a child-side Triton vecadd metadata/provenance gate: the
unknown-gfx950 target spelling and DEFAULT-visible descriptor exception are
accepted for parsing. CP-0022 freezes the generic payload-v2 codec. CP-0023
adds the append-only runtime client contract and a separate local
`HostNativeDispatchStateV2`/bridge-adapter admission lifecycle. CP-0024 adds an
owner-bound MessageType 18 handler and MessageType 19 output plumbing to the
VEGA_X86 bridge plus a shared route-policy harness. CP-0025 completes the
positive owner-bound daemon control lifecycle: bit 8 is advertised with its
dependencies, logical alignment 8 is separated from hidden page backing, v1
`MEMORY_COPY_H2D` publishes the full allocation subrange, and native queue,
signal, packet CRC, type-19 ACK, type-20 retirement, cleanup, and reconnect are
live. That path still does not issue the packet to GPUDispatcher/ComputeUnit or
validate kernel output.

The host-native process must preserve the CP8/CP13 fixed-width transport and
the runtime ownership/generation rules. It must not introduce a second wire
format, serialize host pointers, open `/dev/kfd` or `/dev/dri`, or silently
execute kernels on the CPU.

## Reuse layers

| Layer | Reuse target | New adapter responsibility |
| --- | --- | --- |
| ISA | `src/arch/amdgpu/vega` decoder and instruction classes | Select gfx950 and expose a stable decode/execute entry point |
| GPU execution | `src/gpu-compute` packet/command/dispatcher/CU path | Supply an event queue and host-owned scheduling context |
| Memory | existing sparse host/GPU memory state and page-visible helpers | Map bounded host buffers and preserve VA/ownership rules |
| Control | runtime transport, queue, signal, KMT and code-object codecs | Daemon lifecycle, capability negotiation, and completion loop |
| Reference | existing `VEGA_X86` gem5 bridge | Differential packet, trace, output, and error oracle |

## Gates

1. Inventory exact includes, link objects, and x86/system assumptions; freeze a
   compatibility matrix before source extraction.
2. Add a standalone build target and prove no `VEGA_X86`, x86 ISA objects,
   `System`/CPU/Ruby ports, production GPU DSOs, or device-node probes are
   required at runtime.
3. Connect CP13 staged code objects to one reused GPU path and compare a pinned
   gfx950 vecadd output and trace against the gem5 reference. CP-0016 provided
   the functional memory/dispatch parity adapter and metadata probe. CP-0017-A
   now stages the locked `gpuReadWrite` HSACO, copies exact PT_LOAD file bytes,
   zero-fills BSS, and binds descriptor/entry addresses in a no-x86 target.
   It is fixture-scoped and stops before native translation, dynamic
   AQL/kernarg publication, queue submission, and instruction execution.
4. Run the unmodified pinned Triton tutorial through the normal launcher with
   only simulator device selection changed. Capture compiler, HSACO digest,
   transport, dispatch, output, and fallback-counter evidence.

The first three gates are prerequisites for the Triton gate. CP-0017-A passes
the bounded PT_LOAD staging sub-gate only. CP-0018-B0 adds host-native
dispatch admission and listener-contract smoke: descriptor/entry validation,
280-byte hidden kernarg, 64-byte AQL materialization, and queue-control state.
CP-0019-B1 added `HostNativeQueueCore` and
`HostNativeCommandProcessorCore`: host-owned GPU-VA resolution, queue
registration, AQL publish/ring/fetch, descriptor/MQD/kernarg/signal-object
reads, and `HSAQueueEntry` admission. It reuses the existing packet and queue
ABIs but does not link or instantiate legacy HSAPP/GPUCommandProcessor
SimObjects. Its `aql_submitted=true` flag is scoped to the extracted native CP
core. The MQD host read index is unchanged, and GPUDispatcher/CU connection,
packet retirement, signal decrement, instruction fetch/retirement, ISA
execution, and output differential remain false at the historical B1 boundary.
CP-0020-B2 accepts one locked functional case: four 256-item workgroups,
sixteen wave64 waves, 19 instruction-start PCs per wave (304 total), exact
A/B/C output coverage, packet retirement, MQD read-index update, direct-u64
signal completion, and pin release. The runtime graph still has no CPU,
Process, Ruby, TLB, HSAPP, or GPUCommandProcessor objects. This does not prove
generic gfx950/arbitrary HSACO, timing accuracy, fences/barriers, atomics,
LDS/scratch, GPU TLB/Ruby/coherence, HIP/OpenCL, or performance. The historical
next gate was `P5-TRITON-VECADD-01`; CP-0021 now records its compile/provenance
prerequisite, while Triton and Qwen end-to-end remain 0/1. The following wire
gate was CP-0022. CP-0021
records the pinned unmodified tutorial and 5,408-byte HSACO identities,
12-DWORD (48-byte) descriptor preload, runtime CTest 16/16, and explicit false
compiler/JIT/launcher/transport/execution/fallback fields. It performs no
normal launch and leaves public A1 mapping VAs at zero. CP-0022 accepts a
separate payload-v2 codec boundary: v1 framing remains byte-stable, bit 8 and
records 18/19/20 are opt-in, and owner-scoped identities are strict. CP-0023
accepts the runtime client and local native adapter/admission step. Runtime
CTest is 18/18 and the gem5 protocol suite is 47/47 normally and under
ASAN/UBSAN. The local no-x86 state maps, allocates, publishes kernarg,
publishes/fetches AQL, admits through the extracted CP core, retires, and
unmaps; a post-fetch CP rejection uses `cancelFetch` to restore queue ownership
before pins are released. Admission is not execution. The daemon still neither
advertises capability bit 8 nor routes MessageType 18; GPUDispatcher/CU,
kernel execution, normal Triton launcher, compiler/JIT, and fallback remain
false. The retained 12-DWORD preload is NOT_SUPPORTED. CP-0024 accepts a
bounded partial handler boundary: source routing and canonical type-19 failure
encoding exist, and a policy harness covers absent-capability rejection,
no-mutation failure, 4K/64K page policy, alignment-8 rejection, SUBMIT
rejection, MAP, and UNMAP. The page-size policy is fixed for the owner session.
The live runtime-to-gem5 probe proves the canonical unsupported-capability
hello and baseline reconnect only; it does not send MessageType 18. A daemon
H2D route is also unproven: kernarg bytes retain the existing v1
`MEMORY_COPY_H2D` carrier rather than gaining a new v2 opcode. Normal alignment
8, positive type-18 routing, SUBMIT ACK, MessageType 20, queue/signal/packet/
ticks, launcher, compiler/JIT, execution, and fallback are still false at the
historical CP24 boundary. CP-0025 accepts the positive two-generation daemon
control lifecycle. The wire alignment remains the kernel ABI contract while
4096-byte backing stays private; page size remains owner-session-fixed, and v1
`MEMORY_COPY_H2D` is the only kernarg byte carrier. A durable type-19 ACK is
followed by a type-20 bookkeeping retirement with nonzero, nondecreasing ticks;
disconnect cleanup is verified by generation-advanced slot/VA reuse. Direct
remote counters are not exposed, although the focused native test reaches zero
resources. CP-0025 remains a control-only boundary; its bit-8 completion does
not imply GPU execution.

CP-0026 adds a separate bit-9 execution extension. Runtime word 0 bit 9 maps to
wire byte 1 bit 1 and is selectable only with bit 8 plus topology, queue,
memory, signal, and code-object dependencies. In the live `VEGA_X86` bridge
configuration, the exact 5,528-byte gfx950 `gpuReadWrite` image (SHA-256
`7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56`) has zero
descriptor preload and reaches the reused `GPUDispatcher`, `Shader`, and
`ComputeUnit`: four 256-item workgroups, sixteen wave64 waves, 304 instruction
starts, exact A/B/C output, durable type 20, duplicate D2H verification, and
UNMAP. The daemon JSONL trace is fsynced and is the authority for dispatch/CU
execution and post-ACK quarantine; endpoint JSON is only the authority for
client-delivered bytes. The positive trace has retired/type-20/session-complete
events; the negative post-ACK disconnect trace has only quarantine cleanup and
no type 20, D2H, UNMAP, or client-output claim.

The wire `signal_value_bits` remains expected `1` and
`signal_after_observed=false`; trace `signal_before=1` and `signal_after=0`
refer to the private native AQL completion signal. This is one locked fixture
and one `VEGA_X86` bridge route, not generic gfx950/arbitrary HSACO or a
standalone no-x86 daemon claim. `Gem5IsaSupported=false` and
`LockedGpuReadWriteExecutionSupported=true` remain intentionally distinct.
The 5,408-byte Triton image with 12-DWORD preload, normal launcher,
compiler/JIT, fallback, performance, Triton E2E, and Qwen remain false. The
next planned gate is `P5-PROFILE-01`, to be allocated as CP-0027 when begun.

## Non-goals

This workstream does not promise cycle/timing accuracy, full ROCr/libhsakmt coverage,
fence/barrier or atomic semantics, GPU TLB/Ruby/coherence behavior, full ROCm/OpenCL CTS, HIP/OpenCL compatibility, broad Triton operators, host-parallel threadblocks,
PyTorch/vLLM execution, or performance improvement. Those remain separate
checkpoints after the first deterministic vecadd differential result.
