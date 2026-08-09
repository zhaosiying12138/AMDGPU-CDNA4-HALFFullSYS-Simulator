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
LDS/scratch, GPU TLB/Ruby/coherence, HIP/OpenCL, or performance. The next gate
is `P5-TRITON-VECADD-01`; Triton and Qwen end-to-end remain 0/1.

## Non-goals

This workstream does not promise cycle/timing accuracy, full ROCr/libhsakmt coverage,
fence/barrier or atomic semantics, GPU TLB/Ruby/coherence behavior, full ROCm/OpenCL CTS, HIP/OpenCL compatibility, broad Triton operators, host-parallel threadblocks,
PyTorch/vLLM execution, or performance improvement. Those remain separate
checkpoints after the first deterministic vecadd differential result.
