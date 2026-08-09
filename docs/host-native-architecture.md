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

The AMDGPU device SConscript currently returns unless `BUILD_GPU` and
`USE_X86_ISA` are enabled. In contrast, the generic `gpu-compute` and
`arch/amdgpu/vega` SConscripts are guarded by `BUILD_GPU`; the x86 dependency
is concentrated in the AMDGPU device/system front-end and in configurations
that construct an x86 `System`, CPU ports, and Ruby/SE memory context. This
gives us a measurable first gate: compile and link the reusable GPU set with
the x86 front-end excluded, then audit symbols and dynamic dependencies.

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
   gfx950 vecadd output and trace against the gem5 reference.
4. Run the unmodified pinned Triton tutorial through the normal launcher with
   only simulator device selection changed. Capture compiler, HSACO digest,
   transport, dispatch, output, and fallback-counter evidence.

The first three gates are prerequisites for the Triton gate. No current
checkpoint claims that any of them, or Triton end-to-end, has passed.

## Non-goals

This workstream does not promise cycle accuracy, full ROCr/libhsakmt coverage,
HIP/OpenCL compatibility, broad Triton operators, host-parallel threadblocks,
PyTorch/vLLM execution, or performance improvement. Those remain separate
checkpoints after the first deterministic vecadd differential result.
