# Framework, compiler, runtime, and simulator layering

## Decision

The accepted framework path is an out-of-tree integration, not a fork of
vLLM, PyTorch, or Triton. Pinned upstream source remains byte-for-byte
unchanged. A normal vLLM model uses its existing tensor-parallel layer,
parameter loader, process-group, and `GroupCoordinator` semantics; the project
supplies only the platform, operator, attention, and device-communicator
implementations that vLLM intentionally delegates to plugins.

Likewise, a normal Triton program continues through the pinned Triton frontend,
JIT cache, compiler pipeline, and runtime abstractions. The project-owned
`gemsim_amd` backend implements Triton's backend driver/compiler contracts and
reuses the upstream AMD lowering. It must not edit Triton core
`python/triton/runtime/driver.py`, `python/triton/compiler/compiler.py`, the
language frontend, or upstream AMD lowering to recognize a model or operator.

The formal transparency path uses unchanged upstream ROCr/libhsakmt through
the official Model Interface 1.1 and an eventual compatible HIP/RCCL product.
vLLM and SGLang see their normal ROCm platforms, tensors, and collectives;
upstream Triton sees its normal HIP backend; users do not construct gem5
transport records. The model/provider boundary calls the small, versioned
`self-amdgpu-runtime` ABI, which owns gem5
process/session management, allocation, code-object load, dispatch, wait, and
copy. The separate CCL plugin owns rank rendezvous, ordering, transport, device
reduction, failure propagation, and cleanup. gem5 remains kernel-agnostic.

## Layer ownership

| Layer | Reused upstream contract | Project-owned implementation | Forbidden coupling |
| --- | --- | --- | --- |
| vLLM model and TP | Upstream model classes, `PluggableLayer`, parameter loaders, parallel-state initialization, `GroupCoordinator` | OOT platform/general plugins, strict layer/operator adapters, `DeviceCommunicatorBase` implementation | Editing pinned vLLM, writing private `_TP`, bypassing official weight loaders, model tensor payload through Gloo |
| PyTorch integration | Dispatcher, `torch.library`, Fake/Meta, tensors and process-group control | Project operator schemas/implementations and version-pinned fail-closed compatibility guards only where no OOT hook exists | Editing pinned torch, CPU/NVIDIA arithmetic fallback, broad process-global monkey patches |
| Triton | Frontend, JIT/cache, compiler framework, AMD lowering and code-object generation | OOT `gemsim_amd` backend driver/compiler/launcher and project kernels | Editing Triton core `driver.py`/`compiler.py`, per-model compiler branches, per-kernel runtime admission |
| ROCr/KMD cut | Upstream `libhsakmt` public ABI and Model Interface 1.1 | Model DSO translating KFD/DRM/AQL semantics to typed self-runtime operations | Cloning the 124 thunk exports, linking `/dev/kfd` or host libdrm, fake-success ioctls |
| Runtime | Stable fixed-width C ABI and generic ownership/lifecycle semantics | `self-amdgpu-runtime`, managed provider/session, buffer/module/kernel proxies | Rebuilding per operator, host-pointer wire ABI, application-specific calls |
| Collectives | vLLM communicator API and versioned CCL numeric semantics | `gemsim-ccl` planner/carrier/rendezvous/engine and ordinary Triton SUM kernels | Host reduction, TP2-only topology, hidden Gloo payload fallback |
| Simulator | Generic HSACO, kernarg, allocation, dispatch, memory and trace semantics | gem5 bridge plus reused GPU/Vega execution core | Operator names/hashes/shapes/oracles in admission, autonomous model-aware behavior |

## Required dependency direction

The dependency graph is strictly one way:

```text
ordinary vLLM, SGLang, or Triton .py
  -> pinned framework / ROCm PyTorch / upstream Triton HIP contracts
  -> unchanged ROCr/libhsakmt Model 1.1 and compatible HIP/RCCL facades
  -> stable self-runtime public API
  -> runtime-gem5 bridge
  -> stable gem5 GPU adapter API
  -> unmodified gem5 queue / memory / dispatcher / CU / Vega implementation
```

The runtime-gem5 bridge is the only layer that knows both the self-runtime
wire/lifecycle model and gem5 integration details. It owns codec and capability
negotiation, peer/rank/session identity, ownership and generation translation,
deadline/cancellation, process supervision, request-to-adapter conversion,
completion/error conversion, and trace correlation. These concepts must not
leak upward into vLLM/Torch/Triton or downward into the reusable gem5 GPU core.

The bridge calls a small stable gem5-side adapter expressed only in GPU-domain
operations: allocate/free, copy, code-object map/unmap, kernarg and queue
publication, dispatch, wait/retire, and completion. gem5's dispatcher, compute
units, Vega decoder/instructions, and functional memory do not include runtime
message types, vLLM rank objects, Triton kernel names, Qwen tensor roles, or CCL
operations. Collective SUM remains an ordinary compiler-produced kernel.

Every production layer is operator-agnostic. It may validate target and ABI,
ELF/code-object metadata, kernarg layout, resource limits, address ownership,
grid/workgroup/shared-memory geometry, synchronization, and ISA semantics. It
may not identify or special-case a kernel by name, image hash, model, tensor
shape/role, expected output, or PC list. A new operator exposes a shared
semantic defect or unsupported capability; the repair belongs in that shared
layer and immediately gains unrelated cross-operator regressions.

## Tensor-parallel integration rule

The production TP path must preserve vLLM's public behavior end to end:

1. Use vLLM's supported distributed initialization and named TP
   `GroupCoordinator`; do not assign or patch private parallel-state globals.
2. Let the upstream layer constructor and weight loader compute rank-local
   parameter shapes and copy the exact shard. Project code validates those
   ranges and reconstruction hashes but does not create a second sharding
   framework.
3. Let the OOT `PluggableLayer` adapter implement only the unsupported device
   computation while preserving the upstream layer's dtype, bias,
   quantization, return, alias, and collective contracts.
4. Route `GroupCoordinator` collectives through the OOT
   `DeviceCommunicatorBase`; Gloo may carry audited bootstrap/control traffic,
   but never model tensor payload or arithmetic.
5. Keep CCL generic for every world size 2..16. Each model separately admits
   only TP degrees for which all relevant dimensions and head layouts are
   legal.

Formal Qwen model acceptance is intentionally bounded to TP=2 and TP=4. The
2..16 CCL and simulator-device capacity remains a reusable infrastructure
contract; it does not create TP=8/16 model obligations or operator-specific
branches.

The framework portability lane adds pinned upstream SGLang at Qwen3.5-0.8B
TP=4, using the same runtime-gem5 bridge and CCL engine without changing either
the SGLang checkout or the bridge. vLLM Qwen3.5-9B TP=16 with upstream
`torch.compile` is a separate scale lane. ROCm attention/GEMM selection remains
inside each upstream framework (AITER/ROCM_ATTN/TRITON_ATTN for vLLM and
AITER/Triton for SGLang); no FlashInfer-specific AMD path or per-operator
framework fork is permitted.

The first layer gate is the real Qwen3.5-0.8B MLP `down_proj`: full BF16 weight
`[1024,3584]`, TP2 column shards `[1024,1792]`, rank-local input
`[tokens,1792]`, local output `[tokens,1024]`, and one out-of-place SUM
all-reduce. It must prove the upstream loader's shard ranges and full-weight
reconstruction SHA, input and weight immutability, fresh output, equivalence to
a full single-device projection, exact collective sequence, authoritative
device traces, zero model payload through Gloo, zero fallback, and clean
two-daemon teardown.

## Compatibility scope

The shared device-facade lane implements only capabilities reached by frozen
upstream ROCr/HIP/PyTorch/Triton/vLLM/SGLang callers. Each capability requires
a source-pinned ABI contract, standalone conformance, cross-framework
regression, and the same runtime primitives. It must not become an alternative
model-specific path. ROCjitsu and Linux KFD behavior are reference/test inputs;
neither is a second execution backend.

## Migration and legacy removal

The current implementation is a regression baseline, not an architectural
constraint. Migration proceeds side by side from a content-addressed source
backup: freeze current behavior, introduce the common bridge contract and
stable gem5 adapter, migrate one operation family at a time, and compare output,
trace, failure, cleanup, and performance against the baseline. The new product
becomes active only after the complete accepted OpenCL, Triton, framework,
CCL, layer, and model gates pass.

Legacy code is removed after all of these are true: no production/build/test
reference remains; the new bridge covers its behavior; accepted historical
evidence can still be interpreted without executing the old path; negative and
fault regressions have an equivalent; and a clean build plus full acceptance
suite passes without it. Fixture-specific routes, duplicate codecs, copied
framework state machines, and obsolete environment launchers are deletion
candidates, not permanent compatibility surfaces.

## Acceptance and upgrade policy

- Source and product manifests pin every upstream checkout, installed module,
  OOT plugin, runtime DSO, gem5 binary/config, and generated code object.
- Tests fail closed on an unsupported upstream version or extension signature.
- Disabling project plugins restores upstream behavior; accepted execution
  contains no diagnostic patch.
- Upstream upgrades are handled by updating compatibility checks and rerunning
  operator, communicator, layer, model, trace, fallback, and cleanup gates,
  not by carrying unreviewed source edits.
- A standalone collective, communicator, layer, model prefix, and full TP
  inference are distinct claims. Passing one never promotes the next.
