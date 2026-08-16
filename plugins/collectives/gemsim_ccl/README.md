# gemsim-ccl

`gemsim-ccl` is the out-of-tree functional collective layer for
`amdgpu-sim`. It is separate from the vLLM plugin so standalone transport,
planner, failure, and device-arithmetic gates can pass before framework TP is
enabled.

The first numerical primitive is synchronous device SUM on `gemsim_amd`:

- BF16 loads are extended to FP32, added pairwise, and rounded to BF16 RTNE
  on every executor invocation.
- FP32 performs one binary FP32 addition on every executor invocation.
- The destination is a private mutable workspace; received source storage is
  immutable and disjoint.
- Zero elements are a successful no-op and do not dispatch a kernel.
- Because zero elements access no bytes, they permit any source/destination
  alias; nonzero invocations require disjoint complete staged storages.
- Host arithmetic and CPU/NVIDIA fallback are prohibited in the target path.

The external verifier now formally accepts the reusable engine's BF16/1024
allreduce at N=2/3/4/8/16. It independently reconstructs descriptors, planner
steps, and per-hop BF16 RTNE arithmetic, then binds every normal Triton SUM to
the exact rank session, code object, log, stats, and trace. Every accepted rank
has bitwise oracle output, zero host reduction/fallback, measured FD delta zero,
and no orphan process. Unit and protocol coverage remain parameterized over
every world size 2..16; the live topology matrix intentionally includes odd N=3
and the N=16 capacity boundary.

The same engine is also accepted behind one real N=2 pinned-vLLM
`GroupCoordinator.all_reduce` invocation through the out-of-tree communicator.
That is a standalone communicator result, not a RowParallel layer, model shard,
or Qwen TP result. Model-specific tensor sharding and each additional collective
surface keep separate fail-closed gates.

`gemsim_ccl.torch_process_group` is the framework-neutral PyTorch third-party
ProcessGroup adapter. It supports CPU tensors only as a protocol diagnostic and
the standard PyTorch `cuda` device type only when `torch.version.hip` proves a
ROCm build. Device tensors are copied into immutable transport staging and are
committed back once after every private collective succeeds. This adapter is a
c10d fallback, not a replacement for upstream RCCL/NCCL paths and not a device
or model acceptance result by itself.
