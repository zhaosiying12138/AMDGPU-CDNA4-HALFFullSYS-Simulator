# amdgpu-sim

`amdgpu-sim` is a source-backed, host-side AMDGPU simulation stack.  Its
long-term purpose is to let HIP, OpenCL, Triton, PyTorch, and vLLM workloads
submit real AMDGPU code objects to gem5 without a guest kernel, `/dev/kfd`,
`/dev/dri`, or the production AMD UMD/KMD binaries.

The root is a control-plane and transaction-coordinator repository. The six
pinned upstream source lanes and the project-authored `self-amdgpu-runtime`
lane are standard Git submodules under `projects/`. Each child retains
independent history and an immutable annotated baseline tag, while the root
gitlink records its exact current head, tests, checkpoints, and lessons.
`SOURCE_LOCK.json` owns upstream provenance; `PROJECT_LANES.json` owns our
project baselines; accepted checkpoints own descendant work heads. Model
weights, virtual environments, and build
outputs are downloaded or built by scripts into ignored paths and are never
stored in Git.

The first hard acceptance target is the official text-only
`Qwen/Qwen3.5-0.8B` checkpoint, served by vLLM with tensor parallelism across
two independent gem5 instances, producing at least one greedy token with the
same token ID as the reference and no CPU arithmetic fallback.  The protocol
is N-rank from the beginning so TP=4/8 can follow without a pair-specific
rewrite.

Two later goals are explicit in plan revision 2. Once the user's unmodified
Triton `tutorial/01-vecadd.py` request (the pinned checkout's
`python/tutorials/01-vector-add.py`) runs through GemSim, retained profiles will drive a
small number of high-impact operator/simulator optimizations, followed by a
separately gated experiment for CPU-parallel threadblock simulation that must
preserve dependency and synchronization semantics. After the model path is
usable, a low-priority simulator-aware `rocm-smi` client will report the ON/OFF
state of multiple gem5 daemon instances without probing physical GPUs.

Read [PLAN.md](PLAN.md) for the complete staged plan and [GOAL.md](GOAL.md) for
the immutable acceptance anchor.  A blank-context handoff starts with:

```text
继续执行 amdgpu-sim 计划。从 checkpoint 指定的下一条唯一动作继续：先读取
PLAN.md、GOAL.md、SOURCE_LOCK.json、state/current.json、最新 checkpoint 和
bitlesson，运行 scripts/resume.sh --verify；不要重做已通过的工作。
```

`CP-0007` remains the accepted bridge-private signal/event boundary. The standalone
runtime and gem5 preserve the CP-0004 byte-exact handshake, CP-0005 bounded
queue control, CP-0006 sparse simulated-memory transfer, and N=1/2/3/4/8
isolation gates while adding signed 64-bit signal create/load/store/destroy,
generation-safe one-shot waits, event-queue completion, bounded outbound
accounting, shared request correlation, and retry/poison semantics. The signal
records remain host transport primitives: they do not expose GPU-visible signal
memory or claim packet submission, code-object loading, or GPU execution.

`CP-0008` is now the accepted pinned-dispatch boundary. It preserves every
CP-0004 through CP-0007 gate and proves one source-pinned `gfx950-xor-u8-v1`
wave64, one-CU, one-workgroup AQL execution through the real
`HSAPacketProcessor -> GPUCommandProcessor -> GPUDispatcher -> CU` path, with
exact packet/trace hashes, positive retired/store statistics, CP7 signal
completion at retirement plus one tick, exact non-identity D2H bytes and CRC,
causal clean exit, and zero host fallback. This is still not a generic
code-object, ROCr/libhsakmt, HIP, OpenCL, Triton, PyTorch, vLLM, multi-CU,
collective, or performance claim. The next action is `P2-KMT-ABI-01`: inventory
the pinned ROCr ThunkLoader/libhsakmt ABI and prepare a source-exact provider
skeleton. The frozen `SOURCE_LOCK.json` and registered project baseline remain
immutable.
