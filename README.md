# amdgpu-sim

`amdgpu-sim` is a source-backed, host-side AMDGPU simulation stack.  Its
long-term purpose is to let HIP, OpenCL, Triton, PyTorch, and vLLM workloads
submit real AMDGPU code objects to gem5 without a guest kernel, `/dev/kfd`,
`/dev/dri`, or the production AMD UMD/KMD binaries.

The root is a control-plane and transaction-coordinator repository. The six
pinned upstream source lanes are standard Git submodules under `projects/`;
each child retains independent history and an immutable annotated baseline tag,
while the root gitlink records its exact head, tests, checkpoints, and lessons.
Model weights, virtual environments, and build
outputs are downloaded or built by scripts into ignored paths and are never
stored in Git.

The first hard acceptance target is the official text-only
`Qwen/Qwen3.5-0.8B` checkpoint, served by vLLM with tensor parallelism across
two independent gem5 instances, producing at least one greedy token with the
same token ID as the reference and no CPU arithmetic fallback.  The protocol
is N-rank from the beginning so TP=4/8 can follow without a pair-specific
rewrite.

Read [PLAN.md](PLAN.md) for the complete staged plan and [GOAL.md](GOAL.md) for
the immutable acceptance anchor.  A blank-context handoff starts with:

```text
继续执行 amdgpu-sim 计划。从 checkpoint 指定的下一条唯一动作继续：先读取
PLAN.md、GOAL.md、SOURCE_LOCK.json、state/current.json、最新 checkpoint 和
bitlesson，运行 scripts/resume.sh --verify；不要重做已通过的工作。
```

`CP-0002` is the accepted pristine-source boundary. The next action is
`P0-SELF-01`: add the authored/current lane registry and initialize the
standalone `self-amdgpu-runtime` project from an auditable pristine baseline.
The frozen `SOURCE_LOCK.json` remains the immutable upstream authority.
