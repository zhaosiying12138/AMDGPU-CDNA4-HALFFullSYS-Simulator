# AgentENV Integration Goal

## Objective

Create an isolated feature branch that can run the existing gem5 and active ROCm environments inside Kimi AgentENV, with two independent sandboxes suitable for concurrent vLLM TP4 and SGLang TP1 launches.

This round is an environment/integration task only. It does not repair model, vLLM, SGLang, ROCm, or gem5 inference correctness failures encountered after the launch reaches its existing initialization or weight-loading boundary.

## Acceptance Boundary

The feature is accepted only when all of the following are true:

1. The original `/home/zhaosiying/amdgpu-sim` `main` worktree, builds, and processes remain untouched.
2. AgentENV is pinned, built, and operated from this worktree.
3. The host passes KVM, ublk, Firecracker, and AgentENV lifecycle gates.
4. The active runtime closure is bundled deterministically, including symlinks, permissions, absolute-path compatibility, gem5, runtime DSOs, frameworks, and the model.
5. Two sandboxes have distinct VM, PID, filesystem, network, temporary, socket, cache, log, SMI, and endpoint namespaces.
6. vLLM TP4 and SGLang TP1 can be launched concurrently and each reaches its existing initialization/weight-loading marker, then survives the bounded observation interval.
7. A failure after that boundary is recorded as an existing model/framework/runtime issue, not fixed in this round.

## Explicit Safety Gate

The current WSL kernel lacks ublk. Building and staging a replacement kernel is allowed, but changing `%UserProfile%\\.wslconfig` or running `wsl --shutdown` is a maintenance action that can terminate unrelated WSL work. It is prohibited until the user explicitly approves the exact switch and rollback procedure.

## Current Pins

- Root base: `ba85d279883f6190c7c7639ea6ea7d34ae8e04ab`
- AgentENV: `kvcache-ai/AgentENV`, commit `d2405769618c37de5e181a94dc4052d786ec041a`
- WSL2 kernel: `microsoft/WSL2-Linux-Kernel`, branch `linux-msft-wsl-6.18.y`, commit `14794180686c2fb6307fbe359c359bec765249f3` (source version `6.18.40.1`)

