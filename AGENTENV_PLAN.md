# AgentENV Integration Plan

## Status

This plan is executed only in `feature/agentenv-sandbox-isolation` at `/home/zhaosiying/amdgpu-sim-agentenv`.

## Steps

- [x] Create an isolated worktree and preserve the main branch boundary.
- [x] Persist the goal, plan, checkpoint, and lessons before source changes.
- [x] Pin AgentENV source under `projects/AgentENV`.
- [x] Pin WSL kernel source under the feature build area and add reproducible ublk build tooling.
- [x] Add host preflight and a feature-local AgentENV service/unit bound to loopback.
- [x] Build a deterministic active-runtime closure as tar.zst; do not rely on symlink-preserving directory upload.
- [x] Add guest bootstrap and two-sandbox lifecycle orchestration with per-instance namespaces.
- [x] Run static/unit/build checks without stopping WSL or changing `.wslconfig`.
- [x] Present the exact candidate kernel activation diff and rollback plan; pause before explicit `wsl --shutdown` approval.
- [ ] After explicit approval only: activate the custom kernel, run ublk/Firecracker/AgentENV gates, and perform the bounded concurrent launch acceptance.

## Resource Policy

- No command may modify the original main worktree or its build directories.
- No command may kill or signal unrelated gem5, SGLang, vLLM, build, or AgentENV processes.
- All feature state, logs, sockets, bundles, snapshots, and build outputs live below this worktree's `build/` directory.
- Host-level package installation is permitted when required, but must be recorded and must not trigger WSL shutdown.
- The common sandbox template uses Ubuntu 26.04, 12 vCPUs, 24 GiB RAM, and a 96 GiB root disk; two instances therefore fit within the current 24 CPU/60 GiB WSL budget with bounded headroom.

## Out of Scope

- vLLM/SGLang inference fixes
- TP topology or CCL changes
- model/operator/shape-specific workarounds
- merging this branch into `main`

## Acceptance Boundary

This checkpoint proves source pinning, feature-local build/tooling, deterministic
bundle planning, and lifecycle safety checks. It does not prove that the AgentENV
server can boot on the current WSL kernel, that `/dev/ublk-control` is available,
or that either model workload runs. Those require the separately approved kernel
activation gate and are intentionally not performed in this phase.
