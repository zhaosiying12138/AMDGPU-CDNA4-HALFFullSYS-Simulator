# AgentENV Integration Checkpoint

## Resume Order

Read this file first, then `AGENTENV_GOAL.md`, `AGENTENV_PLAN.md`, and `AGENTENV_BITLESSONS.md`.

## Current State

- Date: 2026-08-17
- Worktree: `/home/zhaosiying/amdgpu-sim-agentenv`
- Branch: `feature/agentenv-sandbox-isolation`
- Initial root commit: `ba85d279883f6190c7c7639ea6ea7d34ae8e04ab`
- No WSL shutdown has been performed by this task.
- No `.wslconfig` change has been made.
- Existing main-worktree SGLang/gem5 processes were observed and left running.

## Host Preflight Snapshot

- WSL kernel: `6.18.33.2-microsoft-standard-WSL2`
- `/dev/kvm`: present, `root:kvm`, mode `0660`; current user is not in group `kvm`.
- `/dev/ublk-control`: absent; current kernel has `CONFIG_BLK_DEV_UBLK` disabled.
- `net.ipv4.ip_forward=0`.
- `.wslconfig` currently contains only `memory=64424509440`.
- The replacement kernel must be source version `6.18.40.1`, commit `14794180686c2fb6307fbe359c359bec765249f3`, with `CONFIG_BLK_DEV_UBLK=m`.

## Next Action

Create the feature-local source pins and tooling, then run all non-disruptive checks. Stop before editing `.wslconfig` or invoking `wsl --shutdown`.

## Recovery

If a future approved kernel switch fails, remove only the added `kernel=` and `kernelModules=` lines from the backed-up Windows `.wslconfig`, then perform the separately approved shutdown and verify the stock `uname -r` before resuming.

