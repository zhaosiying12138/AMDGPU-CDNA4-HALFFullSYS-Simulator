# AgentENV Integration Checkpoint

## Resume Order

Read this file first, then `AGENTENV_GOAL.md`, `AGENTENV_PLAN.md`, and
`AGENTENV_BITLESSONS.md`. Do not infer runtime acceptance from the source
commits or from a dry-run report.

## Current State

- Date: 2026-08-17
- Worktree: `/home/zhaosiying/amdgpu-sim-agentenv`
- Branch: `feature/agentenv-sandbox-isolation`
- Current root commit before the next checkpoint commit: `dda4293` (initial active WSL config replacement)
- Nested AgentENV pin: `d2405769618c37de5e181a94dc4052d786ec041a`
- Nested WSL kernel pin: `14794180686c2fb6307fbe359c359bec765249f3` (`6.18.40.1`)
- The user performed the first WSL shutdown/restart. It returned to the stock
  kernel because the first generated `.wslconfig` used single backslashes in
  Windows paths, which WSL did not accept as the custom-kernel path.
- The active `.wslconfig` has been replaced again with a corrected candidate
  that preserves literal double backslashes. A second shutdown/restart has not
  been performed and remains an explicit user gate.
- Original config backup:
  `/mnt/c/Users/Admin1/.wslconfig.pre-agentenv-20260817-529594c.bak`
  (SHA256 `24177c675423be250afdc6a47422af1ccba9e372a32f00619e7578180c75a0e2`).
- Rejected single-backslash config backup:
  `/mnt/c/Users/Admin1/.wslconfig.invalid-single-backslash-20260817-dda4293.bak`
  (SHA256 `e2bedfc6aede345a159b928e3baf9b1a348ebf0ee8d4a6a80c11f517e7051341`).
- Existing main-worktree SGLang/gem5 processes were observed and left running.
- No AgentENV server or sandbox has been started by this feature branch.

## Host Preflight Snapshot

- WSL kernel: `6.18.33.2-microsoft-standard-WSL2`
- `/dev/kvm`: present, `root:kvm`, mode `0660`; current user is not in group `kvm`.
- `/dev/ublk-control`: absent; current kernel has `CONFIG_BLK_DEV_UBLK` disabled.
- `net.ipv4.ip_forward=0`.
- `.wslconfig` currently contains only `memory=64424509440`.
- The replacement kernel must be source version `6.18.40.1`, commit `14794180686c2fb6307fbe359c359bec765249f3`, with `CONFIG_BLK_DEV_UBLK=m`.
- Feature-local kernel preparation succeeded with config hash
  `06595bf2b7228357f569a105b6bbcd26b41a970d798a1beefb69c2ac5bbf5558`.

## Current Verification

- AgentENV release binaries exist at
  `build/agentenv-cargo/release/{server,aenv}` and both `--help` commands pass.
- Bundle/manager/service/config-renderer Python checks pass (`25` focused tests
  in the latest combined run); `py_compile`, `bash -n`, and `git diff --check`
  pass.
- `tools/agentenv_service.py plan` is feature-local and loopback-only.
- `tools/agentenv_service.py start --dry-run` correctly refuses because the
  current host lacks a usable `/dev/ublk-control` and the current user lacks
  usable `/dev/kvm` permissions.
- The isolated kernel build and staging completed successfully. Candidate
  artifacts are under `C:\Users\Admin1\wsl-kernels\agentenv-6.18.40.1`:
  `bzImage` SHA256 `f9bade1cd44bfc266d6ae8fb8214f8f0b5bf0fc3c2664f1f82ec9c9679f3b134`
  and `modules.vhdx` SHA256
  `7cfdef6b8c4f4c419b7f3911fd5553dbd9f204708ec6b2aec5a29d1597504b69`.
- Candidate/active WSL config is `build/agentenv-kernel/wslconfig.candidate` and
  `/mnt/c/Users/Admin1/.wslconfig` with
  SHA256 `f0cc80effc7453bfaf4300824cb68d5a722b020f1250439c0edbcee10cdd5f8c`.
  Its `kernel` and `kernelModules` values contain literal doubled Windows path
  separators. The running kernel is still `6.18.33.2-microsoft-standard-WSL2`
  until the separately approved second shutdown/restart.
- `start-pair` now rolls back already-created sandboxes if a later create or
  local-state write fails; no orphan is intentionally left by the manager.

## Next Action

The next action is the corrected candidate's second shutdown/restart gate. The
assistant must not invoke it: first show a fresh process snapshot and the
rollback command, then ask the user to run `wsl --shutdown` from Windows.
Rollback is:
restore `/mnt/c/Users/Admin1/.wslconfig.pre-agentenv-20260817-529594c.bak` over
the active file, then perform the separately approved shutdown/restart. Do not
start the service or launch a sandbox until the new kernel and device gates are
verified.

## Recovery

If the corrected kernel switch fails, restore the original backup over the
active Windows `.wslconfig`, then perform a separately approved shutdown and
verify the stock `uname -r` before resuming. The first switch attempt fell back
to stock and made no kernel-level change.
