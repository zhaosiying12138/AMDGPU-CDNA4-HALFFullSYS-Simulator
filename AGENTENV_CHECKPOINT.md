# AgentENV Integration Checkpoint

## Resume Order

Read this file first, then `AGENTENV_GOAL.md`, `AGENTENV_PLAN.md`, and
`AGENTENV_BITLESSONS.md`. Do not infer runtime acceptance from the source
commits or from a dry-run report.

## Current State

- Date: 2026-08-17
- Integration worktree/branch: `/home/zhaosiying/amdgpu-sim`, `main`
- The AgentENV feature history is integrated on main; the original feature
  worktree remains a build-artifact source, not the execution authority.
- Nested AgentENV pin: `d2405769618c37de5e181a94dc4052d786ec041a`
- Nested WSL kernel pin: `14794180686c2fb6307fbe359c359bec765249f3` (`6.18.40.1`)
- The user completed the successful restart. The running kernel is
  `6.18.40.1-aenv-ublk-6.18.40.1`; KVM, ublk, and TUN are available in the
  current session.
- WSL 2.7.10 exposed the original modules VHD one release directory too deep.
  Current-session compatibility links allowed the required modules to load.
  Commit `dab0504` builds a dual-layout replacement VHD for the next
  maintenance restart; current model work must not wait for or request it.
- Original config backup:
  `/mnt/c/Users/Admin1/.wslconfig.pre-agentenv-20260817-529594c.bak`
  (SHA256 `24177c675423be250afdc6a47422af1ccba9e372a32f00619e7578180c75a0e2`).
- Rejected single-backslash config backup:
  `/mnt/c/Users/Admin1/.wslconfig.invalid-single-backslash-20260817-dda4293.bak`
  (SHA256 `e2bedfc6aede345a159b928e3baf9b1a348ebf0ee8d4a6a80c11f517e7051341`).
- Previous active dual-path config backup:
  `/mnt/c/Users/Admin1/.wslconfig.pre-agentenv-dual-v1-20260817-1322.bak`
  (SHA256 `f0cc80effc7453bfaf4300824cb68d5a722b020f1250439c0edbcee10cdd5f8c`).
- The active `.wslconfig` now points at
  `C:\\Users\\Admin1\\wsl-kernels\\agentenv-6.18.40.1-dual-v1` and has SHA256
  `0472c27770618bc54983b21cc959c04799274d56ee381e0b6c52d628765238bf`.
- One feature-local AgentENV service smoke was run. `/health`, KVM, ublk, and
  the ublk daemon passed, but sandbox lifecycle did not; all service,
  Firecracker, ublk-daemon, listener, veth, and network-namespace state from
  that smoke was cleaned before host model work resumed.

## Host Preflight Snapshot

- WSL kernel: `6.18.40.1-aenv-ublk-6.18.40.1`.
- `/dev/kvm` and `/dev/ublk-control`: present and openable read/write by the
  current user for this session.
- `ublk_drv`, `kvm`, `kvm_intel`, and `tun` loaded successfully.
- `net.ipv4.ip_forward` was restored to its original value `0` after smoke
  cleanup.
- Dual-layout staged artifacts:
  `bzImage` SHA256
  `f9bade1cd44bfc266d6ae8fb8214f8f0b5bf0fc3c2664f1f82ec9c9679f3b134`;
  `modules.vhdx` SHA256
  `dbc933c855534ee84dff1b066507db3c687924dcfafaadc6d4e6669923f661a7`.

## Current Verification

- AgentENV release binaries exist at
  `build/agentenv-cargo/release/{server,aenv}` and both `--help` commands pass.
- Bundle/manager/service/config-renderer Python checks pass. The bundler now
  preserves in-root package-provided dangling symlinks while rejecting paths
  escaping allowed roots, and the service renders zero warm-pool watermarks to
  avoid speculative Firecracker/ublk process churn.
- `tools/agentenv_service.py plan` is feature-local and loopback-only.
- `tools/agentenv_service.py start --dry-run` sees no KVM/ublk prerequisite
  blockers. The live service reached HTTP 204 on `/health` and its ublk daemon
  reported features `0x7fff`.
- Remaining lifecycle blockers are bounded: the capability-bearing,
  non-dumpable server cannot be identity-checked through the current
  `/proc/PID/exe` ownership rule, and Firecracker cold-start readiness has not
  passed. They are AgentENV lifecycle work, not model-runtime failures.
- `start-pair` now rolls back already-created sandboxes if a later create or
  local-state write fails; no orphan is intentionally left by the manager.
- Smoke evidence is under
  `build/agentenv-integration/state/agentenv-runtime-smoke-20260817.json` and
  `build/agentenv-integration/server/logs/server.log`.

## Next Action

Do not request another restart while the user is away. Continue unchanged-
upstream SGLang/vLLM TP1 on the host, serially unless resource isolation is
independently proven. Repair the AgentENV process-ownership and cold-start
gates in parallel only when that work is bounded and useful; retry one sandbox
lifecycle before moving a model back into AgentENV.

## Recovery

If a future maintenance restart fails, restore either the original backup or
the immediately previous dual-path backup over the active Windows
`.wslconfig`, then perform a separately approved shutdown and verify the
expected kernel. Never couple that maintenance gate to current host model
bring-up.
