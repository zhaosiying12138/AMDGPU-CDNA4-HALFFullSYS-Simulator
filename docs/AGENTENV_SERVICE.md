# AgentENV Service And Activation Gate

The feature branch keeps the AgentENV server and all generated state below
`build/agentenv-integration/server`. The service binds only to
`127.0.0.1:18080`; it does not install a system service, edit the active
Windows `.wslconfig`, or invoke `wsl --shutdown`.

## Inspect

```bash
tools/agentenv_service.py plan
tools/agentenv_service.py status
tools/agentenv_manager.py --json host-preflight
```

The service uses `build/agentenv-cargo/release/server` and generates a
feature-local `server.env`. Source it before using the CLI:

```bash
source build/agentenv-integration/server/server.env
aenv --help
```

## Start And Stop

Start is deliberately gated by `/dev/kvm` and `/dev/ublk-control`:

```bash
tools/agentenv_service.py start --dry-run
tools/agentenv_service.py start
tools/agentenv_service.py status
```

The current host is expected to fail the dry-run until the candidate WSL
kernel is activated and device permissions are available. `--allow-missing-prereqs`
is diagnostic only and must not be used as an acceptance claim.

```bash
tools/agentenv_service.py stop --dry-run
tools/agentenv_service.py stop --confirm
```

Stop validates PID, Linux start-time ticks, and executable identity before it
signals a process. Never stop by PID alone.

## Kernel Gate

The pinned source is `projects/WSL2-Linux-Kernel` at
`14794180686c2fb6307fbe359c359bec765249f3` (release `6.18.40.1`). Build and
stage are feature-local:

```bash
tools/build_agentenv_wsl_kernel.sh prepare
tools/build_agentenv_wsl_kernel.sh build
tools/build_agentenv_wsl_kernel.sh stage
```

`stage` writes a candidate kernel/modules directory and a candidate WSL config
under `build/agentenv-kernel`; it does not change the active `.wslconfig`.
Before any activation, record the running process snapshot, current config
hash/content, candidate diff, and rollback. Tell the user before invoking
`wsl --shutdown`, then wait for explicit approval.

## Two Sandbox Plan

After the service and bundle gates pass, use the manager in dry-run mode first:

```bash
tools/agentenv_bundle.py --dry-run --json
tools/agentenv_manager.py --dry-run --json start-pair \
  --template agentenv-qwen35-runtime
```

The pair is `vllm-tp4` and `sglang-tp1`. Each instance receives independent
tmp/cache/runtime/socket/log namespaces. The manager refuses unrelated live
gem5/vLLM/SGLang processes by default and records workload commands; it does
not silently repair or claim model inference correctness.
