# AgentENV Manager

`tools/agentenv_manager.py` is the host-side controller for this feature
worktree. It does not change the upstream `projects/AgentENV` source and it
does not run workloads on the host. Runtime state is written below
`build/agentenv-integration/state` unless `--state-dir` is supplied for an
isolated test run.

The manager talks to the loopback AgentENV API with the Python standard
library. It records a distinct namespace, temporary directory, runtime
directory, cache directory, socket directory, and endpoint directory for each
instance. The manager does not claim model correctness; workload commands are
reported as launch commands for the existing vLLM TP4 and SGLang TP1 entry
points.

## Read-only checks

```bash
tools/agentenv_manager.py --json host-preflight
tools/agentenv_manager.py --json status --offline
```

Use `--check-api` for a loopback health check. `--strict` makes
`host-preflight` fail when the kernel or `/dev/kvm` prerequisite is missing.

## Plan and start

First create the deterministic runtime bundle with
[`tools/agentenv_bundle.py`](../tools/agentenv_bundle.py), then plan the pair:

```bash
tools/agentenv_bundle.py --dry-run --json
tools/agentenv_manager.py \
  --dry-run --json start-pair \
  --template agentenv-qwen35-runtime \
  --bundle-manifest build/agentenv-runtime-bundle.tar.zst.manifest.json
```

The live command refuses to proceed if unrelated gem5, vLLM, or SGLang
processes are visible in `/proc`. Inspect the report and pass `--allow-live`
only after deliberately accepting that boundary:

```bash
tools/agentenv_manager.py --json start-pair \
  --template agentenv-qwen35-runtime \
  --bundle-manifest build/agentenv-runtime-bundle.tar.zst.manifest.json \
  --allow-live
```

The manager creates two records, `vllm-tp4` and `sglang-tp1`, and persists the
returned sandbox IDs. It does not automatically execute the model commands;
the recorded commands are:

```text
scripts/run_qwen35_vllm_tp.py --tensor-parallel-size 4
examples/sglang/qwen35_inference.py --tp-size 1
```

Run them through AgentENV's `aenv exec` or the API from inside each recorded
sandbox after the sandbox boot and bundle extraction gates have passed.

## Collect and stop

```bash
tools/agentenv_manager.py --json collect --offline \
  --output collect-report.json
tools/agentenv_manager.py --json status --offline
tools/agentenv_manager.py --dry-run --json stop
tools/agentenv_manager.py --json stop --confirm
```

`stop` is destructive by design and requires `--confirm`; without it, the
command only prints the target IDs. Collection is best-effort and preserves
the local JSON report even when the API is unavailable.

Do not run `wsl --shutdown` as part of this workflow. Kernel switching remains
a separately approved maintenance action because it terminates unrelated WSL
workloads.
