# Active checkpoint: generic model-backed fast copy

Updated: 2026-08-16 Asia/Shanghai

## Current State

The feature is in isolated implementation. The original model worktree is
intentionally untouched and may contain live tests. All feature work belongs
to `/home/zhaosiying/amdgpu-sim-fastcopy`.

ROCr and CLR already have `HSA_ENABLE_DTIF_FAST_COPY`. The generated model
topology exposes the ordinary GPU heap as `FRAME_BUFFER_PUBLIC`; ROCr propagates
`HostAccess=1`, and libhsakmt maps the model render fd at the GPU VA as shared
read/write memory. Qwen weight allocations use this public class. Some internal
VRAM allocations remain private, so an unconditional process-wide raw copy is
still too broad.

The selected implementation reuses the upstream direct path and adds a second
explicit provider authorization plus generic range/type eligibility and
ordering fallbacks. It does not add a provider callback, a KMT opcode, a bridge
message, or a gem5 data-copy implementation.

The root product generators now export both
`HSA_ENABLE_DTIF_FAST_COPY=0` and
`SAGR_HSAKMT_MODEL_FAST_COPY=0`. Activation therefore remains legacy by
default and preserves/restores the provider gate. The focused product tests
pass:

```bash
python3 -m unittest \
  tests.test_product_environment \
  tests.test_conda_product_environment \
  tests.test_rocm_pytorch_product_environment
# 25 tests passed
```

The current shell can select an A/B mode without changing the product:

```bash
source scripts/fastcopy_mode.sh legacy
source scripts/fastcopy_mode.sh fast
```

ROCm Systems commit `4c7de84a26` implements the first runtime slice: model
mode no longer defaults fast copy on, and both ROCr and CLR require the two
explicit gates. Outside model mode, the existing explicit HSA switch retains
its prior behavior. Syntax checks passed for both modified translation units.
ROCm Systems commit `61aa4be538` implements the per-operation H2D/D2H
eligibility and dependency/gang/profiling fallback in four ROCr files. A
focused read-only review found no correctness blocker. The changed ROCr
objects compile in the isolated build when `HIP_DEVICE_LIB_PATH` points at the
locked ROCm 7.2.3 bitcode directory; only pre-existing warnings remain.

The model acceptance rule was narrowed on 2026-08-16. Existing authoritative
`GOAL.md` explicitly says that no full-model vLLM TP2/TP4 result is claimed,
and `PLAN.md` still lists unchanged-upstream vLLM TP1 as future bring-up. A
fresh vLLM preflight found no independently passing full-model TP1 baseline:
the retained records are weight-registration or constrained layer/forward
probes, while the BF16/full-prefill/decode records contain
`output_correct=false`. Therefore this feature skips model E2E and will not
fix vLLM or SGLang. Model-level loaded-weight/inference equality and
weight-load speedup remain unverified; bounded runtime byte/path evidence is
the terminal validation. The detailed evidence and future A/B procedure are
in `docs/fastcopy-vllm-integration.md`.

The exact changed-object command that passes is:

```bash
export HIP_DEVICE_LIB_PATH=/home/zhaosiying/amdgpu-sim/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d/rocm-sysroot/opt/rocm-7.2.3/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
/usr/bin/ninja \
  -C /home/zhaosiying/amdgpu-sim-fastcopy/build/fastcopy-rocr-eligibility \
  -j6 \
  runtime/hsa-runtime/CMakeFiles/hsa-runtime64.dir/core/runtime/amd_gpu_agent.cpp.o \
  runtime/hsa-runtime/CMakeFiles/hsa-runtime64.dir/core/runtime/amd_blit_kernel.cpp.o \
  runtime/hsa-runtime/CMakeFiles/hsa-runtime64.dir/core/runtime/amd_blit_sdma.cpp.o
```

The isolated ROCr library now links successfully at
`build/fastcopy-rocr-eligibility-stage/lib/libhsa-runtime64.so.1.21.0`.
The isolated self-runtime model/runtime and upstream ROCr worker targets also
build under `build/fastcopy-runtime`. The new
`scripts/run_fastcopy_rocr_probe.py` runs the same small upstream HSA worker in
`legacy`, `hsa-only`, or `fast` mode, with a private endpoint, private process groups, and
feature-local gem5/output paths. It also accepts an explicit `--worker`, repeated
`--worker-arg`, and `--allow-idle-gem5` for standalone runtime probes. It records
worker/gem5 logs, trace rows, retired dispatch count, and byte-check exit status.

## Focused Probe Evidence

The same worker and `gpuReadWrite_kernels.hsaco` were run with a private
feature-local gem5 endpoint:

| mode | gates | worker/gem5 | retired dispatches | elapsed |
| --- | --- | --- | ---: | ---: |
| legacy | HSA=0, provider=0 | 0 / 0 | 3 | 1.958 s |
| hsa-only | HSA=1, provider=0 | 0 / 0 | 3 | 1.988 s |
| fast | HSA=1, provider=1 | 0 / 0 | 1 | 0.984 s |

The legacy trace contains two 65536x1x1 copy dispatches plus the 64-element
user kernel. The fast trace contains only the 64-element user kernel. All three
workers print `upstream ROCr standard AQL execution passed`, and the worker's
H2D initialization plus D2H readback checks are byte-exact. Run artifacts are
preserved at:

- `/tmp/fastcopy-probe-legacy-20260816-b`
- `/tmp/fastcopy-probe-hsa-only-20260816`
- `/tmp/fastcopy-probe-fast-20260816`

This is a 256-byte functional probe (64 int32 elements), not a
model-throughput claim. The separate 2 MiB probe below supplies the bounded
MB-scale data-plane evidence; no full-model vLLM TP1 baseline exists in the
repository, so Qwen end-to-end validation is intentionally skipped rather than
repaired here.

The standalone 2 MiB probe is a generic HSA runtime API test. It allocates a CPU
source/destination and a GPU global pool buffer, performs byte-exact synchronous
H2D/D2H, and can issue a ready non-empty dependency copy. Results:

| mode | gates | worker/gem5 | retired dispatches | elapsed | result |
| --- | --- | ---: | ---: | ---: | --- |
| legacy | HSA=0, provider=0 | 0 / 0 | 2 | 2.837 s | exact |
| hsa-only | HSA=1, provider=0 | 0 / 0 | 2 | 3.241 s | exact |
| fast | HSA=1, provider=1 | 0 / expected-idle | 0 | 20.831 s* | exact |
| fast + ready dependency | HSA=1, provider=1 | 0 / 0 | 1 | 2.088 s | exact |

The fast pure-copy run has no AQL packet, so `--allow-idle-gem5` terminates the
private idle simulator after a bounded grace period; its elapsed value is not a
copy-throughput measurement. The ready dependency run retires one legacy copy,
confirming that a non-empty dependency does not enter the raw memcpy path. An
unsatisfied dependency (initial 1, then host store to 0) did not wake the current
model bridge; that existing queue-wakeup capability is recorded as unverified and
is not used to claim fast-copy correctness.

Artifacts:

- `/tmp/fastcopy-probe-large-legacy-20260816-c`
- `/tmp/fastcopy-probe-large-hsa-only-20260816`
- `/tmp/fastcopy-probe-large-fast-20260816-b`
- `/tmp/fastcopy-probe-large-fast-dependency-20260816`

## Decisions Frozen At This Checkpoint

- Generic runtime API boundary; no framework/model/operator special cases.
- Two explicit opt-in gates; legacy remains default.
- Reuse the existing PUBLIC mapping; do not change private mappings to RW.
- Unsupported pointer kinds and semantic constraints fall back to AQL.
- Known allocation overruns fail closed.
- No required gem5 implementation change for the data path.
- Focused probes precede any 25-30 minute model load.
- Final model A/B is conditional on an already passing vLLM TP1 baseline.
  Missing baseline means skip, not framework repair or scope expansion.
- Byte-exact runtime copy checks remain mandatory in either case.
- The 2 MiB probe is the terminal data-plane scale evidence for this branch;
  its wall time includes simulator startup/idle cleanup and is not a model-load
  speedup claim.
- A full vLLM TP1 baseline is absent, so no Qwen weight-load A/B was started;
  this is a deliberate scope stop, not a vLLM failure repair.

## Resume From Zero Context

```bash
cd /home/zhaosiying/amdgpu-sim-fastcopy
sed -n '1,260p' FASTCOPY_CHECKPOINT.md
sed -n '1,240p' FASTCOPY_GOAL.md
sed -n '1,260p' FASTCOPY_PLAN.md
sed -n '1,240p' FASTCOPY_LESSONS.md
sed -n '1,260p' docs/fastcopy-vllm-integration.md
git status --short --branch
git -C projects/rocm-systems status --short --branch
git -C projects/self-amdgpu-runtime status --short --branch
git -C projects/gem5 status --short --branch
```

The current feature-local ROCr, self-runtime, and probe builds are already
complete. First verify the clean statuses and rerun only the bounded probe if
new evidence is needed. Do not schedule a model run: the independently
passing vLLM TP1 baseline is absent. Before running anything, confirm no
command points to `/home/zhaosiying/amdgpu-sim/projects/*/build`.

The standalone 2 MiB target is rebuilt with:

```bash
cmake --build /home/zhaosiying/amdgpu-sim-fastcopy/build/fastcopy-runtime \
  --target self_amdgpu_runtime_upstream_rocr_fastcopy_probe --parallel 6
python3 scripts/run_fastcopy_rocr_probe.py fast \
  --output /tmp/fastcopy-probe-large-fast-next \
  --worker build/fastcopy-runtime/tests/self_amdgpu_runtime_upstream_rocr_fastcopy_probe \
  --allow-idle-gem5
```

## Last Known Commits

- Root feature parent before CP-0048: `8d198e0a95dcd5bb238746c5783e88b3cc17d135`
- Root checkpoint CP-0048: `21351eb082eeb773f2af4f15d3e599df944772af`
- Root base: `fc0a7f34d3059e59255b6ddc656c914ab21d2656`
- ROCm Systems base: `92115a2941982a384de161be3f78cf9bff547027`
- Self-runtime base: `65bdf9669421abb7f380ac154ca237c3ddf6891a`
- gem5 base: `ee05c9a3eef3771afbb370d212dee83ea4a7ecfc`
- ROCm Systems fast-copy gate: `4c7de84a26941fd851d9c8a13b9c97349de371f6`
- ROCm Systems eligibility: `61aa4be538ed85e23a8a72b5437b327b1e50c201`
- Self-runtime fast-copy probe: `a9eeeea43474e28a4476d072555c2025a01899de`.

At the final handoff, resolve the current root feature commit with
`git rev-parse HEAD`; the feature and all nested branches must report clean.
