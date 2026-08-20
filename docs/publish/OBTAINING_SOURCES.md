# Obtaining the sources, models, and rebuildable artifacts

The public repository carries the control plane (root repo contents), the
project-authored patch series for each upstream lane, and the full
`self-amdgpu-runtime` sources. Everything else is either a pinned upstream
checkout or a rebuildable artifact and is deliberately not stored in Git.

## Upstream source lanes

Each lane is a standard upstream clone at a pinned baseline plus the patch
series under `patches/<lane>/`. `SOURCE_LOCK.json` is the authority for
baseline commits, tags, and upstream URLs.

| Lane | Upstream | Baseline commit | Patches |
| --- | --- | --- | --- |
| gem5 | https://github.com/gem5/gem5.git (`stable`) | `cbf0eae213c5e39c727172b546434287d47b5bbe` | `patches/gem5/` |
| rocm-systems | https://github.com/ROCm/rocm-systems.git (`develop`) | `92115a2941982a384de161be3f78cf9bff547027` | `patches/rocm-systems/` |
| llvm-project | https://github.com/ROCm/llvm-project.git (`amd-staging`) | `73f2a21fe16b34e35fd0e149564b8664e59da392` | none (used as toolchain) |
| pytorch | https://github.com/pytorch/pytorch.git | `411e87a93704f547e5146c74c95fa11acf13d646` | none |
| triton | https://github.com/triton-lang/triton.git | `cd513e2798db0f4675b3d1205c8e76eb3381a0b3` | none |
| vllm | https://github.com/vllm-project/vllm.git | `8d9b52f7c2514490bdadfd5eb0c931e58625df2e` | none |

Reconstruct a lane:

```bash
git clone <upstream-url> projects/<lane>
git -C projects/<lane> checkout <baseline-commit>
git -C projects/<lane> am ../../patches/<lane>/*.patch
```

The patch series are linear (no merges) and apply cleanly on the recorded
baselines. SGLang 0.5.17 is used unchanged from upstream (byte-identical;
the acceptance protocol forbids modifying it) and is therefore not carried
here at all: `pip download sglang==0.5.17` or clone the upstream tag.

## self-amdgpu-runtime

The project-authored HSA/KMT model runtime is published as the
`self-amdgpu-runtime` branch of this same repository (full independent
history; the root gitlink records the exact head). Reconstruct the
submodule checkout with:

```bash
git clone -b self-amdgpu-runtime <this-repo-url> projects/self-amdgpu-runtime
```

It builds with the pinned ROCm LLVM toolchain; see `docs/gem5-build.md`
and `scripts/`.

## Models

`models/Qwen3.5-0.8B` is the official `Qwen/Qwen3.5-0.8B` checkpoint at the
pinned revision recorded by `tools/lock_model_source.py` output in
`state/`. Download with `scripts/download_model.py` or from Hugging Face
directly; the lock file verifies the exact file hashes.

## Rebuildable artifacts (never committed)

- `env/` — conda environments and content-addressed ROCm products; built by
  `scripts/setup_rocm_env.sh` and `scripts/setup_framework_env.sh`.
- `build/`, `projects/gem5/build/` — gem5 and runtime build outputs;
  `scripts/build_gem5_lld24.sh` rebuilds `gem5.opt` (plus the unit test
  binaries) reproducibly with ccache/clang/lld.
- `artifacts/` — run evidence, lane logs, capsules, and NVIDIA diagnostic
  goldens. Accepted-evidence bundles record SHA-256 manifests in commit
  trailers; the runs regenerate them.
- NVIDIA operator goldens are diagnostic only and are regenerated with
  `tools/qwen35_nvidia_operator_golden.py` in a separate CUDA environment;
  they never gate acceptance.
