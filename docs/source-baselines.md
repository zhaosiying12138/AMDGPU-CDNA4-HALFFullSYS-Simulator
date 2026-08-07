# Frozen upstream baselines

CP-0002 resolves each source from its official project identity and records the
full commit and tree objects in `SOURCE_LOCK.json`. Canonical HTTPS URLs remain
in `.gitmodules`; `transport_url` records the SSH-over-443 route used on this
machine. Transport is not evidence of ownership or source identity.

| Lane | Official ref | Frozen commit | Purpose |
| --- | --- | --- | --- |
| gem5 | `stable` | `cbf0eae213c5e39c727172b546434287d47b5bbe` | GPU simulator and future host bridge |
| ROCm systems | `develop` | `92115a2941982a384de161be3f78cf9bff547027` | ROCr, libhsakmt, HIP/CLR, and RCCL source of truth |
| ROCm LLVM | `amd-staging` | `73f2a21fe16b34e35fd0e149564b8664e59da392` | LLVM, Clang, LLD, and active AMD device libraries |
| Triton | `main` | `cd513e2798db0f4675b3d1205c8e76eb3381a0b3` | Requested latest compiler/backend development lane |
| PyTorch | `main` | `411e87a93704f547e5146c74c95fa11acf13d646` | Tensor runtime integration lane |
| vLLM | `main` | `8d9b52f7c2514490bdadfd5eb0c931e58625df2e` | Inference and N-rank worker integration lane |
| Qwen | `main` | `2fc06364715b967f1860aea9cf38778875588b17` | Official `Qwen/Qwen3.5-0.8B` acceptance model |

Each Git lane is a non-shallow partial clone with all commits reachable from the
locked head locally traversable. The lock records whether unused historical
objects are filtered with `blob:none` or `tree:0`; it does not claim those
historical trees and blobs are all local. The complete locked baseline tree and
every named compatibility tree are present locally and must pass an offline
`git archive` check with lazy fetching disabled. Each checkout has an annotated
`upstream-baseline/<lane>/<full-commit>` tag whose complete tag object and
payload are frozen in the lock.

The reviewed commit also has a local
`refs/amdgpu-sim/resolved/<upstream-ref>` anchor. Materialization fetches that
exact 40-byte object rather than following the branch again. A later upstream
advance is expected and does not mutate the lock; it must be considered in a
new checkpoint if the project deliberately upgrades its baseline.

## Triton compatibility lane

The frozen PyTorch commit pins Triton
`675c59878aa2280b31f722aaf42b825fcee21de8`. It is not an ancestor of the
latest Triton baseline; their merge base is
`3bb23dde71a3679f0ae3e9c1be6dcae1f4bff462`. The Triton checkout therefore
retains both offline-complete revisions:

- latest development baseline under its `upstream-baseline` tag;
- the PyTorch-required revision under
  `compatibility/pytorch/411e87a/675c59878aa2280b31f722aaf42b825fcee21de8`.

Environment locks must select one explicitly. They may not assume that testing
latest Triton proves compatibility with the frozen PyTorch source.

## Model provenance

The model itself is not a Git submodule and no weights are committed. The lock
contains the official fixed revision, an exact 13-file inventory, Git blob IDs,
and LFS SHA-256 values. `scripts/download_model.py` downloads only that revision
into ignored `models/` storage, rejects unexpected files and symlinks, verifies
every size and content identity, and publishes the snapshot by atomic rename.
A mirror requires an explicit flag and remains marked as mirror provenance.

## Current closure boundary

This checkpoint proves source identity, pristine history ancestry, license-file
identity, and an offline-complete locked root tree. It deliberately does not
initialize the many nested upstream submodules. Consequently it is a source
baseline gate, not yet an offline build closure. Required nested dependencies
must be selected, independently locked, and evidenced before any build is
claimed reproducible.

For `rocm-systems` specifically, the locked root `.gitmodules` declares 36
paths while the same commit contains 34 actual mode-160000 gitlinks. The stale
declarations are `projects/rccl/ext-src/mscclpp` and
`projects/rccl/ext-src/json`; neither path exists in the locked tree, and the
selected `projects/rccl/.gitmodules` is empty. Source preparation therefore
intersects declarations with actual locked gitlinks and never initializes a
path from configuration metadata alone.

Project-authored repositories are intentionally outside this upstream lock.
Their immutable initial identities live in `PROJECT_LANES.json`, while accepted
checkpoint repository maps carry current descendant heads. See
`docs/project-lanes.md` for that ownership and recovery contract.
