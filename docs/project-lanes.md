# Project-authored lanes

`PROJECT_LANES.json` records immutable baselines for repositories created by
this project. It is deliberately separate from `SOURCE_LOCK.json`: an authored
repository has no upstream revision to resolve, and inventing upstream
provenance would weaken the audit trail.

Each lane declares a safe project ID, a mode-160000 path, its absorbed Git
administration path, a sibling-relative origin policy, baseline commit/tree,
annotated tag object and exact payload, baseline checkpoint, and license-file
hash. The first lane is `self-amdgpu-runtime`, whose CP-0003 initial commit
contains only the stable version/status C ABI foundation and build/package
tests. Transport, queue, memory, code-object, and dispatch behavior begin in
later descendant commits; the baseline makes no claim that they already exist.

The current commit and tree do not belong in the registry. Every accepted
checkpoint owns those fields and must satisfy all of the following:

- the current commit is a descendant of the immutable baseline;
- the root gitlink, child `HEAD`, and checkpoint commit/tree agree;
- the child is clean, non-shallow, has no alternate object dependency, and
  uses the exact absorbed `.git/modules/<id>` administration directory;
- the annotated baseline tag, commit trailers, origin/push policy, and license
  bytes still match the registry;
- `.gitmodules` is the exact union of upstream and project-authored lanes.

Append-only behavior is checked against Git history. The verifier finds the
coordinator commit where a lane first appeared, reads that commit's registry
and checkpoint blobs, and requires the current lane object to match the first
declaration. Adding another lane changes the registry file without granting
permission to rewrite an existing lane's baseline, evidence, tag, or license.

The sibling-relative URL `../self-amdgpu-runtime.git` is a reproducible bundle
layout contract, not a claim that a remote is currently reachable. Publication
must create that sibling repository before users rely on network materialization;
until then `origin` has `pushurl=no_push` and cannot be used accidentally.
