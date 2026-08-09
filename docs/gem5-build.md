# Reproducible gem5 build and link

CP-0012 and later gem5 acceptance builds use SCons' native mold selector and
24 build jobs.  From the repository root, the checked-in wrapper is:

```bash
scripts/build_gem5_mold24.sh
```

It defaults to `build/VEGA_X86/gem5.opt` and accepts additional SCons targets.
The wrapper also prepends the environment's `sysroot/usr/bin` so generated
tools such as `m4` are found consistently when the host PATH does not contain
them.
The equivalent command, run from `projects/gem5`, is:

```bash
PATH="$PWD/../../env/gem5-build/sysroot/usr/bin:$PWD/../../env/gem5-build/bin:$PATH" \
  "$PWD/../../env/gem5-build/bin/scons" -j24 --linker=mold \
  build/VEGA_X86/gem5.opt
```

This is the canonical full build/link command for the host-side AMDGPU daemon.
Do not replace `--linker=mold` with an unrecorded `LINKFLAGS` override: gem5's
`SConstruct` validates `-fuse-ld=mold` and preserves the selected linker in the
SCons configuration.  `-j24` controls SCons compilation concurrency; mold
performs the final link internally in parallel.

Before an acceptance link, record:

```bash
command -v mold
mold --version
scons --version
```

After the link, do not run SCons again before capturing the final artifact.
Record the binary identity and confirm every changed source/configuration file
is older than the corresponding object or final executable:

```bash
stat projects/gem5/build/VEGA_X86/gem5.opt
sha256sum projects/gem5/build/VEGA_X86/gem5.opt
projects/gem5/build/VEGA_X86/gem5.opt -B
```

The CP evidence must retain the exact SCons argv, stdout/stderr, exit status,
elapsed time, mold/SCons versions, binary size and SHA-256, and the source ->
object -> executable freshness relationship.  A binary built before the last
source change, or relinked after its identity capture, is stale evidence.

The CP-0012 frozen link used the same invocation with the environment-local
SCons executable and produced:

```text
gem5.opt SHA-256: a594ee36ed413907c90bf1d0e895a058fa82d2258d68e73955c07fc7c40f4218
size: 1185783248 bytes
Build ID: 416ee4e8ffa6136f864bcdc5554eb5e81c2c5a62
mold: 2.40.4
```

The retained command transcript is under
`artifacts/evidence/CP-0012/gem5-mold24-final.log` (ignored evidence storage);
the durable method is this document and
`scripts/build_gem5_mold24.sh`, not the generated binary.
