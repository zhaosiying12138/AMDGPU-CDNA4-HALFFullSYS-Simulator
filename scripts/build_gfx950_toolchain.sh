#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
work_dir="${root_dir}/artifacts/build/gfx950-toolchain"
jobs=24

usage() {
  echo "usage: scripts/build_gfx950_toolchain.sh [--work-dir PATH]" >&2
}

while (($#)); do
  case "$1" in
    --work-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      work_dir=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

llvm_source="${root_dir}/projects/llvm-project/llvm"
device_source="${root_dir}/projects/llvm-project/amd/device-libs"
fixture_source="${root_dir}/tests/fixtures/gfx950/vecadd.cl"
llvm_build="${work_dir}/llvm"
device_a="${work_dir}/device-libs-a"
device_b="${work_dir}/device-libs-b"
fixture_dir="${work_dir}/fixtures"

expected_llvm_head=73f2a21fe16b34e35fd0e149564b8664e59da392
actual_llvm_head=$(git -C "${root_dir}/projects/llvm-project" rev-parse HEAD)
if [[ "$actual_llvm_head" != "$expected_llvm_head" ]]; then
  echo "pinned LLVM head mismatch: $actual_llvm_head" >&2
  exit 1
fi

mkdir -p "$work_dir" "$fixture_dir"

cmake --fresh -S "$llvm_source" -B "$llvm_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS='clang;lld' \
  -DLLVM_TARGETS_TO_BUILD=AMDGPU \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_CURL=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF
cmake --build "$llvm_build" --parallel "$jobs" --target \
  clang llvm-link opt llvm-objdump lld FileCheck

build_device_libs() {
  local build_dir=$1
  cmake --fresh -S "$device_source" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$llvm_build" \
    -DLLVM_DIR="$llvm_build/lib/cmake/llvm" \
    -DClang_DIR="$llvm_build/lib/cmake/clang" \
    -DCLANG_OPTIONS_APPEND=-mcpu=gfx950 \
    -DCMAKE_INSTALL_PREFIX="$build_dir/install"
  cmake --build "$build_dir" --parallel "$jobs" --target rocm-device-libs
  cmake --install "$build_dir"
  find "$build_dir/install" -type f -name '*.bc' -print0 | \
    sort -z | xargs -0 sha256sum | \
    sed "s#${build_dir}/install/##" >"$build_dir/bitcode.sha256"
}

build_device_libs "$device_a"
build_device_libs "$device_b"
cmp "$device_a/bitcode.sha256" "$device_b/bitcode.sha256"

compile_fixture() {
  local device_dir=$1
  local output=$2
  "$llvm_build/bin/clang" \
    --target=amdgcn-amd-amdhsa -mcpu=gfx950 -O2 -x cl \
    -Xclang -finclude-default-header \
    --rocm-path="$device_dir/install" \
    "$fixture_source" -o "$output"
}

compile_fixture "$device_a" "$fixture_dir/vecadd-a.hsaco"
compile_fixture "$device_b" "$fixture_dir/vecadd-b.hsaco"
cmp "$fixture_dir/vecadd-a.hsaco" "$fixture_dir/vecadd-b.hsaco"
"$llvm_build/bin/llvm-objdump" -d --mcpu=gfx950 \
  "$fixture_dir/vecadd-a.hsaco" >"$fixture_dir/vecadd.disassembly"

{
  printf 'llvm_head=%s\n' "$actual_llvm_head"
  printf 'jobs=%s\n' "$jobs"
  printf 'clang_sha256='
  sha256sum "$llvm_build/bin/clang" | cut -d' ' -f1
  printf 'lld_sha256='
  sha256sum "$llvm_build/bin/lld" | cut -d' ' -f1
  printf 'device_libs_manifest_sha256='
  sha256sum "$device_a/bitcode.sha256" | cut -d' ' -f1
  printf 'vecadd_source_sha256='
  sha256sum "$fixture_source" | cut -d' ' -f1
  printf 'vecadd_hsaco_sha256='
  sha256sum "$fixture_dir/vecadd-a.hsaco" | cut -d' ' -f1
} >"$work_dir/manifest.txt"

cat "$work_dir/manifest.txt"
