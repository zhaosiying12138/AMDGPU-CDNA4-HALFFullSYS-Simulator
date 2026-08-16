#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
gem5_dir="${root_dir}/projects/gem5"
scons_bin="${GEM5_SCONS:-${root_dir}/env/gem5-build/bin/scons}"
sysroot_bin="${root_dir}/env/gem5-build/sysroot/usr/bin"
ccache_dir="${CCACHE_DIR:-${HOME}/.cache/amdgpu-sim/ccache}"
python_config=$(command -v python3-config || command -v python-config || true)

if [[ "${GEM5_JOBS:-24}" != 24 ]]; then
  printf 'amdgpu-sim policy requires GEM5_JOBS=24\n' >&2
  exit 2
fi
if [[ ! -x "${scons_bin}" ]]; then
  scons_bin=$(command -v scons || true)
fi
if [[ -z "${scons_bin}" || ! -x "${scons_bin}" ]]; then
  printf 'gem5 SCons executable not found; set GEM5_SCONS\n' >&2
  exit 1
fi
if [[ -z "${python_config}" || ! -x "${python_config}" ]]; then
  printf 'python development config tool is unavailable\n' >&2
  exit 1
fi
for tool in /usr/bin/ccache /usr/lib/ccache/clang /usr/lib/ccache/clang++ /usr/bin/ld.lld; do
  if [[ ! -x "${tool}" ]]; then
    printf 'required build tool is unavailable: %s\n' "${tool}" >&2
    exit 1
  fi
done

mkdir -p "${ccache_dir}"
CCACHE_DIR="${ccache_dir}" /usr/bin/ccache --max-size 50G >/dev/null
cd "${gem5_dir}"
if [[ -d "${sysroot_bin}" ]]; then
  tool_path="${sysroot_bin}:$(dirname "${scons_bin}"):$(dirname "${python_config}"):/usr/lib/ccache:/usr/bin:/bin"
else
  tool_path="$(dirname "${scons_bin}"):$(dirname "${python_config}"):/usr/lib/ccache:/usr/bin:/bin"
fi
if (( $# == 0 )); then
  set -- build/VEGA_X86/gem5.opt
fi
exec env \
  "PATH=${tool_path}" \
  "CCACHE_DIR=${ccache_dir}" \
  "CCACHE_COMPILERCHECK=content" \
  "CCACHE_BASEDIR=${root_dir}" \
  "CCACHE_MAXSIZE=50G" \
  "CC=/usr/lib/ccache/clang" \
  "CXX=/usr/lib/ccache/clang++" \
  "AR=/usr/bin/llvm-ar-21" \
  "RANLIB=/usr/bin/llvm-ranlib-21" \
  "${scons_bin}" -j24 --linker=lld --ignore-style "$@"
