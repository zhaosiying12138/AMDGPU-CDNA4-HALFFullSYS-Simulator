#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
gem5_dir="${root_dir}/projects/gem5"
scons_bin="${GEM5_SCONS:-${root_dir}/env/gem5-build/bin/scons}"
jobs="${GEM5_JOBS:-24}"

if [[ ! -x "${scons_bin}" ]]; then
  scons_bin=$(command -v scons || true)
fi
if [[ -z "${scons_bin}" || ! -x "${scons_bin}" ]]; then
  printf 'gem5 SCons executable not found; set GEM5_SCONS\n' >&2
  exit 1
fi
if ! command -v mold >/dev/null 2>&1; then
  printf 'mold executable not found; install mold or adjust PATH\n' >&2
  exit 1
fi
if [[ ! "${jobs}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GEM5_JOBS must be a positive integer\n' >&2
  exit 2
fi

cd "${gem5_dir}"
exec env "PATH=$(dirname "${scons_bin}"):${PATH}" \
  "${scons_bin}" -j"${jobs}" --linker=mold \
  "${@:-build/VEGA_X86/gem5.opt}"
