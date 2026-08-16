#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---verify-build}
case "${mode}" in
  --install) tool_mode=install ;;
  --verify-prefix) tool_mode=verify-prefix ;;
  --build) tool_mode=build ;;
  --verify-build) tool_mode=verify-build ;;
  *)
    printf 'usage: %s [--install|--verify-prefix|--build|--verify-build]\n' "${0##*/}" >&2
    exit 2
    ;;
esac
exec /usr/bin/python3 "${root_dir}/tools/rocm_model_build_environment.py" \
  "${tool_mode}" --root "${root_dir}" --jobs 24
