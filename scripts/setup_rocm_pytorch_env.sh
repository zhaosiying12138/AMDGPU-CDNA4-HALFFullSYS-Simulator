#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---verify}
case "${mode}" in
  --build|--verify|--print-prefix) ;;
  *) printf 'usage: %s [--build|--verify|--print-prefix] [--profile rocm713|vllm-rocm723]\n' "${0##*/}" >&2; exit 2 ;;
esac
shift $(( $# > 0 ? 1 : 0 ))
exec /usr/bin/python3 "${root_dir}/tools/rocm_pytorch_product_environment.py" \
  "${mode}" --root "${root_dir}" "$@"
