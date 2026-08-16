#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---verify}
case "${mode}" in
  --install|--verify|--print-prefix) ;;
  *) printf 'usage: %s [--install|--verify|--print-prefix]\n' "${0##*/}" >&2; exit 2 ;;
esac
exec /usr/bin/python3 "${root_dir}/tools/conda_product_environment.py" \
  "${mode}" --root "${root_dir}"
