#!/usr/bin/env bash
# Report simulated AMDGPU targets through the product-owned rocminfo tool.
#
# LLVM's upstream amdgpu-arch asks the host for a real offload device. In this
# product the device is described by the topology-backed rocminfo facade, so
# use that same source of truth and preserve one line per reported GPU.
set -euo pipefail

rocminfo_path=${SAGR_SIM_ROCMINFO:-}
if [[ -z $rocminfo_path ]]; then
  rocminfo_path=$(command -v rocminfo || true)
fi
if [[ -z $rocminfo_path || ! -x $rocminfo_path ]]; then
  printf 'sim-amdgpu-arch: rocminfo facade is unavailable\n' >&2
  exit 1
fi

exec "$rocminfo_path" |
  awk '
    /^[[:space:]]+Name:[[:space:]]+gfx[[:alnum:]_.-]+[[:space:]]*$/ {
      print $2
      found = 1
    }
    END {
      if (!found) {
        exit 1
      }
    }
  '
