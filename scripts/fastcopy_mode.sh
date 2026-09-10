#!/usr/bin/env bash

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AMDGPU-CDNA4-HALFFullSYS-Simulator contributors
# Full license text: LICENSE at the repository root.
# Source this file to select one copy path in the current shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source scripts/fastcopy_mode.sh fast|legacy" >&2
    exit 2
fi

case "${1:-}" in
    fast)
        export HSA_ENABLE_DTIF_FAST_COPY=1
        export SAGR_HSAKMT_MODEL_FAST_COPY=1
        ;;
    legacy)
        export HSA_ENABLE_DTIF_FAST_COPY=0
        export SAGR_HSAKMT_MODEL_FAST_COPY=0
        ;;
    *)
        echo "usage: source scripts/fastcopy_mode.sh fast|legacy" >&2
        return 2
        ;;
esac

