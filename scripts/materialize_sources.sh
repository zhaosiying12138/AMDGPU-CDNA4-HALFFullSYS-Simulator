#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
resolution=${1:-artifacts/source-resolution.json}
python3 scripts/source_manager.py materialize \
  --resolution "$resolution" \
  --output artifacts/source-materialization.json
python3 scripts/source_manager.py absorb
