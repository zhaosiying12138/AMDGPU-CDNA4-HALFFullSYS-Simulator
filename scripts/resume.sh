#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

mode=""
online=0
for arg in "$@"; do
  case "$arg" in
    --verify) mode=verify ;;
    --online) online=1 ;;
    *) echo "usage: scripts/resume.sh --verify [--online]" >&2; exit 2 ;;
  esac
done
[[ "$mode" == verify ]] || { echo "usage: scripts/resume.sh --verify [--online]" >&2; exit 2; }

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
if python3 - "$root_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
current = json.loads((root / "state/current.json").read_text(encoding="utf-8"))
if current.get("state") != "archived":
    raise SystemExit(1)
required = (
    "PLAN.md",
    "GOAL.md",
    "ENGINEERING_CONSTRAINTS.md",
    "SOURCE_LOCK.json",
    "PROJECT_LANES.json",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"current contract files are missing: {', '.join(missing)}")
if current.get("next_action_id") is not None or current.get("resume_command") is not None:
    raise SystemExit("archived checkpoint pointer still contains an active command")
print("archived CP pointer verified; resume authority is GOAL.md and PLAN.md")
PY
then
  :
else
  python3 scripts/verify_workspace.py --root "$root_dir"
fi

if (( online )); then
  exec python3 scripts/source_manager.py verify-online
fi
