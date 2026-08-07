#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }
git config core.hooksPath .githooks
chmod +x scripts/*.sh .githooks/pre-commit
echo "configured local hooksPath=.githooks"
