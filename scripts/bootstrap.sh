#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root_dir"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }
git config core.hooksPath .githooks
chmod +x scripts/*.sh scripts/*.py .githooks/pre-commit .githooks/commit-msg
echo "configured local hooksPath=.githooks"
