#!/usr/bin/env bash
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
die() { echo "resume verification failed: $*" >&2; exit 1; }
for f in PLAN.md GOAL.md SOURCE_LOCK.json state/current.json; do
  [[ -f "$f" ]] || die "missing file: $f"
done
command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"
git diff --quiet || die "unstaged changes exist"
git diff --cached --quiet || die "staged changes exist"
for marker in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG; do
  marker_path=$(git rev-parse --git-path "$marker")
  [[ ! -e "$marker_path" ]] || die "git operation in progress: $marker"
done
hooks_path=$(git config --get core.hooksPath || true)
[[ "$hooks_path" == ".githooks" || "$hooks_path" == "$root_dir/.githooks" ]] || die "core.hooksPath must be .githooks (run scripts/bootstrap.sh)"

python3 - "$root_dir" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1])
def load(rel):
    try: return json.loads((root / rel).read_text(encoding="utf-8"))
    except Exception as exc: raise SystemExit(f"invalid JSON {rel}: {exc}")
def digest(rel): return hashlib.sha256((root / rel).read_bytes()).hexdigest()
current = load("state/current.json")
if current.get("schema") != "amdgpu-sim.current.v1": raise SystemExit("unsupported current.json schema")
cpid = current.get("checkpoint_id")
if not re.fullmatch(r"CP-[0-9]{4}", str(cpid or "")): raise SystemExit("checkpoint_id is malformed")
cp_rel = f"state/checkpoints/{cpid}.json"
if not (root / cp_rel).is_file(): raise SystemExit(f"missing checkpoint: {cp_rel}")
cp = load(cp_rel)
if cp.get("id") != cpid: raise SystemExit("checkpoint id mismatch")
for key in ("goal_id", "phase_id"):
    if cp.get(key) != current.get(key): raise SystemExit(f"{key} mismatch")
for rel, expected in {"PLAN.md": cp.get("plan_sha256"), "GOAL.md": cp.get("goal_sha256"), "SOURCE_LOCK.json": cp.get("source_lock_sha256")}.items():
    if not isinstance(expected, str) or expected == "TO-BE-FILLED" or digest(rel) != expected: raise SystemExit(f"checkpoint hash mismatch: {rel}")
action = cp.get("next_action")
if not isinstance(action, dict) or not action.get("id") or not action.get("cwd"): raise SystemExit("checkpoint has no next_action")
if current.get("next_action_id") != action["id"]: raise SystemExit("next action mismatch")
if current.get("state") not in {"ready", "paused", "blocked", "complete"}: raise SystemExit("invalid state")
lock = load("SOURCE_LOCK.json")
if lock.get("schema") != "amdgpu-sim.source-lock.v1": raise SystemExit("unsupported SOURCE_LOCK schema")
for source in lock.get("sources", []):
    rel = source.get("path")
    if not isinstance(rel, str) or rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts: raise SystemExit(f"unsafe source path: {rel!r}")
for path in root.rglob("*"):
    if path.is_symlink() and root not in path.resolve().parents: raise SystemExit(f"path escapes workspace: {path}")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip()
message = subprocess.check_output(["git", "log", "-1", "--format=%B"], text=True, cwd=root)
if f"Checkpoint-ID: {cpid}" not in message: raise SystemExit("HEAD lacks current Checkpoint-ID trailer")
if cp.get("root_commit") not in (None, "self-described-by-checkpoint-trailer", head): raise SystemExit("root commit mismatch")
for rel in ("state/evidence", "state/bitlessons"):
    if not (root / rel).is_dir(): raise SystemExit(f"missing directory: {rel}")
print(f"resume verification passed: {cpid} state={current['state']} next={action['id']} head={head}")
PY

if (( online )); then
  echo "online verification is not enabled by the bootstrap lock" >&2
  exit 2
fi
