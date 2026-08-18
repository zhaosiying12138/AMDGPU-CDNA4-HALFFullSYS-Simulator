#!/usr/bin/env python3
"""Freeze a Hugging Face model repository into SOURCE_LOCK.json.

Adding a model to the acceptance ladder means pinning exactly which bytes the
run consumed. SOURCE_LOCK.json already carries that for Qwen3.5-0.8B, but the
entry was hand-assembled under a constraint that no longer applies: the
official API was unreachable at the time, so its provenance rests on mirror
metadata cross-checked against fixed-revision pages, and its
``official_raw_response_sha256`` is null.

With the official API reachable, a new entry can do better -- resolve the
revision from huggingface.co itself and hash the exact response bytes -- and
doing that by hand for a sixteen-file repository is both tedious and easy to
get subtly wrong. This tool does it reproducibly.

    tools/lock_model_source.py --repo-id Qwen/Qwen3.5-9B \\
        --source-id qwen3.5-9b --path models/Qwen3.5-9B \\
        --role acceptance-model --checkpoint CP-0063

The response is fetched once at a resolved immutable revision and every
subsequent field is derived from those bytes, so the entry and the recorded
hash cannot disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ENDPOINT = "https://huggingface.co"
LOCK_PATH = ROOT / "SOURCE_LOCK.json"
# Large enough for the biggest model index we lock, small enough that a
# redirected or hostile endpoint cannot exhaust memory.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class LockError(RuntimeError):
    pass


def fetch(url: str, timeout: float) -> bytes:
    """Fetch a URL, honouring the proxy environment, and return raw bytes.

    The raw bytes matter: they are what gets hashed into the lock, so the
    response must not be re-encoded or normalised on the way in.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "amdgpu-sim-lock/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise LockError(f"{url} returned HTTP {response.status}")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise LockError(f"{url} response exceeds {MAX_RESPONSE_BYTES} bytes")
    return payload


def resolve_revision(repo_id: str, endpoint: str, timeout: float) -> str:
    """Resolve the repository's current head to an immutable commit sha."""
    payload = fetch(f"{endpoint}/api/models/{repo_id}", timeout)
    revision = json.loads(payload).get("sha")
    if not isinstance(revision, str) or len(revision) != 40:
        raise LockError(f"{repo_id} did not resolve to a commit sha")
    return revision


def file_entries(document: dict) -> list[dict]:
    """Extract the per-file manifest, one entry per repository file.

    LFS files carry the payload hash the download can be verified against;
    plain files are identified by their git blob id. Both are recorded because
    a repository mixes the two, and verifying only one kind would leave the
    other unpinned.
    """
    entries = []
    for sibling in document.get("siblings", []):
        path = sibling.get("rfilename")
        if not isinstance(path, str) or not path:
            raise LockError("a repository file has no name")
        entry: dict = {"path": path, "size": sibling.get("size")}
        blob_id = sibling.get("blobId")
        if isinstance(blob_id, str):
            entry["blob_id"] = blob_id
        lfs = sibling.get("lfs")
        if isinstance(lfs, dict):
            entry["lfs"] = {
                "sha256": lfs.get("sha256"),
                "size": lfs.get("size"),
                "pointer_size": lfs.get("pointerSize"),
            }
        entries.append(entry)
    if not entries:
        raise LockError("repository manifest lists no files")
    return sorted(entries, key=lambda item: item["path"])


def normalized_manifest_sha256(entries: list[dict]) -> str:
    """Hash the manifest in the same normalised shape the 0.8B entry uses.

    Matching that normalisation -- sorted compact JSON over path, blob id, size
    and the LFS triple -- keeps the two entries comparable, so a reviewer can
    apply one procedure to both rather than reverse-engineering each.
    """
    shape = [
        {
            "path": entry["path"],
            "blobId": entry.get("blob_id"),
            "size": entry.get("size"),
            "lfs": entry.get("lfs"),
        }
        for entry in entries
    ]
    text = json.dumps(shape, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_entry(args: argparse.Namespace) -> dict:
    revision = args.revision or resolve_revision(
        args.repo_id, args.endpoint, args.timeout
    )
    # Pin the revision in the URL. Asking for "main" a second time can return a
    # different tree than the one just resolved, which would silently decouple
    # the recorded hash from the recorded file list.
    url = f"{args.endpoint}/api/models/{args.repo_id}/revision/{revision}?blobs=true"
    payload = fetch(url, args.timeout)
    document = json.loads(payload)
    if document.get("sha") != revision:
        raise LockError(
            f"revision drifted while fetching: asked {revision}, got {document.get('sha')}"
        )
    if document.get("private") or document.get("gated"):
        raise LockError(f"{args.repo_id} is private or gated; it cannot be locked")

    entries = file_entries(document)
    total = sum(entry.get("size") or 0 for entry in entries)
    tree = f"{args.endpoint}/{args.repo_id}/tree/{revision}"
    return {
        "id": args.source_id,
        "repo_id": args.repo_id,
        "path": args.path,
        "role": args.role,
        "upstream_url": f"{args.endpoint}/{args.repo_id}",
        "upstream_ref": "main",
        "materialization": "external-download",
        "materialized_size": total,
        "weights_in_git": False,
        "download_method": (
            "huggingface_hub.snapshot_download with resolved immutable revision"
        ),
        "download_status": "not-downloaded",
        "official_revision": revision,
        "official_revision_status": "frozen-from-official-api-at-resolved-revision",
        # The 0.8B entry could not record this: the official API was
        # unreachable and its provenance had to lean on mirror metadata.
        "official_raw_response_sha256": hashlib.sha256(payload).hexdigest(),
        "official_evidence": [tree, f"{args.endpoint}/{args.repo_id}/commit/{revision}"],
        "official_evidence_observed_at": args.observed_at,
        "model_manifest_provenance": (
            "official Hugging Face API response at a resolved immutable "
            "revision; raw response bytes hashed into this entry"
        ),
        "normalized_revision_manifest_sha256": normalized_manifest_sha256(entries),
        "normalization": (
            "sorted compact JSON over path, blobId, size, and LFS "
            "sha256/size/pointerSize"
        ),
        "frozen_by_checkpoint": args.checkpoint,
        "files": entries,
    }


def apply_entry(entry: dict, lock_path: Path) -> str:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen":
        raise LockError("SOURCE_LOCK.json is not frozen; refusing to edit it")
    sources = lock.get("sources", [])
    action = "added"
    for index, source in enumerate(sources):
        if source.get("id") == entry["id"]:
            sources[index] = entry
            action = "replaced"
            break
    else:
        sources.append(entry)
    sources.sort(key=lambda item: item.get("id", ""))
    lock["sources"] = sources
    temporary = lock_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, lock_path)
    return action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--path", required=True, help="repository-relative model path")
    parser.add_argument("--role", default="acceptance-model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", default=None, help="pin instead of resolving")
    parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.path.startswith("models/"):
        print("model path must live under models/", file=sys.stderr)
        return 2
    try:
        entry = build_entry(args)
        if args.dry_run:
            summary = dict(entry)
            summary["files"] = f"<{len(entry['files'])} files>"
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        action = apply_entry(entry, LOCK_PATH)
    except (LockError, OSError, json.JSONDecodeError) as error:
        print(f"lock failed: {error}", file=sys.stderr)
        return 1
    print(
        f"{action} {entry['id']} at {entry['official_revision']} "
        f"({len(entry['files'])} files, {entry['materialized_size']} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
