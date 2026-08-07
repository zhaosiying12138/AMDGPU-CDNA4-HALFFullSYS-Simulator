#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve and materialize immutable P0 source baselines.

This tool uses only the Python standard library and Git. Resolution writes an
ignored evidence candidate; it never edits SOURCE_LOCK.json. Materialization
clones the exact resolved commits, checks out a complete baseline worktree, and
creates immutable annotated tags. Root gitlink registration is a separate
explicit step so source review happens before the coordinator commit. Once the
reviewed gitlinks are in the root index, ``absorb`` moves every child repository
under the root repository's standard submodule administrative directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_policy import EvidencePolicyError, validate_evidence  # noqa: E402


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LOCK_ID_RE = re.compile(r"^SL-[0-9]{4}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
CP_RE = re.compile(r"^CP-[0-9]{4}$")
ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "SOURCE_LOCK.json"


class SourceError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
    check: bool = True,
    timeout: float | None = 90,
    extra_env: dict[str, str] | None = None,
    discard_stdout: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1024")
    env.setdefault("GIT_HTTP_LOW_SPEED_TIME", "60")
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.DEVNULL if discard_stdout else (subprocess.PIPE if capture else None),
            stderr=subprocess.PIPE if capture else None,
            input=input_text,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SourceError(f"command timed out after {timeout}s: {argv!r}") from exc
    if check and proc.returncode:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SourceError(f"command failed ({proc.returncode}): {argv!r}: {detail}")
    return proc


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(path, data)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_workspace_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise SourceError(f"unsafe workspace path: {value!r}")
    path = (ROOT / posix).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise SourceError(f"path escapes workspace: {value!r}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_bytes(repo: Path, revision: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"},
    )
    if proc.returncode:
        detail = proc.stderr.decode(errors="replace").strip()
        raise SourceError(
            f"cannot read accepted Git blob {revision}:{path}: {detail}"
        )
    return proc.stdout


def github_codeload_url(url: str, commit: str) -> str:
    match = re.fullmatch(
        r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?", url
    )
    if not match or not SHA_RE.match(commit):
        raise SourceError(f"cannot derive immutable GitHub archive URL: {url!r}")
    owner, repository = match.groups()
    return f"https://codeload.github.com/{owner}/{repository}/tar.gz/{commit}"


def download_file(url: str, destination: Path) -> str:
    request = Request(url, headers={"User-Agent": "amdgpu-sim-source-lock/1"})
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=90) as response, destination.open("wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"cannot download immutable archive {url}: {exc}") from exc
    return digest.hexdigest()


def safe_tar_member(member: tarfile.TarInfo, _destination: str) -> tarfile.TarInfo:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or member.isdev() or member.isfifo():
        raise SourceError(f"unsafe path or file type in source archive: {member.name!r}")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise SourceError(f"unsafe link in source archive: {member.linkname!r}")
    return member


def gitlink_paths(repo: Path, commit: str) -> list[str]:
    raw = run(
        ["git", "ls-tree", "-r", "-z", commit], cwd=repo, capture=True
    ).stdout or ""
    paths = []
    for entry in raw.split("\0"):
        if not entry:
            continue
        metadata, relative = entry.split("\t", 1)
        if metadata.split(" ", 1)[0] == "160000":
            paths.append(relative)
    return paths


def hydrate_from_github_archive(
    repo: Path, canonical_url: str, commit: str, expected_tree: str
) -> dict[str, str]:
    """Import exact baseline blobs from GitHub's fixed-revision codeload tar.

    Git still supplies every commit and tree object. The archive only avoids a
    second slow promisor fetch for baseline blobs. Rebuilding the index and
    matching the immutable tree makes the archive transport self-verifying.
    """

    archive_url = github_codeload_url(canonical_url, commit)
    with tempfile.TemporaryDirectory(prefix=".amdgpu-sim-archive-", dir=repo.parent) as temp:
        temporary = Path(temp)
        archive = temporary / "source.tar.gz"
        archive_sha256 = download_file(archive_url, archive)
        extracted = temporary / "extracted"
        extracted.mkdir()
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(extracted, filter=safe_tar_member)
        except (tarfile.TarError, SourceError, OSError) as exc:
            raise SourceError(f"cannot safely extract {archive_url}: {exc}") from exc
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise SourceError(f"archive does not contain one repository root: {archive_url}")
        archive_root = roots[0]
        for relative in gitlink_paths(repo, commit):
            path = archive_root / relative
            if path.exists() and not path.is_dir():
                raise SourceError(f"archive materialized non-directory at gitlink {relative}")
            path.mkdir(parents=True, exist_ok=True)
        temporary_index = temporary / "verification.index"
        index_env = {"GIT_INDEX_FILE": str(temporary_index)}
        run(["git", "read-tree", commit], cwd=repo, extra_env=index_env)
        run(
            ["git", f"--work-tree={archive_root}", "add", "-A"],
            cwd=repo,
            capture=False,
            timeout=None,
            extra_env=index_env,
        )
        actual_tree = run(
            ["git", "write-tree"], cwd=repo, extra_env=index_env
        ).stdout.strip()
        if actual_tree != expected_tree:
            summary = run(
                ["git", f"--work-tree={archive_root}", "diff", "--cached", "--stat", commit],
                cwd=repo,
                check=False,
                extra_env=index_env,
            ).stdout.strip()
            raise SourceError(
                f"codeload tree mismatch for {repo}: {actual_tree} != {expected_tree}; {summary}"
            )
    return {"archive_url": archive_url, "archive_sha256": archive_sha256}


def resolve_git(source: dict[str, Any]) -> dict[str, Any]:
    url = source["upstream_url"]
    transport = source.get("transport_url", url)
    output = run(["git", "ls-remote", "--symref", transport, "HEAD"]).stdout or ""
    branch = None
    commit = None
    for line in output.splitlines():
        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
            branch = line.split("\t", 1)[0].removeprefix("ref: refs/heads/")
        elif line.endswith("\tHEAD") and SHA_RE.match(line.split("\t", 1)[0]):
            commit = line.split("\t", 1)[0]
    if not branch or not commit:
        raise SourceError(f"could not resolve symbolic HEAD for {url}")
    expected_ref = source.get("upstream_ref")
    if expected_ref and expected_ref != branch:
        raise SourceError(f"default branch drift for {source['id']}: {expected_ref} -> {branch}")
    return {
        "id": source["id"],
        "kind": "git",
        "url": url,
        "transport_url": transport,
        "default_branch": branch,
        "commit": commit,
        "previous_observation": source.get("observed_head"),
        "observation_changed": source.get("observed_head") not in (None, commit),
        "command": ["git", "ls-remote", "--symref", transport, "HEAD"],
    }


def hf_repo_id(source: dict[str, Any]) -> str:
    prefix = "https://huggingface.co/"
    url = source["upstream_url"].rstrip("/")
    if not url.startswith(prefix):
        raise SourceError(f"unsupported Hugging Face URL: {url}")
    repo_id = url[len(prefix) :]
    if repo_id.count("/") != 1:
        raise SourceError(f"invalid Hugging Face repository id: {repo_id}")
    return repo_id


def resolve_huggingface(source: dict[str, Any]) -> dict[str, Any]:
    repo_id = hf_repo_id(source)
    endpoint = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    request = Request(endpoint, headers={"User-Agent": "amdgpu-sim-source-lock/1"})
    try:
        with urlopen(request, timeout=45) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SourceError(f"official Hugging Face API unavailable for {repo_id}: {exc}") from exc
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceError(f"invalid Hugging Face API JSON for {repo_id}: {exc}") from exc
    commit = info.get("sha")
    if not isinstance(commit, str) or not SHA_RE.match(commit):
        raise SourceError(f"official Hugging Face API returned no immutable SHA for {repo_id}")

    revision_endpoint = f"https://huggingface.co/api/models/{repo_id}/revision/{commit}?blobs=true"
    revision_request = Request(
        revision_endpoint, headers={"User-Agent": "amdgpu-sim-source-lock/1"}
    )
    try:
        with urlopen(revision_request, timeout=45) as response:
            revision_raw = response.read()
        revision_info = json.loads(revision_raw)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SourceError(f"official fixed-revision API unavailable for {repo_id}: {exc}") from exc
    if revision_info.get("sha") != commit:
        raise SourceError(f"fixed-revision API returned a different SHA for {repo_id}")

    git_result = resolve_git(source)
    if git_result["commit"] != commit:
        raise SourceError(
            f"Hugging Face API/Git disagreement for {repo_id}: {commit} != {git_result['commit']}"
        )
    files = []
    for item in revision_info.get("siblings", []):
        entry: dict[str, Any] = {
            "path": item.get("rfilename"),
            "blob_id": item.get("blobId"),
        }
        if item.get("size") is not None:
            entry["size"] = item["size"]
        lfs = item.get("lfs")
        if isinstance(lfs, dict):
            entry["lfs"] = {key: lfs[key] for key in ("sha256", "size", "pointerSize") if key in lfs}
        files.append(entry)
    files.sort(key=lambda item: str(item.get("path")))
    canonical_manifest = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return {
        "id": source["id"],
        "kind": "huggingface-model",
        "repo_id": repo_id,
        "url": source["upstream_url"],
        "default_branch": git_result["default_branch"],
        "commit": commit,
        "last_modified": revision_info.get("lastModified"),
        "license": (revision_info.get("cardData") or {}).get("license"),
        "pipeline_tag": revision_info.get("pipeline_tag"),
        "files": files,
        "api_endpoint": endpoint,
        "revision_api_endpoint": revision_endpoint,
        "revision_manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
    }


def frozen_model_manifest_sha256(source: dict[str, Any]) -> str:
    files = []
    for item in source.get("files", []):
        entry: dict[str, Any] = {
            "path": item["path"],
            "blobId": item["blob_id"],
            "size": item["size"],
        }
        if isinstance(item.get("lfs"), dict):
            lfs = item["lfs"]
            entry["lfs"] = {
                "sha256": lfs["sha256"],
                "size": lfs["size"],
                "pointerSize": lfs["pointer_size"],
            }
        files.append(entry)
    files.sort(key=lambda item: item["path"])
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_command(args: argparse.Namespace) -> int:
    lock = load_json(LOCK_PATH)
    sources = list(lock.get("sources", []))
    ordered: list[dict[str, Any] | None] = [None] * len(sources)
    errors: list[dict[str, str]] = []

    def resolve_index(index: int) -> tuple[int, dict[str, Any]]:
        source = sources[index]
        if source.get("role") == "acceptance-model" or str(
            source.get("materialization", "")
        ).startswith("external-download"):
            return index, resolve_huggingface(source)
        return index, resolve_git(source)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(sources) or 1)) as pool:
        pending = {pool.submit(resolve_index, index): index for index in range(len(sources))}
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            source = sources[index]
            try:
                resolved_index, result = future.result()
                ordered[resolved_index] = result
            except (KeyError, SourceError) as exc:
                errors.append({"id": str(source.get("id")), "error": str(exc)})
            except Exception as exc:  # Preserve an evidence record for unexpected failures.
                errors.append({"id": str(source.get("id")), "error": f"unexpected: {exc}"})
    results = [result for result in ordered if result is not None]
    payload = {
        "schema": "amdgpu-sim.source-resolution.v1",
        "resolved_at": utc_now(),
        "source_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "status": "resolved" if not errors else "failed",
        "results": results,
        "errors": errors,
    }
    output = safe_workspace_path(args.output)
    atomic_json(output, payload)
    print(output.relative_to(ROOT))
    if errors:
        for error in errors:
            print(f"{error['id']}: {error['error']}", file=sys.stderr)
        return 1
    return 0


def ensure_complete_baseline(repo: Path, commit: str) -> None:
    absent = missing_object_ids(repo, commit)
    if absent:
        raise SourceError(f"baseline tree has {len(absent)} missing objects in {repo}")
    run(
        ["git", "archive", "--format=tar", commit],
        cwd=repo,
        capture=False,
        timeout=None,
        extra_env={"GIT_NO_LAZY_FETCH": "1"},
        discard_stdout=True,
    )


def quarantine_temporary_packs(repo: Path, context: str) -> list[str]:
    """Preserve failed packs after this process's sole fetch has exited."""

    git_dir = Path(
        run(["git", "rev-parse", "--absolute-git-dir"], cwd=repo).stdout.strip()
    )
    temporary = sorted((git_dir / "objects" / "pack").glob("tmp_pack_*"))
    if not temporary:
        return []
    root = git_dir / "amdgpu-sim" / "quarantine"
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"{context}-", dir=root))
    moved = []
    for source in temporary:
        target = destination / source.name
        os.replace(source, target)
        moved.append(str(target))
    return moved


def resilient_history_fetch(
    repo: Path,
    transport: str,
    branch: str,
    commit: str,
    *,
    deepen_by: int,
    filter_spec: str = "blob:none",
) -> None:
    if deepen_by <= 0:
        raise SourceError("history deepen increment must be positive")
    if filter_spec not in {"blob:none", "tree:0"}:
        raise SourceError(f"unsupported history filter: {filter_spec!r}")
    actual_transport = run(["git", "remote", "get-url", "upstream"], cwd=repo).stdout.strip()
    if actual_transport != transport:
        raise SourceError(
            f"upstream transport changed in {repo}: {actual_transport} != {transport}"
        )
    present = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=False)
    shallow = run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo).stdout.strip()
    anchor = f"refs/amdgpu-sim/resolved/{branch}"
    anchor_present = run(["git", "rev-parse", "--verify", anchor], cwd=repo, check=False)
    if present.returncode == 0 and shallow == "false":
        if anchor_present.returncode:
            run(["git", "update-ref", anchor, commit], cwd=repo)
        elif (anchor_present.stdout or "").strip() != commit:
            raise SourceError(f"immutable resolution anchor changed: {anchor}")
        return
    if anchor_present.returncode:
        try:
            run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    "--no-auto-maintenance",
                    f"--filter={filter_spec}",
                    f"--depth={deepen_by}",
                    "upstream",
                    commit,
                ],
                cwd=repo,
                capture=False,
                timeout=None,
            )
        except SourceError:
            quarantine_temporary_packs(repo, "history-initial")
            raise
        fetched = run(["git", "rev-parse", "FETCH_HEAD"], cwd=repo).stdout.strip()
        if fetched != commit:
            raise SourceError(f"exact fetch returned {fetched}, expected {commit}")
        run(["git", "update-ref", anchor, commit], cwd=repo)
    anchored = run(["git", "rev-parse", anchor], cwd=repo).stdout.strip()
    if anchored != commit:
        raise SourceError(
            f"immutable resolution anchor changed: {anchor}={anchored}, expected {commit}"
        )
    while run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo).stdout.strip() == "true":
        before = int(run(["git", "rev-list", "--count", anchor], cwd=repo).stdout.strip())
        try:
            run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    "--no-auto-maintenance",
                    f"--filter={filter_spec}",
                    f"--deepen={deepen_by}",
                    "upstream",
                    commit,
                ],
                cwd=repo,
                capture=False,
                timeout=None,
            )
        except SourceError:
            quarantine_temporary_packs(repo, "history-deepen")
            raise
        after = int(run(["git", "rev-list", "--count", anchor], cwd=repo).stdout.strip())
        still_shallow = (
            run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo).stdout.strip()
            == "true"
        )
        if still_shallow and after <= before:
            raise SourceError(f"history deepening made no progress in {repo}")
    run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo)


def commit_tree_oid(repo: Path, commit: str) -> str:
    payload = run(
        ["git", "cat-file", "commit", commit],
        cwd=repo,
        extra_env={"GIT_NO_LAZY_FETCH": "1"},
    ).stdout or ""
    first_line = payload.splitlines()[0] if payload else ""
    match = re.fullmatch(r"tree ([0-9a-f]{40})", first_line)
    if not match:
        raise SourceError(f"cannot parse root tree from commit {commit} in {repo}")
    return match.group(1)


def missing_object_ids(repo: Path, commit: str) -> list[str]:
    root_tree = commit_tree_oid(repo, commit)
    output = run(
        ["git", "rev-list", "--objects", "--missing=print", root_tree],
        cwd=repo,
        extra_env={"GIT_NO_LAZY_FETCH": "1"},
    ).stdout or ""
    result = []
    for line in output.splitlines():
        if not line.startswith("?"):
            continue
        object_id = line[1:].split(None, 1)[0]
        if not SHA_RE.fullmatch(object_id):
            raise SourceError(f"malformed missing object ID in {repo}: {line!r}")
        result.append(object_id)
    return result


def require_local_objects(repo: Path, object_ids: list[str]) -> None:
    if not object_ids:
        return
    inspected = run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=repo,
        input_text="\n".join(object_ids) + "\n",
        extra_env={"GIT_NO_LAZY_FETCH": "1"},
    ).stdout or ""
    lines = inspected.splitlines()
    if len(lines) != len(object_ids):
        raise SourceError(f"object verification count mismatch in {repo}")
    missing = []
    for requested, line in zip(object_ids, lines, strict=True):
        fields = line.split()
        if len(fields) != 2 or fields[0] != requested or fields[1] == "missing":
            missing.append(requested)
    if missing:
        raise SourceError(
            f"exact-object fetch left {len(missing)} requested objects missing in {repo}"
        )


def hydrate_missing_objects(
    repo: Path,
    transport: str,
    commit: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise SourceError("blob fetch batch size must be positive")
    actual_transport = run(["git", "remote", "get-url", "upstream"], cwd=repo).stdout.strip()
    if actual_transport != transport:
        raise SourceError(
            f"upstream transport changed in {repo}: {actual_transport} != {transport}"
        )
    requested: set[str] = set()
    waves = 0
    while True:
        missing = sorted(set(missing_object_ids(repo, commit)))
        if not missing:
            break
        waves += 1
        if waves > 512:
            raise SourceError(f"baseline hydration exceeded 512 dependency waves in {repo}")
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            try:
                run(
                    [
                        "git",
                        "fetch",
                        "--no-tags",
                        "--no-auto-maintenance",
                        "--no-write-fetch-head",
                        "--refetch",
                        "--stdin",
                        "upstream",
                    ],
                    cwd=repo,
                    timeout=None,
                    discard_stdout=True,
                    input_text="\n".join(batch) + "\n",
                )
                require_local_objects(repo, batch)
            except SourceError:
                quarantine_temporary_packs(repo, "baseline-objects")
                raise
            requested.update(batch)
    return {
        "hydration_method": "resumable-git-object-batches",
        "hydrated_object_count": len(requested),
        "hydration_batch_size": batch_size,
        "hydration_dependency_waves": waves,
    }


def configure_child_safety(repo: Path) -> str:
    hooks = os.path.relpath(ROOT / ".githooks", repo)
    run(["git", "remote", "set-url", "--push", "upstream", "no_push"], cwd=repo)
    run(["git", "config", "core.hooksPath", hooks], cwd=repo)
    return hooks


def install_frozen_tag(repo: Path, tag: str, payload: str, expected_object: str) -> None:
    actual = run(
        ["git", "hash-object", "-t", "tag", "-w", "--stdin"],
        cwd=repo,
        input_text=payload,
    ).stdout.strip()
    if actual != expected_object:
        raise SourceError(f"frozen tag payload hash mismatch for {tag}: {actual}")
    existing = run(["git", "rev-parse", f"refs/tags/{tag}"], cwd=repo, check=False)
    if existing.returncode == 0 and (existing.stdout or "").strip() != actual:
        raise SourceError(f"refusing to replace a different existing tag: {tag}")
    run(["git", "update-ref", f"refs/tags/{tag}", actual], cwd=repo)


def materialize_one(
    source: dict[str, Any],
    resolved: dict[str, Any],
    *,
    archive_hydration: bool,
    history_chunk: int,
    blob_batch_size: int,
) -> dict[str, Any]:
    path = safe_workspace_path(source["path"])
    url = source["upstream_url"]
    transport = source.get("transport_url", url)
    commit = resolved["commit"]
    if not SHA_RE.match(commit):
        raise SourceError(f"invalid resolved commit for {source['id']}: {commit}")
    filter_spec = (source.get("history") or {}).get(
        "partial_clone_filter", source.get("materialization_filter", "blob:none")
    )
    if filter_spec not in {"blob:none", "tree:0"}:
        raise SourceError(f"unsupported materialization filter for {source['id']}: {filter_spec!r}")
    if path.exists() and any(path.iterdir()):
        if not (path / ".git").exists():
            raise SourceError(f"non-empty non-repository source path: {path}")
        existing_url = run(["git", "remote", "get-url", "upstream"], cwd=path).stdout.strip()
        if existing_url != transport:
            raise SourceError(f"upstream URL mismatch in {path}: {existing_url} != {transport}")
        existing_filter = run(
            ["git", "config", "--get", "remote.upstream.partialclonefilter"],
            cwd=path,
            check=False,
        ).stdout.strip()
        if existing_filter != filter_spec:
            raise SourceError(
                f"partial clone filter mismatch in {path}: {existing_filter!r} != {filter_spec!r}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=True)
        run(["git", "init", "--initial-branch=amdgpu-sim/materialize", str(path)])
        run(["git", "remote", "add", "upstream", transport], cwd=path)
        run(["git", "config", "remote.upstream.promisor", "true"], cwd=path)
        run(["git", "config", "remote.upstream.partialclonefilter", filter_spec], cwd=path)
    resilient_history_fetch(
        path,
        transport,
        resolved["default_branch"],
        commit,
        deepen_by=history_chunk,
        filter_spec=filter_spec,
    )
    branch = source["work_branch"]
    tree = run(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=path).stdout.strip()
    hydration: dict[str, Any] = {}
    if archive_hydration:
        hydration = hydrate_from_github_archive(path, url, commit, tree)
    else:
        hydration = hydrate_missing_objects(
            path, transport, commit, batch_size=blob_batch_size
        )
    run(["git", "checkout", "-B", branch, commit], cwd=path, capture=False, timeout=None)
    tag = f"upstream-baseline/{source['id']}/{commit}"
    tag_target = run(["git", "rev-list", "-n", "1", tag], cwd=path, check=False)
    frozen_payload = source.get("baseline_tag_payload")
    frozen_object = source.get("baseline_tag_object")
    if isinstance(frozen_payload, str) and isinstance(frozen_object, str):
        install_frozen_tag(path, tag, frozen_payload, frozen_object)
    elif tag_target.returncode:
        message = (
            f"Pristine upstream baseline for {source['id']}\n\n"
            f"Upstream-URL: {url}\nUpstream-Ref: {resolved['default_branch']}\n"
            f"Baseline-Commit: {commit}\nBaseline-Tree: {tree}\n"
            f"Resolved-At: {resolved.get('resolved_at', 'resolution-manifest')}"
        )
        run(["git", "tag", "-a", tag, commit, "-m", message], cwd=path)
    elif (tag_target.stdout or "").strip() != commit:
        raise SourceError(f"baseline tag collision in {path}: {tag}")
    ensure_complete_baseline(path, commit)
    compatibility_results = []
    for compatibility in source.get("compatibility_revisions", []):
        compatibility_commit = compatibility["commit"]
        present = run(
            ["git", "cat-file", "-e", f"{compatibility_commit}^{{commit}}"],
            cwd=path,
            check=False,
        )
        if present.returncode:
            run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    "--no-auto-maintenance",
                    f"--filter={filter_spec}",
                    "upstream",
                    compatibility_commit,
                ],
                cwd=path,
                capture=False,
                timeout=None,
            )
        compatibility_tree = run(
            ["git", "rev-parse", f"{compatibility_commit}^{{tree}}"], cwd=path
        ).stdout.strip()
        hydrate_missing_objects(
            path, transport, compatibility_commit, batch_size=blob_batch_size
        )
        compatibility_tag = compatibility["tag"]
        frozen_compatibility_payload = compatibility.get("tag_payload")
        frozen_compatibility_object = compatibility.get("tag_object")
        if isinstance(frozen_compatibility_payload, str) and isinstance(
            frozen_compatibility_object, str
        ):
            install_frozen_tag(
                path,
                compatibility_tag,
                frozen_compatibility_payload,
                frozen_compatibility_object,
            )
        else:
            existing = run(
                ["git", "rev-list", "-n", "1", compatibility_tag],
                cwd=path,
                check=False,
            )
            if existing.returncode:
                message = (
                    f"Triton compatibility baseline for {compatibility.get('required_by_source', 'consumer')} "
                    f"{compatibility.get('required_by_commit', 'unknown')}\n\n"
                    f"Upstream-URL: {url}\n"
                    f"Upstream-Ref: {compatibility.get('purpose', 'compatibility revision')}\n"
                    f"Compatibility-Commit: {compatibility_commit}\n"
                    f"Compatibility-Tree: {compatibility_tree}\n"
                    f"Resolved-At: {resolved.get('resolved_at', 'resolution-manifest')}"
                )
                run(
                    [
                        "git",
                        "tag",
                        "-a",
                        compatibility_tag,
                        compatibility_commit,
                        "-m",
                        message,
                    ],
                    cwd=path,
                )
            elif (existing.stdout or "").strip() != compatibility_commit:
                raise SourceError(f"compatibility tag collision in {path}: {compatibility_tag}")
        ensure_complete_baseline(path, compatibility_commit)
        compatibility_results.append(
            {
                "commit": compatibility_commit,
                "tree": compatibility_tree,
                "tag": compatibility_tag,
                "tag_object": run(
                    ["git", "rev-parse", compatibility_tag], cwd=path
                ).stdout.strip(),
                "offline_tree_verified": True,
            }
        )
    hooks_path = configure_child_safety(path)
    status = run(["git", "status", "--porcelain"], cwd=path).stdout or ""
    if status:
        raise SourceError(f"materialized repository is dirty: {path}")
    head = run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()
    tag_object = run(["git", "rev-parse", tag], cwd=path).stdout.strip()
    tag_payload = run(["git", "cat-file", "tag", tag], cwd=path).stdout or ""
    commit_time = run(["git", "show", "-s", "--format=%cI", "HEAD"], cwd=path).stdout.strip()
    return {
        "id": source["id"],
        "path": source["path"],
        "upstream_url": url,
        "transport_url": transport,
        "default_branch": resolved["default_branch"],
        "baseline_commit": head,
        "baseline_tree": tree,
        "baseline_tag": tag,
        "baseline_tag_object": tag_object,
        "baseline_tag_payload": tag_payload,
        "baseline_tag_payload_sha256": hashlib.sha256(tag_payload.encode()).hexdigest(),
        "work_branch": branch,
        "commit_time": commit_time,
        "complete_baseline_tree": True,
        "hooks_path": hooks_path,
        "upstream_push_url": "no_push",
        "compatibility_revisions": compatibility_results,
        **hydration,
    }


def materialization_resolution(
    lock: dict[str, Any], resolution_path: Path, *, lock_path: Path = LOCK_PATH
) -> tuple[dict[str, dict[str, Any]], str, str]:
    if lock.get("status") == "frozen":
        by_id = {
            source["id"]: {
                "id": source["id"],
                "kind": "git",
                "commit": source["baseline_commit"],
                "default_branch": source["upstream_ref"],
                "resolved_at": lock["frozen_at"],
            }
            for source in lock.get("sources", [])
            if source.get("materialization") == "gitlink"
        }
        resolution_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        resolution_source = "frozen-source-lock"
    else:
        resolution = load_json(resolution_path)
        if resolution.get("status") != "resolved":
            raise SourceError("refusing to materialize an incomplete source resolution")
        by_id = {item["id"]: item for item in resolution["results"]}
        resolution_sha256 = hashlib.sha256(resolution_path.read_bytes()).hexdigest()
        resolution_source = "online-resolution-manifest"
    return by_id, resolution_sha256, resolution_source


def materialize_command(args: argparse.Namespace) -> int:
    lock = load_json(LOCK_PATH)
    resolution_path = safe_workspace_path(args.resolution)
    by_id, resolution_sha256, resolution_source = materialization_resolution(
        lock, resolution_path
    )
    entries = []
    for source in lock.get("sources", []):
        if source.get("path", "").startswith("models/"):
            continue
        resolved = by_id.get(source["id"])
        if not resolved or resolved.get("kind") != "git":
            raise SourceError(f"missing git resolution for {source['id']}")
        entries.append(
            materialize_one(
                source,
                resolved,
                archive_hydration=args.archive_hydration,
                history_chunk=args.history_chunk,
                blob_batch_size=args.blob_batch_size,
            )
        )
    payload = {
        "schema": "amdgpu-sim.source-materialization.v1",
        "materialized_at": utc_now(),
        "resolution_sha256": resolution_sha256,
        "resolution_source": resolution_source,
        "repositories": entries,
    }
    output = safe_workspace_path(args.output)
    atomic_json(output, payload)
    print(output.relative_to(ROOT))
    return 0


def git_source_revision(lock: dict[str, Any], source: dict[str, Any]) -> str:
    """Return the immutable revision an absorbed root gitlink must record."""

    source_id = source.get("id")
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise SourceError(f"source has an unsafe submodule name: {source_id!r}")
    if lock.get("status") == "frozen":
        baseline = source.get("baseline_commit")
        work_head = source.get("work_head")
        if not SHA_RE.fullmatch(baseline or "") or not SHA_RE.fullmatch(work_head or ""):
            raise SourceError(
                f"frozen source lacks baseline/work head identity: {source_id}"
            )
        if baseline != work_head:
            raise SourceError(
                f"frozen baseline/work head mismatch for {source_id}: "
                f"{baseline} != {work_head}"
            )
        return work_head
    observed = source.get("observed_head")
    if not SHA_RE.fullmatch(observed or ""):
        raise SourceError(f"pre-freeze source lacks an observed head: {source_id}")
    return observed


def absorption_sources(lock: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Select Git sources from a frozen or pre-freeze SOURCE_LOCK."""

    if lock.get("schema") != "amdgpu-sim.source-lock.v1":
        raise SourceError("cannot absorb sources from an invalid SOURCE_LOCK schema")
    if lock.get("status") not in {"frozen", "observed-not-fetched"}:
        raise SourceError(f"cannot absorb sources from lock status {lock.get('status')!r}")
    sources = lock.get("sources")
    if not isinstance(sources, list):
        raise SourceError("SOURCE_LOCK sources must be a list")
    selected: list[tuple[dict[str, Any], str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise SourceError("SOURCE_LOCK source entries must be objects")
        materialization = source.get("materialization")
        if materialization not in {"pending", "gitlink"}:
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise SourceError(f"source has an unsafe submodule name: {source_id!r}")
        path_text = source.get("path")
        if not isinstance(path_text, str):
            raise SourceError(f"source has no path: {source_id}")
        path = PurePosixPath(path_text)
        if (
            path.is_absolute()
            or path_text in {"", "."}
            or path.as_posix() != path_text
            or ".." in path.parts
            or any(ord(character) < 32 or ord(character) == 127 for character in path_text)
        ):
            raise SourceError(f"source has an unsafe submodule path: {source_id}:{path_text!r}")
        if source_id in seen_ids or path_text in seen_paths:
            raise SourceError(f"duplicate source id or path in SOURCE_LOCK: {source_id}")
        seen_ids.add(source_id)
        seen_paths.add(path_text)
        selected.append((source, git_source_revision(lock, source)))
    if not selected:
        raise SourceError("SOURCE_LOCK contains no Git sources to absorb")
    return selected


def root_common_git_dir() -> Path:
    value = run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT).stdout.strip()
    common = Path(value)
    if not common.is_absolute():
        common = ROOT / common
    return common.resolve()


def gitmodules_path_entries() -> dict[str, list[str]]:
    modules = ROOT / ".gitmodules"
    if not modules.is_file() or modules.is_symlink():
        raise SourceError("root .gitmodules is missing or is not a regular file")
    tracked = run(
        ["git", "ls-files", "--stage", "--", ".gitmodules"], cwd=ROOT
    ).stdout or ""
    records = [line for line in tracked.splitlines() if line]
    if len(records) != 1 or not records[0].startswith(("100644 ", "100755 ")):
        raise SourceError("root .gitmodules is not a regular file in the index")
    if run(["git", "diff", "--quiet", "--", ".gitmodules"], cwd=ROOT, check=False).returncode:
        raise SourceError("root .gitmodules differs from its indexed content")
    configured = run(
        [
            "git",
            "config",
            "-z",
            "-f",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        cwd=ROOT,
        check=False,
    )
    if configured.returncode not in (0, 1):
        raise SourceError("cannot parse root .gitmodules")
    entries: dict[str, list[str]] = {}
    for record in (configured.stdout or "").split("\0"):
        if not record:
            continue
        try:
            key, value = record.split("\n", 1)
        except ValueError as exc:
            raise SourceError("malformed path entry in root .gitmodules") from exc
        entries.setdefault(key, []).append(value)
    return entries


def indexed_gitlink(path: str) -> tuple[str, str, str]:
    output = run(
        ["git", "ls-files", "--stage", "-z", "--", path],
        cwd=ROOT,
        extra_env={"GIT_LITERAL_PATHSPECS": "1"},
    ).stdout or ""
    records = [record for record in output.split("\0") if record]
    if len(records) != 1:
        raise SourceError(f"root index does not contain exactly one entry for {path}")
    try:
        metadata, indexed_path = records[0].split("\t", 1)
        mode, object_id, stage = metadata.split(" ")
    except ValueError as exc:
        raise SourceError(f"malformed root index entry for {path}") from exc
    if indexed_path != path:
        raise SourceError(f"root index returned a non-exact path for {path}: {indexed_path!r}")
    return mode, object_id, stage


def verify_absorbed_source(source: dict[str, Any]) -> dict[str, str]:
    """Verify Git's absorbed layout without accepting linked worktrees."""

    source_id = source.get("id")
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise SourceError(f"source has an unsafe submodule name: {source_id!r}")
    repo = safe_workspace_path(source["path"])
    gitfile = repo / ".git"
    common = root_common_git_dir()
    modules_dir = common / "modules"
    expected_admin = modules_dir / source_id
    if modules_dir.is_symlink() or expected_admin.is_symlink():
        raise SourceError(f"absorbed administration path is a symlink: {source_id}")
    expected_admin = expected_admin.resolve()
    if common not in expected_admin.parents:
        raise SourceError(f"absorbed administration path escapes root Git dir: {source_id}")
    expected_gitfile = f"gitdir: {os.path.relpath(expected_admin, repo)}\n"
    if not gitfile.is_file() or gitfile.is_symlink():
        raise SourceError(f"source still has an embedded or invalid .git entry: {source_id}")
    if gitfile.read_text(encoding="utf-8") != expected_gitfile:
        raise SourceError(f"source has a non-canonical absorbed gitfile: {source_id}")
    absolute_git_dir = Path(
        run(["git", "rev-parse", "--absolute-git-dir"], cwd=repo).stdout.strip()
    ).resolve()
    child_common_text = run(
        ["git", "rev-parse", "--git-common-dir"], cwd=repo
    ).stdout.strip()
    child_common = Path(child_common_text)
    if not child_common.is_absolute():
        child_common = repo / child_common
    child_common = child_common.resolve()
    if absolute_git_dir != expected_admin or child_common != expected_admin:
        raise SourceError(
            f"source Git dir/common dir is not the canonical absorbed directory: {source_id}"
        )
    return {
        "id": source_id,
        "path": source["path"],
        "administrative_git_dir": f"modules/{source_id}",
        "gitfile": expected_gitfile.rstrip("\n"),
    }


def absorb_sources(lock: dict[str, Any]) -> list[dict[str, str]]:
    """Validate and absorb every locked Git source into the root repository."""

    selected = absorption_sources(lock)
    module_entries = gitmodules_path_entries()
    paths: list[str] = []
    for source, expected_commit in selected:
        source_id = source["id"]
        path_text = source["path"]
        canonical_key = f"submodule.{source_id}.path"
        if module_entries.get(canonical_key) != [path_text]:
            raise SourceError(
                f"root .gitmodules lacks the canonical name/path for {source_id}"
            )
        aliases = [
            key
            for key, values in module_entries.items()
            if path_text in values and key != canonical_key
        ]
        if aliases:
            raise SourceError(
                f"root .gitmodules maps {path_text} through non-canonical names: {aliases}"
            )
        mode, object_id, stage = indexed_gitlink(path_text)
        if (mode, object_id, stage) != ("160000", expected_commit, "0"):
            raise SourceError(
                f"root index gitlink mismatch for {source_id}: "
                f"{mode} {object_id} stage={stage}; expected 160000 {expected_commit} stage=0"
            )
        repo = safe_workspace_path(path_text)
        git_entry = repo / ".git"
        if not repo.is_dir() or not git_entry.exists():
            raise SourceError(f"source repository is not materialized: {source_id}")
        if git_entry.is_symlink() or not (git_entry.is_dir() or git_entry.is_file()):
            raise SourceError(f"source has an invalid .git entry: {source_id}")
        if git_entry.is_file():
            # An idempotent rerun is allowed only for the exact layout produced
            # by the root repository's standard absorbgitdirs operation.
            verify_absorbed_source(source)
        top = Path(
            run(["git", "rev-parse", "--show-toplevel"], cwd=repo).stdout.strip()
        ).resolve()
        if top != repo:
            raise SourceError(f"source path is not its Git worktree root: {source_id}")
        head = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
        if head != expected_commit:
            raise SourceError(
                f"source HEAD mismatch for {source_id}: {head} != {expected_commit}"
            )
        if run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo
        ).stdout:
            raise SourceError(f"refusing to absorb dirty source: {source_id}")
        paths.append(path_text)
    run(
        ["git", "submodule", "absorbgitdirs", "--", *paths],
        cwd=ROOT,
        capture=False,
        timeout=None,
    )
    return [verify_absorbed_source(source) for source, _commit in selected]


def absorb_command(_args: argparse.Namespace) -> int:
    lock = load_json(LOCK_PATH)
    repositories = absorb_sources(lock)
    payload = {
        "schema": "amdgpu-sim.source-absorption.v1",
        "absorbed_at": utc_now(),
        "source_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "repositories": repositories,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def tag_identity(repo: Path, tag: str, commit: str) -> dict[str, str]:
    if run(["git", "cat-file", "-t", tag], cwd=repo).stdout.strip() != "tag":
        raise SourceError(f"baseline marker is not an annotated tag: {repo}:{tag}")
    target = run(["git", "rev-list", "-n", "1", tag], cwd=repo).stdout.strip()
    if target != commit:
        raise SourceError(f"tag target mismatch: {repo}:{tag}: {target} != {commit}")
    payload = run(["git", "cat-file", "tag", tag], cwd=repo).stdout or ""
    return {
        "tag_object": run(["git", "rev-parse", tag], cwd=repo).stdout.strip(),
        "tag_payload": payload,
        "tag_payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def require_tag_lines(payload: str, required: list[str], label: str) -> None:
    lines = set(payload.splitlines())
    missing = [line for line in required if line not in lines]
    if missing:
        raise SourceError(f"annotated tag payload is missing {missing!r}: {label}")


def declared_submodule_paths(repo: Path) -> list[str]:
    modules = repo / ".gitmodules"
    if not modules.is_file():
        return []
    result = run(
        ["git", "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
        cwd=repo,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise SourceError(f"cannot parse nested submodules in {repo}")
    paths = []
    for line in (result.stdout or "").splitlines():
        if line.strip():
            paths.append(line.split(None, 1)[1])
    if len(paths) != len(set(paths)):
        raise SourceError(f"duplicate nested submodule path in {repo}")
    return sorted(paths)


def skip_worktree_count(repo: Path) -> int:
    output = run(["git", "ls-files", "-v", "-z"], cwd=repo).stdout or ""
    return sum(
        1
        for entry in output.split("\0")
        if entry and entry[0] in {"S", "s"}
    )


def freeze_git_source(source: dict[str, Any]) -> dict[str, Any]:
    repo = safe_workspace_path(source["path"])
    if not repo.is_dir():
        raise SourceError(f"source is not materialized: {source['path']}")
    if run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo).stdout.strip() != "false":
        raise SourceError(f"refusing to freeze shallow source: {source['id']}")
    if (run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo).stdout or ""):
        raise SourceError(f"refusing to freeze dirty source: {source['id']}")
    sparse = run(
        ["git", "config", "--bool", "core.sparseCheckout"], cwd=repo, check=False
    )
    if sparse.returncode not in (0, 1) or (sparse.stdout or "").strip() == "true":
        raise SourceError(f"refusing to freeze sparse source: {source['id']}")
    skipped = skip_worktree_count(repo)
    if skipped:
        raise SourceError(
            f"refusing to freeze source with {skipped} skip-worktree entries: {source['id']}"
        )
    for selected in source.get("selected_paths", []):
        selected_path = (repo / PurePosixPath(selected)).resolve()
        if repo not in selected_path.parents or not selected_path.exists():
            raise SourceError(f"selected source path is missing or unsafe: {source['id']}:{selected}")
    license_files = source.get("license_files")
    if not isinstance(license_files, list) or not license_files:
        raise SourceError(f"source has no frozen license inventory: {source['id']}")
    for license_file in license_files:
        relative_license = PurePosixPath(license_file["path"])
        if relative_license.is_absolute() or ".." in relative_license.parts:
            raise SourceError(f"unsafe license path: {source['id']}:{relative_license}")
        license_path = repo / relative_license
        if not license_path.is_file() or file_sha256(license_path) != license_file["sha256"]:
            raise SourceError(f"license file identity mismatch: {source['id']}:{license_file['path']}")
    branch = run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo).stdout.strip()
    if branch != source.get("work_branch"):
        raise SourceError(f"unexpected work branch for {source['id']}: {branch}")
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    if commit != source.get("observed_head"):
        raise SourceError(
            f"materialized commit differs from online observation for {source['id']}: {commit}"
        )
    tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo).stdout.strip()
    tracking_ref = source.get(
        "upstream_tracking_ref", f"refs/amdgpu-sim/resolved/{source['upstream_ref']}"
    )
    tracked_commit = run(
        ["git", "rev-parse", "--verify", tracking_ref], cwd=repo, check=False
    )
    if tracked_commit.returncode or (tracked_commit.stdout or "").strip() != commit:
        raise SourceError(f"immutable upstream tracking ref mismatch for {source['id']}")
    tag = f"upstream-baseline/{source['id']}/{commit}"
    marker = tag_identity(repo, tag, commit)
    require_tag_lines(
        marker["tag_payload"],
        [
            f"Upstream-URL: {source['upstream_url']}",
            f"Upstream-Ref: {source['upstream_ref']}",
            f"Baseline-Commit: {commit}",
            f"Baseline-Tree: {tree}",
        ],
        source["id"],
    )
    ensure_complete_baseline(repo, commit)
    hooks = os.path.relpath(ROOT / ".githooks", repo)
    if run(["git", "config", "--get", "core.hooksPath"], cwd=repo).stdout.strip() != hooks:
        raise SourceError(f"child hooksPath is not portable for {source['id']}")
    fetch_url = run(["git", "remote", "get-url", "upstream"], cwd=repo).stdout.strip()
    push_url = run(
        ["git", "remote", "get-url", "--push", "upstream"], cwd=repo
    ).stdout.strip()
    remotes = (run(["git", "remote"], cwd=repo).stdout or "").splitlines()
    if remotes != ["upstream"]:
        raise SourceError(f"unexpected remote set for {source['id']}: {remotes}")
    if fetch_url != source.get("transport_url") or push_url != "no_push":
        raise SourceError(f"unsafe or unexpected upstream remote for {source['id']}")
    child_git_dir = Path(
        run(["git", "rev-parse", "--absolute-git-dir"], cwd=repo).stdout.strip()
    )
    child_common_dir_text = run(
        ["git", "rev-parse", "--git-common-dir"], cwd=repo
    ).stdout.strip()
    child_common_dir = Path(child_common_dir_text)
    if not child_common_dir.is_absolute():
        child_common_dir = (repo / child_common_dir).resolve()
    else:
        child_common_dir = child_common_dir.resolve()
    root_common_dir_text = run(
        ["git", "rev-parse", "--git-common-dir"], cwd=ROOT
    ).stdout.strip()
    root_common_dir = Path(root_common_dir_text)
    if not root_common_dir.is_absolute():
        root_common_dir = (ROOT / root_common_dir).resolve()
    else:
        root_common_dir = root_common_dir.resolve()
    source_id = source.get("id")
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise SourceError(f"source has an unsafe submodule name: {source_id!r}")
    admin_relative = (PurePosixPath("modules") / source_id).as_posix()
    expected_child_common_dir = (root_common_dir / admin_relative).resolve()
    expected_gitfile = f"gitdir: {os.path.relpath(expected_child_common_dir, repo)}\n"
    if (
        not (repo / ".git").is_file()
        or child_common_dir != expected_child_common_dir
        or child_git_dir.resolve() != child_common_dir
        or (repo / ".git").read_text(encoding="utf-8") != expected_gitfile
    ):
        raise SourceError(f"source is not an absorbed root submodule: {source['id']}")
    if (child_git_dir / "info" / "grafts").is_file():
        raise SourceError(f"source history uses a grafts file: {source['id']}")
    if (run(["git", "replace", "-l"], cwd=repo).stdout or "").strip():
        raise SourceError(f"source history uses replacement objects: {source['id']}")
    alternates = child_git_dir / "objects" / "info" / "alternates"
    if alternates.is_file() and alternates.read_text(encoding="utf-8").strip():
        raise SourceError(f"source still depends on alternates: {source['id']}")
    object_counts = run(["git", "count-objects", "-v"], cwd=repo).stdout or ""
    counts = dict(
        line.split(": ", 1) for line in object_counts.splitlines() if ": " in line
    )
    if counts.get("garbage", "0") != "0":
        raise SourceError(f"source object database has garbage: {source['id']}")
    run(
        ["git", "fsck", "--connectivity-only"],
        cwd=repo,
        timeout=None,
        extra_env={"GIT_NO_LAZY_FETCH": "1"},
    )
    submodule_status = run(
        ["git", "submodule", "status", "--recursive"], cwd=repo, check=False
    )
    if submodule_status.returncode:
        raise SourceError(f"cannot inspect nested submodules for {source['id']}")
    initialized = [
        line for line in (submodule_status.stdout or "").splitlines() if line and not line.startswith("-")
    ]
    if initialized:
        raise SourceError(f"nested submodules unexpectedly materialized in {source['id']}")
    declared_submodules = declared_submodule_paths(repo)
    actual_gitlinks = sorted(gitlink_paths(repo, commit))
    partial_filter = (
        run(
            ["git", "config", "--get", "remote.upstream.partialclonefilter"],
            cwd=repo,
            check=False,
        ).stdout
        or "none"
    ).strip()
    if partial_filter not in {"none", "blob:none", "tree:0"}:
        raise SourceError(
            f"unsupported partial clone filter for {source['id']}: {partial_filter!r}"
        )
    promisor = (
        run(
            ["git", "config", "--bool", "--get", "remote.upstream.promisor"],
            cwd=repo,
            check=False,
        ).stdout
        or "false"
    ).strip()
    if promisor not in {"true", "false"} or (partial_filter != "none") != (
        promisor == "true"
    ):
        raise SourceError(f"promisor/filter configuration mismatch for {source['id']}")
    reachable_commit_count = int(
        run(
            ["git", "rev-list", "--count", commit],
            cwd=repo,
            extra_env={"GIT_NO_LAZY_FETCH": "1"},
        ).stdout.strip()
    )
    frozen = {
        **source,
        "materialization": "gitlink",
        "baseline_commit": commit,
        "baseline_tree": tree,
        "baseline_tag": tag,
        "baseline_tag_object": marker["tag_object"],
        "baseline_tag_payload": marker["tag_payload"],
        "baseline_tag_payload_sha256": marker["tag_payload_sha256"],
        "work_head": commit,
        "work_tree": tree,
        "commit_time": run(["git", "show", "-s", "--format=%cI", commit], cwd=repo).stdout.strip(),
        "expected_remotes": {"upstream": fetch_url},
        "expected_push_urls": {"upstream": push_url},
        "administrative_git_dir": admin_relative,
        "upstream_tracking_ref": tracking_ref,
        "hooks_path": hooks,
        "history": {
            "shallow": False,
            "commit_ancestry_scope": "all commits reachable from the locked baseline head",
            "commit_ancestry_offline_traversable": True,
            "reachable_commit_count": reachable_commit_count,
            "partial_clone_filter": partial_filter,
            "promisor_remote": promisor == "true",
            "historical_tree_blob_scope": (
                "all-local" if partial_filter == "none" else "promisor-filtered"
            ),
            "locked_baseline_tree_fully_hydrated": True,
            "fresh_offline_clone_bundle_available": False,
        },
        "offline_tree_verified": True,
        "worktree_verification": {
            "full_checkout": True,
            "sparse_checkout": False,
            "skip_worktree_entries": 0,
        },
        "nested_submodules": {
            "declared_count": len(declared_submodules),
            "declared_paths": declared_submodules,
            "gitlink_count": len(actual_gitlinks),
            "gitlink_paths": actual_gitlinks,
            "stale_declarations": sorted(set(declared_submodules) - set(actual_gitlinks)),
            "undeclared_gitlinks": sorted(set(actual_gitlinks) - set(declared_submodules)),
            "materialized": False,
            "offline_build_closure": False,
        },
        "baseline_verification": {
            "tree_object_verified": True,
            "offline_archive_verified": True,
            "alternates_required": False,
        },
    }
    compatibilities = []
    for compatibility in source.get("compatibility_revisions", []):
        compat_commit = compatibility["commit"]
        if not SHA_RE.match(compat_commit):
            raise SourceError(f"invalid compatibility revision in {source['id']}")
        run(["git", "cat-file", "-e", f"{compat_commit}^{{commit}}"], cwd=repo)
        compat_tree = run(
            ["git", "rev-parse", f"{compat_commit}^{{tree}}"], cwd=repo
        ).stdout.strip()
        compat_tag = compatibility["tag"]
        compat_marker = tag_identity(repo, compat_tag, compat_commit)
        require_tag_lines(
            compat_marker["tag_payload"],
            [
                f"Upstream-URL: {source['upstream_url']}",
                f"Compatibility-Commit: {compat_commit}",
                f"Compatibility-Tree: {compat_tree}",
            ],
            f"{source['id']}:{compat_tag}",
        )
        ensure_complete_baseline(repo, compat_commit)
        ancestor = run(
            ["git", "merge-base", "--is-ancestor", compat_commit, commit],
            cwd=repo,
            check=False,
        ).returncode == 0
        compatibilities.append(
            {
                **compatibility,
                "tree": compat_tree,
                "commit_time": run(
                    ["git", "show", "-s", "--format=%cI", compat_commit], cwd=repo
                ).stdout.strip(),
                "tag_object": compat_marker["tag_object"],
                "tag_payload": compat_marker["tag_payload"],
                "tag_payload_sha256": compat_marker["tag_payload_sha256"],
                "ancestor_of_baseline": ancestor,
                "merge_base_with_baseline": run(
                    ["git", "merge-base", compat_commit, commit], cwd=repo
                ).stdout.strip(),
                "offline_tree_verified": True,
            }
        )
    if compatibilities:
        frozen["compatibility_revisions"] = compatibilities
    return frozen


def freeze_command(args: argparse.Namespace) -> int:
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    lock = load_json(LOCK_PATH)
    if lock.get("status") == "frozen":
        raise SourceError("SOURCE_LOCK is already frozen and is append-only")
    if not LOCK_ID_RE.fullmatch(args.lock_id) or not CP_RE.fullmatch(args.checkpoint):
        raise SourceError("freeze requires canonical SL-NNNN and CP-NNNN identities")
    try:
        frozen_time = dt.datetime.fromisoformat(args.frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceError("frozen-at is not a valid ISO-8601 timestamp") from exc
    if frozen_time.tzinfo is None:
        raise SourceError("frozen-at must include an explicit timezone")
    pre_freeze_candidate = LOCK_PATH.read_bytes()
    pre_freeze_candidate_sha256 = hashlib.sha256(pre_freeze_candidate).hexdigest()
    if not args.candidate_artifact.startswith("artifacts/"):
        raise SourceError("pre-freeze candidate artifact must remain under artifacts/")
    candidate_artifact_path = safe_workspace_path(args.candidate_artifact)
    atomic_bytes(candidate_artifact_path, pre_freeze_candidate)
    accepted_parent_root_commit = run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT
    ).stdout.strip()
    accepted_parent_source_lock_sha256 = hashlib.sha256(
        git_blob_bytes(ROOT, accepted_parent_root_commit, "SOURCE_LOCK.json")
    ).hexdigest()
    root_git_dir = Path(
        run(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT).stdout.strip()
    )
    journal = load_json(
        root_git_dir / "amdgpu-sim" / "txn" / f"{args.checkpoint}.json"
    )
    if (
        journal.get("checkpoint_id") != args.checkpoint
        or journal.get("phase") != "prepare"
        or journal.get("participants_locked") is not True
        or journal.get("previous_root") != accepted_parent_root_commit
    ):
        raise SourceError("active transaction does not bind the accepted parent root")
    expected_evidence_relative = f"state/evidence/{args.resolution_evidence_id}.json"
    if args.resolution_evidence != expected_evidence_relative:
        raise SourceError("resolution evidence must use its canonical tracked path")
    evidence_path = safe_workspace_path(args.resolution_evidence)
    evidence = load_json(evidence_path)
    try:
        command_result_ids, evidence_artifacts = validate_evidence(
            evidence,
            expected_id=args.resolution_evidence_id,
            checkpoint_id=args.checkpoint,
        )
    except EvidencePolicyError as exc:
        raise SourceError(f"resolution evidence violates policy: {exc}") from exc
    if evidence.get("type") != "official-source-resolution":
        raise SourceError("resolution evidence has the wrong evidence type")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    records = evidence.get("repo_commits")
    if not isinstance(records, list) or not records:
        raise SourceError("resolution evidence has no revision records")
    record_ids = [item.get("id") for item in records if isinstance(item, dict)]
    source_ids = [item.get("id") for item in lock.get("sources", [])]
    if (
        len(record_ids) != len(records)
        or len(record_ids) != len(set(record_ids))
        or len(source_ids) != len(set(source_ids))
        or set(record_ids) != set(source_ids)
    ):
        raise SourceError("resolution evidence record set is incomplete or duplicated")
    evidence_records = {item["id"]: item for item in records}
    artifact_path = safe_workspace_path(args.resolution_artifact)
    resolution_artifact = load_json(artifact_path)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact_size = artifact_path.stat().st_size
    external = {
        item["path"]: item for item in evidence_artifacts
    }.get(args.resolution_artifact)
    if (
        resolution_artifact.get("schema") != "amdgpu-sim.source-resolution.v1"
        or external is None
        or external.get("sha256") != artifact_sha256
        or external.get("size") != artifact_size
    ):
        raise SourceError("resolution artifact is not cryptographically bound by evidence")
    model_seen = False
    frozen_sources = []
    for source in lock.get("sources", []):
        expected_revision = source.get("official_revision") or source.get("observed_head")
        record = evidence_records[source["id"]]
        expected_record = {
            "id": source["id"],
            "revision": expected_revision,
            "ref": source["upstream_ref"],
            "canonical_url": source["upstream_url"],
            "transport_url": source.get("transport_url"),
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise SourceError(f"resolution evidence provenance mismatch: {source['id']}")
        try:
            record_time = dt.datetime.fromisoformat(
                str(record.get("observed_at", "")).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise SourceError(f"resolution record time is invalid: {source['id']}") from exc
        if (
            record_time.tzinfo is None
            or record_time > frozen_time
            or record.get("command_result_id") not in command_result_ids
        ):
            raise SourceError(f"resolution evidence chronology mismatch: {source['id']}")
        if source.get("role") == "acceptance-model":
            model_seen = True
            revision = source.get("official_revision")
            if not isinstance(revision, str) or not SHA_RE.match(revision):
                raise SourceError("model official revision has not been reviewed and frozen")
            if not isinstance(source.get("files"), list) or not source["files"]:
                raise SourceError("model file inventory has not been reviewed and frozen")
            mirror = source.get("mirror_metadata")
            if not isinstance(mirror, dict):
                raise SourceError("model mirror cross-check metadata is absent")
            expected_manifest = mirror.get("normalized_revision_manifest_sha256")
            if (
                mirror.get("endpoint") != "https://hf-mirror.com"
                or mirror.get("revision") != revision
                or mirror.get("status")
                != "cross-check-only; never substitutes for official revision evidence"
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(mirror.get("mirror_api_manifest_sha256", ""))
                )
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest or ""))
            ):
                raise SourceError("model mirror cross-check provenance is inconsistent")
            if frozen_model_manifest_sha256(source) != expected_manifest:
                raise SourceError("model file inventory hash does not match reviewed metadata")
            if sum(item["size"] for item in source["files"]) != source.get(
                "materialized_size"
            ):
                raise SourceError("model materialized size does not match file inventory")
            official_evidence = source.get("official_evidence")
            try:
                official_observed = dt.datetime.fromisoformat(
                    str(source.get("official_evidence_observed_at", "")).replace(
                        "Z", "+00:00"
                    )
                )
            except ValueError as exc:
                raise SourceError("model official evidence timestamp is invalid") from exc
            if (
                source.get("official_revision_status")
                != "frozen-from-official-fixed-revision-pages"
                or not isinstance(official_evidence, list)
                or not official_evidence
                or not any(revision in url for url in official_evidence)
                or official_observed.tzinfo is None
                or source.get("official_raw_response_sha256", "missing") is not None
            ):
                raise SourceError("model official fixed-revision evidence is incomplete")
            frozen_sources.append(
                {
                    **source,
                    "resolution_evidence_id": args.resolution_evidence_id,
                    "resolution_record_id": source["id"],
                    "model_manifest_provenance": (
                        "mirror metadata cryptographically normalized and cross-checked "
                        "against official fixed-revision pages; no raw official API archive"
                    ),
                }
            )
        else:
            frozen_sources.append(
                {
                    **freeze_git_source(source),
                    "resolution_evidence_id": args.resolution_evidence_id,
                    "resolution_record_id": source["id"],
                }
            )
    if not model_seen:
        raise SourceError("acceptance model is absent from SOURCE_LOCK")
    pytorch = next(item for item in frozen_sources if item["id"] == "pytorch")
    triton = next(item for item in frozen_sources if item["id"] == "triton")
    pin = pytorch.get("triton_pin_at_observed_head")
    compat_commits = {entry["commit"] for entry in triton.get("compatibility_revisions", [])}
    if pin not in compat_commits:
        raise SourceError("PyTorch Triton pin is not frozen as a Triton compatibility revision")
    if hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest() != pre_freeze_candidate_sha256:
        raise SourceError("SOURCE_LOCK candidate changed while freeze validation was running")
    if (
        not candidate_artifact_path.is_file()
        or candidate_artifact_path.stat().st_size != len(pre_freeze_candidate)
        or file_sha256(candidate_artifact_path) != pre_freeze_candidate_sha256
    ):
        raise SourceError("pre-freeze candidate artifact changed during freeze validation")
    lock.update(
        {
            "lock_id": args.lock_id,
            "status": "frozen",
            "frozen_at": args.frozen_at,
            "frozen_by_checkpoint": args.checkpoint,
            "accepted_parent_root_commit": accepted_parent_root_commit,
            "accepted_parent_source_lock_sha256": accepted_parent_source_lock_sha256,
            "pre_freeze_candidate_sha256": pre_freeze_candidate_sha256,
            "pre_freeze_candidate_artifact": {
                "path": args.candidate_artifact,
                "sha256": pre_freeze_candidate_sha256,
                "size": len(pre_freeze_candidate),
                "required_for_resume": False,
            },
            "resolution_evidence_id": args.resolution_evidence_id,
            "resolution_evidence_path": args.resolution_evidence,
            "resolution_evidence_sha256": evidence_sha256,
            "resolution_artifact": {
                "path": args.resolution_artifact,
                "sha256": artifact_sha256,
                "size": artifact_size,
                "status": resolution_artifact.get("status"),
                "required_for_resume": False,
            },
            "observation_note": (
                "Git revisions were revalidated online and baseline-tagged. Every commit "
                "reachable from each locked head is locally traversable, and every locked "
                "baseline or compatibility tree is offline-complete. Unrelated historical "
                "trees and blobs may remain promised exactly as recorded by each lane's "
                "frozen partial-clone filter."
            ),
            "sources": frozen_sources,
        }
    )
    atomic_json(LOCK_PATH, lock)
    print(f"froze {LOCK_PATH.name} as {args.lock_id}")
    return 0


def verify_online_command(_args: argparse.Namespace) -> int:
    lock = load_json(LOCK_PATH)
    if lock.get("status") != "frozen":
        raise SourceError("online verification requires a frozen SOURCE_LOCK")
    git_sources = [
        source for source in lock.get("sources", []) if source.get("materialization") == "gitlink"
    ]
    observations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(6, len(git_sources)))
    ) as pool:
        futures = {pool.submit(resolve_git, source): source for source in git_sources}
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                resolved = future.result()
                observations.append(
                    {
                        "id": source["id"],
                        "locked_revision": source["baseline_commit"],
                        "current_revision": resolved["commit"],
                        "default_branch": resolved["default_branch"],
                        "reachable": True,
                        "branch_advanced": resolved["commit"] != source["baseline_commit"],
                    }
                )
            except Exception as exc:
                errors.append({"id": source["id"], "error": str(exc)})
    model = next(
        (source for source in lock.get("sources", []) if source.get("role") == "acceptance-model"),
        None,
    )
    if model is None:
        errors.append({"id": "acceptance-model", "error": "model is absent from lock"})
    else:
        revision = model.get("official_revision")
        url = f"https://huggingface.co/api/models/{model['repo_id']}/revision/{revision}"
        request = Request(url, headers={"User-Agent": "amdgpu-sim-source-lock/1"})
        try:
            with urlopen(request, timeout=30) as response:
                response.read(1)
                final_url = response.geturl()
                if response.status != 200 or not final_url.startswith("https://huggingface.co/"):
                    raise SourceError(
                        f"unexpected official model response: status={response.status} url={final_url}"
                    )
            observations.append(
                {
                    "id": model["id"],
                    "locked_revision": revision,
                    "current_revision": revision,
                    "reachable": True,
                    "branch_advanced": None,
                }
            )
        except (HTTPError, URLError, TimeoutError, OSError, SourceError) as exc:
            errors.append({"id": model["id"], "error": str(exc)})
    payload = {
        "schema": "amdgpu-sim.online-verification.v1",
        "checked_at": utc_now(),
        "status": "reachable" if not errors else "failed",
        "observations": sorted(observations, key=lambda item: item["id"]),
        "errors": sorted(errors, key=lambda item: item["id"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--output", default="artifacts/source-resolution.json")
    resolve.set_defaults(func=resolve_command)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--resolution", default="artifacts/source-resolution.json")
    materialize.add_argument("--output", default="artifacts/source-materialization.json")
    materialize.add_argument("--history-chunk", type=int, default=1000)
    materialize.add_argument("--blob-batch-size", type=int, default=512)
    hydration = materialize.add_mutually_exclusive_group()
    hydration.add_argument(
        "--archive-hydration",
        action="store_true",
        help="use fixed-SHA GitHub codeload instead of resumable Git object batches",
    )
    hydration.add_argument(
        "--no-archive-hydration",
        action="store_false",
        dest="archive_hydration",
        help=argparse.SUPPRESS,
    )
    materialize.set_defaults(archive_hydration=False)
    materialize.set_defaults(func=materialize_command)
    absorb = sub.add_parser("absorb")
    absorb.set_defaults(func=absorb_command)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--lock-id", required=True)
    freeze.add_argument("--frozen-at", required=True)
    freeze.add_argument("--checkpoint", required=True)
    freeze.add_argument("--resolution-evidence-id", default="EV-0002")
    freeze.add_argument(
        "--resolution-evidence", default="state/evidence/EV-0002.json"
    )
    freeze.add_argument(
        "--resolution-artifact", default="artifacts/source-resolution.json"
    )
    freeze.add_argument(
        "--candidate-artifact", default="artifacts/source-lock-pre-freeze.json"
    )
    freeze.set_defaults(func=freeze_command)
    verify_online = sub.add_parser("verify-online")
    verify_online.set_defaults(func=verify_online_command)
    return top


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except SourceError as exc:
        print(f"source manager failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
