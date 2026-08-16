#!/usr/bin/env python3
"""Freeze a pip resolution report into a verified, offline wheel lock.

The resolver is allowed to choose versions, but publication is stricter: every
artifact must be a CPython 3.12 wheel from the pinned ROCm index or PyPI's
content store, and the bytes in the repository cache must match pip's hash.
This utility deliberately does not install packages or execute package code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from urllib.parse import quote, unquote, urlparse
from zipfile import BadZipFile, ZipFile


OFFICIAL_ROOT = (
    "https://wheels.vllm.ai/rocm/"
    "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"
)
ALLOWED_HOSTS = {"files.pythonhosted.org", "wheels.vllm.ai"}
OFFICIAL_NAMES = {
    "amd-aiter",
    "amdsmi",
    "flash-attn",
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
    "vllm",
}
SCHEMA = "amdgpu-sim.rocm-pytorch-wheel-lock.v1"


class LockError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
            or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
        raise LockError(f"untrusted wheel cache file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
        after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev
    ):
        raise LockError(f"wheel changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def normalized_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def wheel_url(report_url: str, name: str, filename: str) -> str:
    parsed = urlparse(report_url)
    if parsed.scheme == "file":
        if normalized_name(name) not in OFFICIAL_NAMES:
            raise LockError(f"unmapped direct wheel: {name}")
        return f"{OFFICIAL_ROOT}/{quote(filename, safe='._-+')}"
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_HOSTS:
        raise LockError(f"wheel URL host is not allowed: {report_url}")
    return report_url


def metadata(path: Path) -> tuple[str, str]:
    try:
        with ZipFile(path) as archive:
            entries = [n for n in archive.namelist()
                       if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
            if len(entries) != 1:
                raise LockError(f"metadata count differs: {path.name}")
            lines = archive.read(entries[0]).decode("utf-8").splitlines()
    except (BadZipFile, OSError, UnicodeDecodeError) as error:
        raise LockError(f"invalid wheel: {path}") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.startswith("Name: "):
            fields.setdefault("name", normalized_name(line[6:]))
        elif line.startswith("Version: "):
            fields.setdefault("version", line[9:])
        if set(fields) == {"name", "version"}:
            break
    if set(fields) != {"name", "version"}:
        raise LockError(f"wheel metadata is incomplete: {path.name}")
    return fields["name"], fields["version"]


def load_report(path: Path) -> list[dict[str, object]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"invalid pip report: {path}") from error
    entries = document.get("install")
    if not isinstance(entries, list) or not entries:
        raise LockError("pip report has no install entries")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise LockError("pip report entry is not an object")
        meta = item.get("metadata")
        info = item.get("download_info")
        if not isinstance(meta, dict) or not isinstance(info, dict):
            raise LockError("pip report entry lacks metadata/download_info")
        name = normalized_name(str(meta.get("name", "")))
        version = str(meta.get("version", ""))
        url = info.get("url")
        hashes = info.get("archive_info")
        if not name or not version or not isinstance(url, str) or not isinstance(hashes, dict):
            raise LockError("pip report entry has invalid identity")
        expected = hashes.get("hashes")
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise LockError(f"pip report lacks sha256: {name}")
        raw_filename = unquote(Path(urlparse(url).path).name)
        if not raw_filename.endswith(".whl") or Path(raw_filename).name != raw_filename:
            raise LockError(f"non-wheel artifact in report: {name}")
        if name in seen:
            raise LockError(f"duplicate resolved package: {name}")
        seen.add(name)
        result.append({
            "name": name,
            "version": version,
            "filename": raw_filename,
            "url": wheel_url(url, name, raw_filename),
            "sha256": expected["sha256"],
        })
    return sorted(result, key=lambda record: str(record["name"]))


def fetch_one(record: dict[str, object], cache: Path) -> None:
    filename = str(record["filename"])
    destination = cache / filename
    expected = str(record["sha256"])
    if destination.is_file():
        size, digest = sha256(destination)
        if digest == expected:
            return
        raise LockError(f"existing wheel hash differs: {destination}")
    temporary = destination.with_name(f".{filename}.partial.{os.getpid()}")
    command = [
        "/usr/bin/curl", "-fL", "--retry", "12", "--retry-all-errors",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
        "-o", str(temporary), str(record["url"]),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
        size, digest = sha256(temporary)
        if digest != expected:
            raise LockError(f"download hash differs: {filename}")
        os.replace(temporary, destination)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()[-1000:]
        raise LockError(f"download failed for {filename}: {detail}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def generate(report: Path, cache: Path, output: Path, jobs: int) -> None:
    records = load_report(report)
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fetch_one, record, cache) for record in records]
        for future in as_completed(futures):
            future.result()
    packages: list[dict[str, object]] = []
    for record in records:
        path = cache / str(record["filename"])
        size, digest = sha256(path)
        if digest != record["sha256"]:
            raise LockError(f"cache hash differs after fetch: {path.name}")
        actual_name, actual_version = metadata(path)
        if actual_name != record["name"] or actual_version != record["version"]:
            raise LockError(f"metadata differs: {path.name}")
        packages.append({**record, "bytes": size})
    document = {
        "schema": SCHEMA,
        "architecture": "gfx950",
        "cache_directory": str(cache),
        "index_url": "https://pypi.org/simple/",
        "python": {"implementation": "CPython", "version": "3.12", "abi": "cp312"},
        "rocm_version": "7.2.3",
        "packages": packages,
        "resolution_report": {"path": str(report), "sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_bytes(canonical(document))
    os.replace(temporary, output)
    print(json.dumps({"packages": len(packages), "bytes": sum(int(p["bytes"]) for p in packages),
                      "output": str(output)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.jobs > 32:
        parser.error("--jobs must be in 1..32")
    try:
        generate(args.report, args.cache, args.output, args.jobs)
    except (LockError, OSError, ValueError) as error:
        print(f"wheel lock error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
