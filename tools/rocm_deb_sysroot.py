#!/usr/bin/env python3
"""Lock and materialize a private ROCm Debian user-space sysroot.

This module deliberately stops at the ROCm user-space boundary.  A caller must
provide the HSA/ROCr implementation separately; packages or files which would
install another ROCr provider are rejected.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.parser import Parser
import fnmatch
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable


LOCK_SCHEMA = "amdgpu-sim.rocm-deb-sysroot-lock.v1"
REPOSITORY_HOST = "https://repo.radeon.com/"
FORBIDDEN_PACKAGES = frozenset(("hsa-rocr", "hsa-rocr-dev"))
FORBIDDEN_FILE_PATTERNS = (
    "**/libhsa-runtime64.so",
    "**/libhsa-runtime64.so.*",
    "**/libhsakmt.so",
    "**/libhsakmt.so.*",
)
PACKAGE_FIELDS = (
    "Package",
    "Version",
    "Architecture",
    "Filename",
    "Size",
    "SHA256",
    "Depends",
)


class SysrootError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid not in {os.getuid(), 0}
        or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
    ):
        raise SysrootError(f"expected a trusted regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after):
        raise SysrootError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def _read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SysrootError(f"invalid canonical JSON: {path}") from error
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise SysrootError(f"JSON is not a canonical object: {path}")
    return document, payload


def _safe_relative(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SysrootError(f"{label} is invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SysrootError(f"{label} is not a safe relative path")
    return path


def _run(arguments: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=True,
            capture_output=capture,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise SysrootError(
            f"command failed with exit {error.returncode}: {arguments[0]}"
            + (f": {detail[-4000:]}" if detail else "")
        ) from error


def parse_packages_index(payload: bytes) -> dict[str, dict[str, str]]:
    try:
        text = gzip.decompress(payload).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SysrootError("ROCm Packages.gz is invalid") from error
    records: dict[str, dict[str, str]] = {}
    for block in text.strip().split("\n\n"):
        parsed = dict(Parser().parsestr(block).items())
        name = parsed.get("Package")
        if not name:
            continue
        if name in records:
            raise SysrootError(f"duplicate package in ROCm index: {name}")
        missing = set(PACKAGE_FIELDS[:-1]).difference(parsed)
        if missing:
            raise SysrootError(f"package index record is incomplete: {name}")
        records[name] = {field: parsed.get(field, "") for field in PACKAGE_FIELDS}
    if not records:
        raise SysrootError("ROCm package index is empty")
    return records


def dependency_groups(value: str) -> list[list[str]]:
    result: list[list[str]] = []
    if not value.strip():
        return result
    for raw_group in value.split(","):
        alternatives: list[str] = []
        for raw_alternative in raw_group.split("|"):
            match = re.match(r"^\s*([a-z0-9][a-z0-9+.-]*)(?::[a-z0-9-]+)?", raw_alternative)
            if not match:
                raise SysrootError(f"unsupported Debian dependency: {raw_alternative!r}")
            alternatives.append(match.group(1))
        if alternatives:
            result.append(alternatives)
    return result


def resolve_packages(
    records: dict[str, dict[str, str]],
    roots: Iterable[str],
    replacements: dict[str, str],
    external_dependencies: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    selected: set[str] = set()
    resolutions: list[dict[str, Any]] = []
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name in FORBIDDEN_PACKAGES or name not in records:
            raise SysrootError(f"root ROCm package is unavailable or forbidden: {name}")
        selected.add(name)
        for alternatives in dependency_groups(records[name].get("Depends", "")):
            internal = next(
                (
                    candidate
                    for candidate in alternatives
                    if candidate in records and candidate not in FORBIDDEN_PACKAGES
                ),
                None,
            )
            replacement = next(
                (candidate for candidate in alternatives if candidate in replacements),
                None,
            )
            external = next(
                (candidate for candidate in alternatives if candidate in external_dependencies),
                None,
            )
            if internal is not None:
                pending.append(internal)
                resolution = {"kind": "package", "name": internal}
            elif replacement is not None:
                resolution = {
                    "kind": "replacement",
                    "name": replacement,
                    "provider": replacements[replacement],
                }
            elif external is not None:
                resolution = {"kind": "external", "name": external}
            else:
                raise SysrootError(
                    f"unresolved dependency of {name}: {' | '.join(alternatives)}"
                )
            resolutions.append(
                {"package": name, "alternatives": alternatives, "resolution": resolution}
            )
    resolutions.sort(
        key=lambda value: (
            value["package"],
            value["alternatives"],
            canonical_json(value["resolution"]),
        )
    )
    return sorted(selected), resolutions


def _metadata_record(root: Path, relative: Path, url: str) -> dict[str, Any]:
    record = file_record(root / relative)
    record["path"] = relative.as_posix()
    record["url"] = url
    return record


def generate_lock(
    *,
    root: Path,
    repository_directory: Path,
    package_cache_directory: Path,
    repository_base_url: str,
    suite: str,
    component: str,
    architecture: str,
    version: str,
    roots: list[str],
    replacements: dict[str, str],
    external_dependencies: list[str],
) -> dict[str, Any]:
    if not repository_base_url.startswith(REPOSITORY_HOST) or not repository_base_url.endswith("/"):
        raise SysrootError("ROCm repository base URL is invalid")
    repository_relative = repository_directory.relative_to(root)
    package_cache_relative = package_cache_directory.relative_to(root)
    packages_path = repository_directory / "Packages.gz"
    records = parse_packages_index(packages_path.read_bytes())
    selected, resolutions = resolve_packages(
        records, roots, replacements, set(external_dependencies)
    )
    packages: list[dict[str, Any]] = []
    for name in selected:
        source = records[name]
        filename = source["Filename"]
        packages.append(
            {
                "name": name,
                "version": source["Version"],
                "architecture": source["Architecture"],
                "filename": Path(filename).name,
                "repository_path": filename,
                "url": repository_base_url + filename,
                "bytes": int(source["Size"]),
                "sha256": source["SHA256"],
            }
        )
    metadata_urls = {
        "inrelease": f"{repository_base_url}dists/{suite}/InRelease",
        "release": f"{repository_base_url}dists/{suite}/Release",
        "release_signature": f"{repository_base_url}dists/{suite}/Release.gpg",
        "packages_index": (
            f"{repository_base_url}dists/{suite}/{component}/binary-{architecture}/Packages.gz"
        ),
        "signing_key": f"{REPOSITORY_HOST}rocm/rocm.gpg.key",
        "signing_keyring": f"{REPOSITORY_HOST}rocm/rocm.gpg.key#dearmored",
    }
    metadata = {
        role: _metadata_record(root, repository_relative / filename, metadata_urls[role])
        for role, filename in (
            ("inrelease", "InRelease"),
            ("release", "Release"),
            ("release_signature", "Release.gpg"),
            ("packages_index", "Packages.gz"),
            ("signing_key", "rocm.gpg.key"),
            ("signing_keyring", "rocm.gpg"),
        )
    }
    return {
        "schema": LOCK_SCHEMA,
        "provider": "signed-rocm-apt-sysroot",
        "rocm_version": version,
        "architecture": architecture,
        "sdk_root": f"opt/rocm-{version}",
        "repository": {
            "base_url": repository_base_url,
            "suite": suite,
            "component": component,
            "signing_fingerprint": "CA8BB4727A47B4D09B4EE8969386B48A1A693C5C",
            "metadata": metadata,
        },
        "package_cache_directory": package_cache_relative.as_posix(),
        "roots": sorted(set(roots)),
        "replacements": dict(sorted(replacements.items())),
        "external_dependencies": sorted(set(external_dependencies)),
        "dependency_resolution": resolutions,
        "packages": packages,
    }


def _verify_record(root: Path, descriptor: dict[str, Any], *, label: str) -> Path:
    if set(descriptor) != {"path", "url", "bytes", "sha256"}:
        raise SysrootError(f"{label} record is invalid")
    path = root / _safe_relative(descriptor["path"], label=f"{label} path")
    actual = file_record(path)
    if actual["bytes"] != descriptor["bytes"] or actual["sha256"] != descriptor["sha256"]:
        raise SysrootError(f"{label} differs from lock")
    return path


def _verify_release(repository: dict[str, Any], paths: dict[str, Path]) -> None:
    keyring = paths["signing_keyring"]
    expected_fingerprint = repository["signing_fingerprint"]
    fingerprints: list[set[str]] = []
    for key_path in (paths["signing_key"], keyring):
        result = _run(
            [
                "/usr/bin/gpg",
                "--batch",
                "--no-default-keyring",
                "--with-colons",
                "--show-keys",
                "--fingerprint",
                str(key_path),
            ],
            capture=True,
        )
        fingerprints.append(
            {
                fields[9]
                for line in result.stdout.splitlines()
                if (fields := line.split(":"))[0] == "fpr" and len(fields) > 9
            }
        )
    if (
        fingerprints[0] != fingerprints[1]
        or expected_fingerprint not in fingerprints[0]
    ):
        raise SysrootError("ROCm repository signing key fingerprint differs")
    _run(["/usr/bin/gpgv", "--keyring", str(keyring), str(paths["inrelease"])])
    _run(
        [
            "/usr/bin/gpgv",
            "--keyring",
            str(keyring),
            str(paths["release_signature"]),
            str(paths["release"]),
        ]
    )
    release = paths["release"].read_text(encoding="utf-8")
    required_headers = {
        "Suite": repository["suite"],
        "Codename": repository["suite"],
        "Version": repository["version"],
        "Architectures": repository["architecture"],
    }
    for name, expected in required_headers.items():
        if f"{name}: {expected}\n" not in release:
            raise SysrootError(f"signed ROCm Release {name} differs")
    relative = f"{repository['component']}/binary-{repository['architecture']}/Packages.gz"
    expected = paths["packages_index"]
    needle = f" {expected.stat().st_size} {relative}"
    if f" {hashlib.sha256(expected.read_bytes()).hexdigest()}{needle}\n" not in release:
        raise SysrootError("signed ROCm Release does not bind Packages.gz")


def _deb_identity(path: Path) -> tuple[str, str, str]:
    result = _run(
        [
            "/usr/bin/dpkg-deb",
            "--show",
            f"--showformat=${{Package}}\n${{Version}}\n${{Architecture}}\n",
            str(path),
        ],
        capture=True,
    )
    fields = result.stdout.splitlines()
    if len(fields) != 3 or not all(fields):
        raise SysrootError(f"Debian package metadata is invalid: {path}")
    return fields[0], fields[1], fields[2]


def load_lock(
    root: Path,
    lock_path: Path,
    *,
    require_packages: bool = True,
) -> tuple[dict[str, Any], bytes]:
    document, payload = _read_canonical(lock_path)
    required_keys = {
        "schema",
        "provider",
        "rocm_version",
        "architecture",
        "sdk_root",
        "repository",
        "package_cache_directory",
        "roots",
        "replacements",
        "external_dependencies",
        "dependency_resolution",
        "packages",
    }
    if set(document) != required_keys or document.get("schema") != LOCK_SCHEMA:
        raise SysrootError("ROCm Debian sysroot lock contract is invalid")
    if document.get("provider") != "signed-rocm-apt-sysroot":
        raise SysrootError("ROCm Debian provider is invalid")
    architecture = document.get("architecture")
    version = document.get("rocm_version")
    if architecture != "amd64" or not isinstance(version, str) or not version:
        raise SysrootError("ROCm Debian target is invalid")
    if document.get("sdk_root") != f"opt/rocm-{version}":
        raise SysrootError("ROCm Debian SDK root is invalid")
    repository = document.get("repository")
    repository_keys = {
        "base_url",
        "suite",
        "component",
        "signing_fingerprint",
        "metadata",
    }
    if not isinstance(repository, dict) or set(repository) != repository_keys:
        raise SysrootError("ROCm Debian repository record is invalid")
    if (
        not repository["base_url"].startswith(REPOSITORY_HOST)
        or not repository["base_url"].endswith("/")
        or repository["signing_fingerprint"]
        != "CA8BB4727A47B4D09B4EE8969386B48A1A693C5C"
    ):
        raise SysrootError("ROCm Debian repository authority is invalid")
    repository_with_target = {
        **repository,
        "version": version,
        "architecture": architecture,
    }
    metadata = repository.get("metadata")
    roles = {
        "inrelease",
        "release",
        "release_signature",
        "packages_index",
        "signing_key",
        "signing_keyring",
    }
    if not isinstance(metadata, dict) or set(metadata) != roles:
        raise SysrootError("ROCm Debian repository metadata set is invalid")
    paths = {
        role: _verify_record(root, metadata[role], label=f"repository {role}")
        for role in sorted(roles)
    }
    _verify_release(repository_with_target, paths)
    index = parse_packages_index(paths["packages_index"].read_bytes())
    roots = document.get("roots")
    replacements = document.get("replacements")
    external = document.get("external_dependencies")
    if (
        not isinstance(roots, list)
        or roots != sorted(set(roots))
        or not all(isinstance(name, str) and name for name in roots)
        or not isinstance(replacements, dict)
        or set(replacements) != FORBIDDEN_PACKAGES
        or not all(isinstance(value, str) and value for value in replacements.values())
        or not isinstance(external, list)
        or external != sorted(set(external))
    ):
        raise SysrootError("ROCm Debian dependency policy is invalid")
    selected, resolutions = resolve_packages(index, roots, replacements, set(external))
    if resolutions != document.get("dependency_resolution"):
        raise SysrootError("ROCm Debian dependency resolution drifted")
    package_cache = root / _safe_relative(
        document.get("package_cache_directory"), label="package cache directory"
    )
    packages = document.get("packages")
    if not isinstance(packages, list) or [item.get("name") for item in packages] != selected:
        raise SysrootError("ROCm Debian package set is invalid")
    expected_package_keys = {
        "name",
        "version",
        "architecture",
        "filename",
        "repository_path",
        "url",
        "bytes",
        "sha256",
    }
    seen_filenames: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or set(package) != expected_package_keys:
            raise SysrootError("ROCm Debian package record is invalid")
        name = package["name"]
        source = index.get(name)
        filename = package["filename"]
        if (
            name in FORBIDDEN_PACKAGES
            or source is None
            or Path(filename).name != filename
            or filename in seen_filenames
            or package["repository_path"] != source["Filename"]
            or package["url"] != repository["base_url"] + source["Filename"]
            or package["version"] != source["Version"]
            or package["architecture"] != source["Architecture"]
            or package["bytes"] != int(source["Size"])
            or package["sha256"] != source["SHA256"]
        ):
            raise SysrootError(f"ROCm Debian package differs from signed index: {name}")
        seen_filenames.add(filename)
        if require_packages:
            path = package_cache / filename
            actual = file_record(path)
            if actual["bytes"] != package["bytes"] or actual["sha256"] != package["sha256"]:
                raise SysrootError(f"ROCm Debian package cache differs: {name}")
            if _deb_identity(path) != (name, package["version"], package["architecture"]):
                raise SysrootError(f"ROCm Debian package metadata differs: {name}")
    return document, payload


def _entry(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        return {
            "kind": "regular",
            "mode": stat.S_IMODE(metadata.st_mode),
            "bytes": metadata.st_size,
            "sha256": file_record(path)["sha256"],
        }
    if stat.S_ISLNK(metadata.st_mode):
        return {"kind": "symlink", "target": os.readlink(path)}
    if stat.S_ISDIR(metadata.st_mode):
        return {"kind": "directory"}
    raise SysrootError(f"special file in ROCm Debian package: {path}")


def _validate_relative_symlink(relative: PurePosixPath, target: str) -> None:
    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        raise SysrootError(f"absolute symlink in ROCm sysroot: {relative} -> {target}")
    normalized: list[str] = []
    for component in (*relative.parent.parts, *target_path.parts):
        if component in ("", "."):
            continue
        if component == "..":
            if not normalized:
                raise SysrootError(f"symlink escapes ROCm sysroot: {relative} -> {target}")
            normalized.pop()
        else:
            normalized.append(component)


def _forbidden_file(relative: str) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in FORBIDDEN_FILE_PATTERNS)


def _merge_package(source: Path, destination: Path, owners: dict[str, list[str]], package: str) -> None:
    for current, directories, files in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = current_path / name
            relative = path.relative_to(source).as_posix()
            if _forbidden_file(relative):
                raise SysrootError(f"ROCr provider file is forbidden: {package}:{relative}")
            source_entry = _entry(path)
            target = destination / relative
            if source_entry["kind"] == "directory":
                if target.exists() and not target.is_dir():
                    raise SysrootError(f"ROCm package path collision: {relative}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            if source_entry["kind"] == "symlink":
                _validate_relative_symlink(PurePosixPath(relative), source_entry["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if _entry(target) != source_entry:
                    raise SysrootError(f"ROCm package content collision: {relative}")
            elif source_entry["kind"] == "regular":
                shutil.copy2(path, target, follow_symlinks=False)
            else:
                os.symlink(source_entry["target"], target)
            owners.setdefault(relative, []).append(package)


def tree_summary(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        symlink_directories = sorted(
            name for name in directories if (current_path / name).is_symlink()
        )
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        for name in sorted(files) + symlink_directories:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            descriptor = {"path": relative, **_entry(path)}
            digest.update(canonical_json(descriptor))
            count += 1
            total += descriptor.get("bytes", 0)
    return {"file_count": count, "regular_bytes": total, "sha256": digest.hexdigest()}


def materialize_sysroot(root: Path, lock: dict[str, Any], destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise SysrootError(f"ROCm sysroot destination must be absent: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = root / _safe_relative(
        lock["package_cache_directory"], label="package cache directory"
    )
    owners: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix=".rocm-sysroot.", dir=destination.parent) as temporary:
        temporary_root = Path(temporary)
        merged = temporary_root / "merged"
        merged.mkdir()
        extract = temporary_root / "extract"
        for package in lock["packages"]:
            if extract.exists():
                shutil.rmtree(extract)
            extract.mkdir()
            _run(["/usr/bin/dpkg-deb", "--extract", str(cache / package["filename"]), str(extract)])
            _merge_package(extract, merged, owners, package["name"])
        sdk_root = merged / lock["sdk_root"]
        if not sdk_root.is_dir():
            raise SysrootError("materialized ROCm SDK root is absent")
        for path in merged.rglob("*"):
            if path.is_file() and _forbidden_file(path.relative_to(merged).as_posix()):
                raise SysrootError(f"materialized ROCm sysroot contains another ROCr: {path}")
        summary = tree_summary(merged)
        os.replace(merged, destination)
    return {
        "schema": "amdgpu-sim.rocm-deb-sysroot-install.v1",
        "sdk_root": str(destination / lock["sdk_root"]),
        "packages": [package["name"] for package in lock["packages"]],
        "owned_path_count": len(owners),
        "tree": summary,
    }


def fetch_packages(root: Path, lock: dict[str, Any], *, jobs: int) -> dict[str, Any]:
    if not 1 <= jobs <= 16:
        raise SysrootError("package fetch jobs must be in [1,16]")
    cache = root / _safe_relative(
        lock["package_cache_directory"], label="package cache directory"
    )
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(package: dict[str, Any]) -> tuple[str, int]:
        destination = cache / package["filename"]
        if destination.exists():
            actual = file_record(destination)
            if actual["bytes"] == package["bytes"] and actual["sha256"] == package["sha256"]:
                if _deb_identity(destination) != (
                    package["name"],
                    package["version"],
                    package["architecture"],
                ):
                    raise SysrootError(f"cached Debian metadata differs: {package['name']}")
                return package["name"], package["bytes"]
            raise SysrootError(f"existing Debian cache file differs: {destination}")
        partial = destination.with_name(destination.name + ".partial")
        _run(
            [
                "/usr/bin/curl",
                "--fail",
                "--location",
                "--retry",
                "12",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--continue-at",
                "-",
                "--silent",
                "--show-error",
                "--output",
                str(partial),
                package["url"],
            ]
        )
        actual = file_record(partial)
        if actual["bytes"] != package["bytes"] or actual["sha256"] != package["sha256"]:
            raise SysrootError(f"downloaded Debian package differs: {package['name']}")
        if _deb_identity(partial) != (
            package["name"],
            package["version"],
            package["architecture"],
        ):
            raise SysrootError(f"downloaded Debian metadata differs: {package['name']}")
        os.replace(partial, destination)
        return package["name"], package["bytes"]

    completed: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(fetch, package): package["name"] for package in lock["packages"]}
        try:
            for future in as_completed(futures):
                completed.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    completed.sort()
    return {
        "schema": "amdgpu-sim.rocm-deb-package-cache.v1",
        "package_count": len(completed),
        "bytes": sum(value for _, value in completed),
        "packages": [name for name, _ in completed],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lock", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--materialize", type=Path)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--fetch", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-directory", type=Path)
    parser.add_argument("--package-cache-directory", type=Path)
    parser.add_argument("--repository-base-url")
    parser.add_argument("--suite")
    parser.add_argument("--component")
    parser.add_argument("--architecture")
    parser.add_argument("--rocm-version")
    parser.add_argument("--root-package", action="append", default=[])
    parser.add_argument("--replacement", action="append", default=[])
    parser.add_argument("--external-dependency", action="append", default=[])
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve(strict=True)
    try:
        if arguments.generate:
            required = {
                "output": arguments.output,
                "repository-directory": arguments.repository_directory,
                "package-cache-directory": arguments.package_cache_directory,
                "repository-base-url": arguments.repository_base_url,
                "suite": arguments.suite,
                "component": arguments.component,
                "architecture": arguments.architecture,
                "rocm-version": arguments.rocm_version,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing or not arguments.root_package:
                raise SysrootError(
                    "lock generation arguments are incomplete: " + ",".join(missing)
                )
            replacements: dict[str, str] = {}
            for value in arguments.replacement:
                name, separator, provider = value.partition("=")
                if not separator or not name or not provider or name in replacements:
                    raise SysrootError(f"invalid replacement: {value}")
                replacements[name] = provider
            repository_directory = arguments.repository_directory
            package_cache_directory = arguments.package_cache_directory
            output = arguments.output
            if not repository_directory.is_absolute():
                repository_directory = root / repository_directory
            if not package_cache_directory.is_absolute():
                package_cache_directory = root / package_cache_directory
            if not output.is_absolute():
                output = root / output
            if output.exists() or output.is_symlink():
                raise SysrootError(f"lock output must be absent: {output}")
            result = generate_lock(
                root=root,
                repository_directory=repository_directory.resolve(strict=True),
                package_cache_directory=package_cache_directory.resolve(strict=True),
                repository_base_url=arguments.repository_base_url,
                suite=arguments.suite,
                component=arguments.component,
                architecture=arguments.architecture,
                version=arguments.rocm_version,
                roots=arguments.root_package,
                replacements=replacements,
                external_dependencies=arguments.external_dependency,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(canonical_json(result))
                stream.flush()
                os.fsync(stream.fileno())
            result = {"schema": LOCK_SCHEMA, "output": str(output), "package_count": len(result["packages"])}
        else:
            if arguments.lock is None:
                raise SysrootError("--lock is required")
            lock_path = arguments.lock
            if not lock_path.is_absolute():
                lock_path = root / lock_path
            lock, _ = load_lock(
                root,
                lock_path.resolve(strict=True),
                require_packages=not arguments.fetch,
            )
            if arguments.fetch:
                result = fetch_packages(root, lock, jobs=arguments.jobs)
                load_lock(root, lock_path.resolve(strict=True), require_packages=True)
            elif arguments.materialize is None:
                result = {"schema": LOCK_SCHEMA, "package_count": len(lock["packages"])}
            else:
                destination = arguments.materialize
                if not destination.is_absolute():
                    destination = root / destination
                result = materialize_sysroot(root, lock, destination)
    except (SysrootError, OSError, ValueError) as error:
        print(f"ROCm Debian sysroot error: {error}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
