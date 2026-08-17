#!/usr/bin/env python3
"""Create a deterministic AgentENV runtime bundle.

The bundle is deliberately built from a manifest rather than from a directory
upload.  AgentENV's directory upload path does not preserve all POSIX metadata,
while ROCm/conda installations rely on symlinks and executable bits.  This
tool therefore walks the selected workspace roots, records a content manifest,
and writes a reproducible tar.zst archive.  It is safe to use with ``--dry-run``
on a host that is running other workloads.

The command does not start a VM or modify the AgentENV service.  The companion
``agentenv_manager.py`` consumes the manifest when creating sandboxes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterator, Sequence


BUNDLE_SCHEMA = "amdgpu-sim.agentenv-runtime-bundle.v1"
MANIFEST_SCHEMA = "amdgpu-sim.agentenv-runtime-manifest.v1"
DEFAULT_GUEST_ROOT = "/home/zhaosiying/amdgpu-sim"
DEFAULT_ACTIVE_POINTERS = (
    "env/rocm/active-product",
    "env/conda/active-product",
    "env/conda/active-rocm-pytorch",
)
DEFAULT_INCLUDES = (
    "config",
    "examples",
    "plugins",
    "protocol",
    "scripts",
    "tools",
    "projects/gem5/configs/example/gemsim",
    "projects/self-amdgpu-runtime/tools",
    "projects/vllm",
    "projects/sglang",
    "projects/sglang-0.5.17",
    "projects/sglang-v0.5.17-src",
)
DEFAULT_GEM5_CANDIDATES = (
    "projects/gem5/build/VEGA_X86/gem5.opt",
    "projects/gem5/build/HOSTGPU_NATIVE_CONTROL/gem5.opt",
)
DEFAULT_MODEL_CANDIDATES = (
    "models/Qwen/Qwen3.5-0.8B",
    "models/Qwen3.5-0.8B",
    "models/Qwen3.5-0.8b",
)


class BundleError(RuntimeError):
    """Raised when a bundle cannot be made without violating its contract."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    name: str
    guest: PurePosixPath


@dataclass
class Entry:
    source: Path
    archive_name: str
    kind: str
    mode: int
    size: int = 0
    sha256: str | None = None
    link_target: str | None = None
    link_target_present: bool | None = None


class BundleBuilder:
    """Collect and materialize a reproducible runtime bundle."""

    def __init__(
        self,
        *,
        source_roots: Sequence[Path],
        guest_root: str = DEFAULT_GUEST_ROOT,
        allow_external_roots: Sequence[Path] = (),
        follow_external_symlinks: bool = False,
    ) -> None:
        if not source_roots:
            raise BundleError("at least one source root is required")
        guest = PurePosixPath(guest_root)
        if not guest.is_absolute() or guest == PurePosixPath("/"):
            raise BundleError("guest root must be an absolute non-root path")
        self.guest_root = guest
        roots: list[SourceRoot] = []
        seen: set[Path] = set()
        for candidate in source_roots:
            path = absolute_path(candidate)
            if not path.is_dir():
                raise BundleError(f"source root is not a directory: {path}")
            real = Path(os.path.realpath(path))
            if real in seen:
                continue
            seen.add(real)
            roots.append(SourceRoot(path=path, name=path.name or "root", guest=guest))
        for candidate in allow_external_roots:
            path = absolute_path(candidate)
            if not path.is_dir():
                raise BundleError(f"external source root is not a directory: {path}")
            real = Path(os.path.realpath(path))
            if real in seen:
                continue
            seen.add(real)
            roots.append(
                SourceRoot(
                    path=path,
                    name=path.name or "root",
                    guest=PurePosixPath(str(path)),
                )
            )
        # Longest first ensures a nested root maps to its own exact archive
        # path rather than being swallowed by its parent root.
        self.roots = tuple(sorted(roots, key=lambda item: len(item.path.parts), reverse=True))
        self.allowed_real_roots = tuple(
            Path(os.path.realpath(absolute_path(item)))
            for item in (*source_roots, *allow_external_roots)
        )
        self.follow_external_symlinks = follow_external_symlinks
        self.entries: dict[str, Entry] = {}
        self.missing: list[dict[str, str]] = []
        self.unresolved_references: list[dict[str, str]] = []
        self.overridden: list[dict[str, str]] = []
        self.skipped_references: list[dict[str, str]] = []
        self.labels: dict[str, list[str]] = {}
        self._visited_dirs: set[Path] = set()
        self._parsed_manifests: set[Path] = set()

    def _allowed(self, path: Path) -> bool:
        real = Path(os.path.realpath(path))
        return any(is_relative_to(real, root) for root in self.allowed_real_roots)

    def _archive_name(self, path: Path) -> str:
        lexical = absolute_path(path)
        for root in self.roots:
            if is_relative_to(lexical, root.path):
                relative = lexical.relative_to(root.path)
                guest = root.guest.joinpath(*relative.parts)
                return PurePosixPath(str(guest).lstrip("/")).as_posix()
        raise BundleError(
            f"path is outside source roots and cannot retain absolute layout: {path}"
        )

    def _record_missing(self, path: Path, label: str) -> None:
        self.missing.append({"path": str(path), "label": label})

    def add(self, path: Path, *, label: str, required: bool = True) -> None:
        path = absolute_path(path)
        if not os.path.lexists(path):
            if required:
                self._record_missing(path, label)
            return
        if not self._allowed(path):
            if not self.follow_external_symlinks:
                raise BundleError(
                    f"refusing path outside allowed roots (use --allow-external-root): {path}"
                )
        self.labels.setdefault(str(path), []).append(label)
        self._walk(path)

    def _walk(self, path: Path) -> None:
        lexical = absolute_path(path)
        archive_name = self._archive_name(lexical)
        metadata = os.lstat(lexical)
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(lexical)
            resolved = Path(os.path.realpath(lexical))
            target_present = os.path.lexists(resolved)
            entry = Entry(
                source=lexical,
                archive_name=archive_name,
                kind="symlink",
                mode=mode,
                link_target=target,
                link_target_present=target_present,
            )
            self._merge_entry(entry)
            if self._allowed(resolved):
                # Include the target closure, while retaining the original
                # symlink in the archive.  Absolute links continue to work in
                # the guest because extraction preserves the documented guest
                # path (/home/zhaosiying/amdgpu-sim).
                # Package trees can intentionally ship platform-specific
                # dangling links. Preserve those links without trying to walk
                # a nonexistent target, and make that state explicit in the
                # manifest. A target outside the allowed roots is still
                # rejected below even when it does not exist.
                if target_present:
                    self._walk(resolved)
            elif self.follow_external_symlinks:
                raise BundleError(
                    "external symlink targets need an explicit source-root mapping "
                    f"so absolute guest references remain valid: {lexical} -> {target}"
                )
            else:
                raise BundleError(
                    f"symlink target escapes allowed roots: {lexical} -> {target}"
                )
            return
        if stat.S_ISDIR(metadata.st_mode):
            entry = Entry(source=lexical, archive_name=archive_name, kind="directory", mode=mode)
            self._merge_entry(entry)
            real = Path(os.path.realpath(lexical))
            if real in self._visited_dirs:
                return
            self._visited_dirs.add(real)
            for child in sorted(lexical.iterdir(), key=lambda item: item.name):
                self._walk(child)
            return
        if stat.S_ISREG(metadata.st_mode):
            entry = Entry(
                source=lexical,
                archive_name=archive_name,
                kind="file",
                mode=mode,
                size=metadata.st_size,
            )
            self._merge_entry(entry)
            return
        raise BundleError(f"unsupported special file in bundle: {lexical}")

    def _merge_entry(self, entry: Entry) -> None:
        prior = self.entries.get(entry.archive_name)
        if prior is None:
            self.entries[entry.archive_name] = entry
            return
        # A source can be reached both from a selected tree and from a
        # manifest-referenced symlink.  Identical source paths are harmless;
        # different sources mapping to one guest path are ambiguous and must
        # fail instead of silently changing the runtime.
        if os.path.realpath(prior.source) == os.path.realpath(entry.source):
            return
        prior_rank = self._root_rank(prior.source)
        entry_rank = self._root_rank(entry.source)
        if prior_rank is None or entry_rank is None or prior_rank == entry_rank:
            raise BundleError(
                "two source paths map to one guest path: "
                f"{prior.source} and {entry.source} -> {entry.archive_name}"
            )
        # The feature worktree is passed first and wins over the read-only
        # sibling checkout.  This permits a staged file in the new branch to
        # override an unchanged file while still pulling active environments
        # from the sibling checkout when they are absent in the worktree.
        if entry_rank < prior_rank:
            winner, loser = entry, prior
            self.entries[entry.archive_name] = entry
        else:
            winner, loser = prior, entry
        self.overridden.append(
            {
                "path": entry.archive_name,
                "winner": str(winner.source),
                "loser": str(loser.source),
            }
        )

    def _root_rank(self, path: Path) -> int | None:
        lexical = absolute_path(path)
        for index, root in enumerate(self.roots):
            if is_relative_to(lexical, root.path):
                return index
        return None

    @staticmethod
    def _json_strings(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from BundleBuilder._json_strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from BundleBuilder._json_strings(item)

    def _resolve_manifest_reference(self, value: str) -> Path | None:
        if not value.startswith("/") or "\0" in value:
            return None
        path = Path(value)
        if os.path.lexists(path):
            return absolute_path(path)
        try:
            relative = PurePosixPath(value).relative_to(self.guest_root)
        except ValueError:
            return None
        for root in self.roots:
            if root.guest != self.guest_root:
                continue
            candidate = root.path.joinpath(*relative.parts)
            if os.path.lexists(candidate):
                return candidate
        return None

    def add_manifest_closure(self, *, strict: bool) -> None:
        """Include whole-path references from product control manifests.

        ROCm product manifests contain absolute references to base prefixes in
        addition to ordinary symlinks.  Parsing only a narrow control-file set
        avoids treating arbitrary package metadata as a runtime dependency.
        Newly discovered prefix manifests are parsed to a fixed point.
        """

        while True:
            candidates = [
                entry.source
                for entry in self.entries.values()
                if entry.kind == "file"
                and (
                    entry.source.name == "manifest.json"
                    or entry.source.name.endswith(".manifest.json")
                    or entry.source.name in {"active-product", "active-rocm-pytorch"}
                )
                and entry.source not in self._parsed_manifests
            ]
            if not candidates:
                return
            for manifest in sorted(candidates):
                self._parsed_manifests.add(manifest)
                try:
                    metadata = manifest.stat()
                    if metadata.st_size > 64 * 1024 * 1024:
                        raise BundleError(f"control manifest is unexpectedly large: {manifest}")
                    document = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise BundleError(f"invalid control manifest {manifest}: {error}") from error
                for value in sorted(set(self._json_strings(document))):
                    if not value.startswith("/"):
                        continue
                    if "amdgpu-sim-fastcopy" in value:
                        raise BundleError(
                            "control manifest depends on excluded fastcopy worktree: "
                            f"{manifest}: {value}"
                        )
                    reference = self._resolve_manifest_reference(value)
                    if reference is None:
                        item = {"manifest": str(manifest), "path": value}
                        self.unresolved_references.append(item)
                        if strict and value.startswith(str(self.guest_root / "env")):
                            raise BundleError(
                                f"runtime manifest reference is missing: {manifest}: {value}"
                            )
                        continue
                    if not self._allowed(reference):
                        continue
                    if self._skip_manifest_reference(reference):
                        self.skipped_references.append(
                            {"manifest": str(manifest), "path": value, "reason": "non-runtime-tree"}
                        )
                        continue
                    # Prefix directories are closure dependencies.  Other
                    # directory-valued source/build metadata is descriptive;
                    # only its concrete referenced files are included.
                    if reference.is_dir() and not self._is_runtime_directory(reference):
                        self.skipped_references.append(
                            {
                                "manifest": str(manifest),
                                "path": value,
                                "reason": "non-runtime-directory",
                            }
                        )
                        continue
                    self.add(
                        reference,
                        label=f"manifest-ref:{self._archive_name(manifest)}",
                        required=True,
                    )

    def _guest_relative(self, path: Path) -> PurePosixPath | None:
        try:
            return PurePosixPath("/" + self._archive_name(path)).relative_to(self.guest_root)
        except ValueError:
            return None

    def _skip_manifest_reference(self, path: Path) -> bool:
        relative = self._guest_relative(path)
        if relative is None or not relative.parts:
            return True
        top = relative.parts[0]
        if top in {"artifacts", "cache", "logs", "tmp", "runs", "downloads"}:
            return True
        if len(relative.parts) >= 2 and relative.parts[0:2] == ("env", "product-state"):
            return True
        return False

    def _is_runtime_directory(self, path: Path) -> bool:
        relative = self._guest_relative(path)
        if relative is None or not relative.parts:
            return False
        if relative.parts[0] == "models":
            return True
        if relative.parts[0] != "env":
            return False
        return any(
            part.startswith(("product-v1-", "rocm-pytorch-", "gfx950-", "hip-facade-stage"))
            for part in relative.parts
        )

    def _hash_entries(self, *, hash_files: bool) -> None:
        for entry in self.entries.values():
            if entry.kind != "file":
                continue
            if hash_files:
                entry.sha256 = sha256_file(entry.source)

    def manifest(self, *, hash_files: bool) -> dict[str, Any]:
        self._hash_entries(hash_files=hash_files)
        entries = []
        total_bytes = 0
        for name in sorted(self.entries):
            entry = self.entries[name]
            item: dict[str, Any] = {
                "path": "/" + name,
                "kind": entry.kind,
                "mode": entry.mode,
            }
            if entry.kind == "file":
                item["size"] = entry.size
                if entry.sha256 is not None:
                    item["sha256"] = entry.sha256
                total_bytes += entry.size
            elif entry.kind == "symlink":
                item["target"] = entry.link_target
                item["target_present"] = entry.link_target_present
            entries.append(item)
        source_roots = [str(item.path) for item in self.roots]
        source_mappings = [
            {"source": str(item.path), "guest": str(item.guest)} for item in self.roots
        ]
        labels = {
            path: sorted(set(values)) for path, values in sorted(self.labels.items())
        }
        return {
            "schema": MANIFEST_SCHEMA,
            "bundle_schema": BUNDLE_SCHEMA,
            "guest_root": str(self.guest_root),
            "source_roots": source_roots,
            "source_mappings": source_mappings,
            "entries": entries,
            "entry_count": len(entries),
            "regular_file_bytes": total_bytes,
            "missing": sorted(self.missing, key=lambda item: (item["path"], item["label"])),
            "unresolved_manifest_references": sorted(
                self.unresolved_references,
                key=lambda item: (item["manifest"], item["path"]),
            ),
            "overridden": sorted(
                self.overridden,
                key=lambda item: (item["path"], item["winner"], item["loser"]),
            ),
            "skipped_manifest_references": sorted(
                self.skipped_references,
                key=lambda item: (item["manifest"], item["path"], item["reason"]),
            ),
            "labels": labels,
        }

    def dry_run(self, *, hash_files: bool = False) -> dict[str, Any]:
        return self.manifest(hash_files=hash_files)

    def write(
        self,
        output: Path,
        *,
        manifest_output: Path | None = None,
        compression_level: int = 19,
    ) -> dict[str, Any]:
        manifest = self.manifest(hash_files=True)
        if manifest["missing"]:
            missing = ", ".join(item["path"] for item in manifest["missing"])
            raise BundleError(f"required bundle inputs are missing: {missing}")
        output = absolute_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("zstd") is None:
            raise BundleError("zstd is required to write a .tar.zst bundle")
        with tempfile.TemporaryDirectory(prefix="agentenv-bundle-", dir=output.parent) as temp_dir:
            tar_path = Path(temp_dir) / "runtime.tar"
            with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(self.entries):
                    self._write_entry(archive, self.entries[name])
                payload = canonical_json(manifest)
                info = tarfile.TarInfo("agentenv-bundle/manifest.json")
                info.size = len(payload)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                archive.addfile(info, fileobj=_BytesReader(payload))
            compressed = Path(temp_dir) / "runtime.tar.zst"
            command = [
                "zstd",
                "--quiet",
                "--force",
                f"-{compression_level}",
                "--threads=1",
                "--no-check",
                str(tar_path),
                "-o",
                str(compressed),
            ]
            proc = subprocess.run(command, check=False, text=True, capture_output=True)
            if proc.returncode:
                raise BundleError(
                    "zstd failed with exit "
                    f"{proc.returncode}: {(proc.stderr or proc.stdout).strip()}"
                )
            os.replace(compressed, output)
        archive_sha256 = sha256_file(output)
        result = {
            **manifest,
            "archive": str(output),
            "archive_bytes": output.stat().st_size,
            "archive_sha256": archive_sha256,
        }
        if manifest_output is None:
            manifest_output = output.with_suffix(output.suffix + ".manifest.json")
        manifest_output = absolute_path(manifest_output)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_bytes(canonical_json(result))
        return result

    @staticmethod
    def _write_entry(archive: tarfile.TarFile, entry: Entry) -> None:
        info = tarfile.TarInfo(entry.archive_name)
        info.mode = entry.mode
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        if entry.kind == "directory":
            info.type = tarfile.DIRTYPE
            info.size = 0
            archive.addfile(info)
        elif entry.kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = entry.link_target or ""
            info.size = 0
            archive.addfile(info)
        else:
            info.type = tarfile.REGTYPE
            info.size = entry.size
            with entry.source.open("rb") as stream:
                archive.addfile(info, fileobj=stream)


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self.payload) - self.offset
        start = self.offset
        self.offset = min(len(self.payload), self.offset + size)
        return self.payload[start : self.offset]


def discover_roots(repo: Path, explicit: Sequence[Path]) -> list[Path]:
    roots = [repo]
    if not explicit:
        sibling = repo.parent / "amdgpu-sim"
        if sibling.is_dir() and sibling != repo:
            roots.append(sibling)
    roots.extend(explicit)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in roots:
        path = absolute_path(candidate)
        real = Path(os.path.realpath(path))
        if real in seen:
            continue
        seen.add(real)
        unique.append(path)
    return unique


def _find_relative(roots: Sequence[Path], relative: str) -> Path | None:
    for root in roots:
        candidate = root / relative
        if os.path.lexists(candidate):
            return candidate
    return None


def _active_prefixes(
    roots: Sequence[Path], pointers: Sequence[str]
) -> list[tuple[Path, Path, str]]:
    prefixes: list[tuple[Path, Path, str]] = []
    for pointer in pointers:
        path = _find_relative(roots, pointer)
        if path is None:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BundleError(f"invalid active environment pointer {path}: {error}") from error
        prefix = document.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            raise BundleError(f"active environment pointer has no prefix: {path}")
        prefix_path = Path(prefix)
        if not prefix_path.is_absolute():
            prefix_path = path.parent / prefix_path
        prefixes.append((path, prefix_path, f"active:{pointer}"))
    return prefixes


def _default_candidates(roots: Sequence[Path], relative_paths: Sequence[str]) -> list[Path]:
    for relative in relative_paths:
        path = _find_relative(roots, relative)
        if path is not None:
            return [path]
    return []


def build_from_args(args: argparse.Namespace) -> tuple[BundleBuilder, dict[str, Any]]:
    repo = absolute_path(Path(args.repo))
    roots = discover_roots(repo, [Path(item) for item in args.source_root])
    builder = BundleBuilder(
        source_roots=roots,
        guest_root=args.guest_root,
        allow_external_roots=[Path(item) for item in args.allow_external_root],
        follow_external_symlinks=args.follow_external_symlinks,
    )
    includes = args.include or list(DEFAULT_INCLUDES)
    for relative in includes:
        path = _find_relative(roots, relative) or (repo / relative)
        builder.add(path, label=f"include:{relative}", required=not args.allow_missing)
    if not args.no_active_env:
        pointers = args.active_pointer or list(DEFAULT_ACTIVE_POINTERS)
        active = _active_prefixes(roots, pointers)
        if not active and not args.allow_missing:
            raise BundleError(
                "no active ROCm environment pointers were found; pass --allow-missing "
                "only for a planning dry-run"
            )
        for pointer_path, prefix_path, label in active:
            builder.add(pointer_path, label=f"pointer:{label}", required=True)
            builder.add(prefix_path, label=label, required=True)
    if not args.no_gem5:
        gem5 = Path(args.gem5) if args.gem5 else None
        if gem5 is not None and not gem5.is_absolute():
            gem5 = repo / gem5
        candidates = [gem5] if gem5 else _default_candidates(roots, DEFAULT_GEM5_CANDIDATES)
        for candidate in candidates:
            builder.add(candidate, label="gem5", required=not args.allow_missing)
    if not args.no_model:
        model = Path(args.model) if args.model else None
        if model is not None and not model.is_absolute():
            model = repo / model
        candidates = [model] if model else _default_candidates(roots, DEFAULT_MODEL_CANDIDATES)
        for candidate in candidates:
            builder.add(candidate, label="model", required=not args.allow_missing)
    if not args.no_manifest_closure:
        builder.add_manifest_closure(strict=not args.allow_unresolved_manifest_refs)
    return builder, builder.dry_run(hash_files=False)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-root", action="append", default=[], help="additional source root")
    parser.add_argument("--allow-external-root", action="append", default=[])
    parser.add_argument("--follow-external-symlinks", action="store_true")
    parser.add_argument("--guest-root", default=DEFAULT_GUEST_ROOT)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--active-pointer", action="append", default=[])
    parser.add_argument("--no-active-env", action="store_true")
    parser.add_argument("--gem5")
    parser.add_argument("--model")
    parser.add_argument("--no-gem5", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--no-manifest-closure", action="store_true")
    parser.add_argument("--allow-unresolved-manifest-refs", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", default="build/agentenv-runtime-bundle.tar.zst")
    parser.add_argument("--manifest-output")
    parser.add_argument("--compression-level", type=int, default=19, choices=range(1, 20))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hash-dry-run", action="store_true", help="hash files during dry-run")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        builder, _ = build_from_args(args)
        if args.dry_run:
            result = builder.manifest(hash_files=args.hash_dry_run)
        else:
            output = Path(args.output)
            if not output.is_absolute():
                output = Path(args.repo) / output
            manifest_output = Path(args.manifest_output) if args.manifest_output else None
            if manifest_output is not None and not manifest_output.is_absolute():
                manifest_output = Path(args.repo) / manifest_output
            result = builder.write(
                output,
                manifest_output=manifest_output,
                compression_level=args.compression_level,
            )
        if args.json:
            sys.stdout.buffer.write(canonical_json(result))
        else:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    except BundleError as error:
        print(f"agentenv bundle: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
