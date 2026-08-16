"""Framework-neutral bootstrap binding for out-of-tree CCL adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Mapping
import re


SCHEMA = "amdgpu-sim.ccl-bootstrap.v2"
LEGACY_VLLM_SCHEMA = "amdgpu-sim.vllm-ccl-bootstrap.v1"
_MAX_BYTES = 1024 * 1024
_claimed_groups: set[tuple[str, str]] = set()
_claim_lock = threading.RLock()


class BootstrapError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
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


def _read_regular(path: Path, *, private: bool, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"could not safely open {path}") from error
    try:
        metadata = os.fstat(descriptor)
        forbidden = (
            stat.S_IRWXG | stat.S_IRWXO
            if private
            else stat.S_IWGRP | stat.S_IWOTH
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & forbidden
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise BootstrapError(f"unsafe file identity for {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise BootstrapError(f"file was truncated while reading {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BootstrapError(f"file changed while reading {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _mapping(value: Any, keys: tuple[str, ...], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise BootstrapError(f"{label} fields are invalid")
    return value


def _integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise BootstrapError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _hex(value: Any, size: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != size * 2:
        raise BootstrapError(f"{label} has the wrong width")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise BootstrapError(f"{label} is not lowercase hex") from error
    if decoded.hex() != value or not any(decoded):
        raise BootstrapError(f"{label} is not canonical nonzero hex")
    return value


def _normal_absolute(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise BootstrapError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise BootstrapError(f"{label} must be normalized and absolute")
    return path


def _file_record(value: Any, label: str) -> tuple[Path, bytes]:
    record = _mapping(value, ("path", "bytes", "sha256"), label)
    path = _normal_absolute(record["path"], f"{label}.path")
    expected_bytes = _integer(record["bytes"], 1, 1 << 40, f"{label}.bytes")
    expected_sha = _hex(record["sha256"], 32, f"{label}.sha256")
    payload = _read_regular(path, private=False, maximum=1 << 40)
    if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise BootstrapError(f"{label} content identity mismatch")
    return path, payload


def _validate_product(value: Any) -> dict[str, Any]:
    product = _mapping(
        value,
        ("prefix", "manifest", "runtime_library"),
        "product",
    )
    prefix = _normal_absolute(product["prefix"], "product.prefix").resolve(strict=True)
    manifest_path, manifest_payload = _file_record(product["manifest"], "product.manifest")
    runtime_path, _ = _file_record(
        product["runtime_library"], "product.runtime_library"
    )
    if manifest_path != prefix / "manifest.json":
        raise BootstrapError("product manifest path does not match the product prefix")
    try:
        manifest = json.loads(manifest_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("product manifest is invalid JSON") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != "amdgpu-sim.product-prefix.v1"
        or manifest.get("prefix") != str(prefix)
        or manifest_payload != _canonical_json(manifest)
    ):
        raise BootstrapError("product manifest contract mismatch")
    runtime = manifest.get("artifacts", {}).get("runtime_library")
    if runtime != dict(product["runtime_library"]) or runtime_path != Path(runtime["path"]):
        raise BootstrapError("bootstrap runtime differs from the frozen product")
    return {
        "prefix": prefix,
        "manifest": dict(product["manifest"]),
        "runtime_library": dict(product["runtime_library"]),
    }


def _validate_group(value: Any, *, schema: str) -> dict[str, Any]:
    group = _mapping(
        value,
        ("unique_name", "identity", "rank"),
        "group",
    )
    unique_name = group["unique_name"]
    if schema == LEGACY_VLLM_SCHEMA:
        if (
            not isinstance(unique_name, str)
            or not unique_name.startswith("tp:")
            or not unique_name[3:].isdigit()
        ):
            raise BootstrapError("only named tp:* communicator groups are accepted")
    elif not isinstance(unique_name, str) or re.fullmatch(
        r"[a-z][a-z0-9_]{0,31}:[a-zA-Z0-9_.-]{1,63}", unique_name
    ) is None:
        raise BootstrapError("communicator group name is not canonical")
    identity = _mapping(
        group["identity"],
        (
            "world_size",
            "epoch",
            "group_generation",
            "job_uuid",
            "group_uuid",
            "model_identity_sha256",
        ),
        "group.identity",
    )
    world = _integer(identity["world_size"], 2, 16, "group.identity.world_size")
    _integer(identity["epoch"], 1, (1 << 64) - 1, "group.identity.epoch")
    _integer(
        identity["group_generation"],
        1,
        (1 << 64) - 1,
        "group.identity.group_generation",
    )
    _hex(identity["job_uuid"], 16, "group.identity.job_uuid")
    _hex(identity["group_uuid"], 16, "group.identity.group_uuid")
    _hex(
        identity["model_identity_sha256"],
        32,
        "group.identity.model_identity_sha256",
    )
    rank = _mapping(
        group["rank"],
        (
            "rank",
            "capability_fd",
            "broker_pid",
            "broker_start_time_ticks",
            "join_timeout_ns",
            "collective_timeout_ns",
            "credits_per_peer",
        ),
        "group.rank",
    )
    rank_number = _integer(rank["rank"], 0, world - 1, "group.rank.rank")
    capability_fd = _integer(
        rank["capability_fd"], 0, (1 << 31) - 1, "group.rank.capability_fd"
    )
    try:
        metadata = os.fstat(capability_fd)
    except OSError as error:
        raise BootstrapError("group capability FD is unavailable") from error
    if not stat.S_ISSOCK(metadata.st_mode):
        raise BootstrapError("group capability FD is not a socket")
    _integer(rank["broker_pid"], 1, (1 << 31) - 1, "group.rank.broker_pid")
    _integer(
        rank["broker_start_time_ticks"],
        1,
        (1 << 64) - 1,
        "group.rank.broker_start_time_ticks",
    )
    _integer(rank["join_timeout_ns"], 1, (1 << 63) - 1, "group.rank.join_timeout_ns")
    _integer(
        rank["collective_timeout_ns"],
        1,
        (1 << 63) - 1,
        "group.rank.collective_timeout_ns",
    )
    _integer(rank["credits_per_peer"], 1, 16, "group.rank.credits_per_peer")
    return {
        "unique_name": unique_name,
        "identity": dict(identity),
        "rank": {**dict(rank), "rank": rank_number},
    }


def claim_group(
    unique_name: str,
    *,
    expected_rank: int,
    expected_world_size: int,
    expected_product_prefix: Path | None = None,
    descriptor_path: Path | None = None,
) -> dict[str, Any]:
    path = descriptor_path
    if path is None:
        value = os.environ.get("GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR")
        if value is None:
            raise BootstrapError("GEMSIM_CCL_BOOTSTRAP_DESCRIPTOR is required")
        path = _normal_absolute(value, "CCL bootstrap descriptor")
    path = Path(path)
    payload = _read_regular(path, private=True, maximum=_MAX_BYTES)
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("CCL bootstrap descriptor is invalid JSON") from error
    envelope = _mapping(document, ("schema", "product", "groups"), "bootstrap")
    schema = envelope["schema"]
    if schema not in (SCHEMA, LEGACY_VLLM_SCHEMA) or payload != _canonical_json(document):
        raise BootstrapError("CCL bootstrap descriptor is not canonical")
    product = _validate_product(envelope["product"])
    if expected_product_prefix is not None:
        prefix = Path(expected_product_prefix).resolve(strict=True)
        if prefix != product["prefix"]:
            raise BootstrapError("CCL bootstrap product is not the active product")
    if not isinstance(envelope["groups"], list) or not envelope["groups"]:
        raise BootstrapError("CCL bootstrap groups are missing")
    groups = [_validate_group(item, schema=schema) for item in envelope["groups"]]
    names = [item["unique_name"] for item in groups]
    fds = [item["rank"]["capability_fd"] for item in groups]
    if len(names) != len(set(names)) or len(fds) != len(set(fds)):
        raise BootstrapError("CCL bootstrap group names and capability FDs must be unique")
    matching = [item for item in groups if item["unique_name"] == unique_name]
    if len(matching) != 1:
        raise BootstrapError("requested communicator group is not present exactly once")
    selected = matching[0]
    if (
        selected["rank"]["rank"] != expected_rank
        or selected["identity"]["world_size"] != expected_world_size
    ):
        raise BootstrapError("CCL bootstrap rank/world differs from the control group")
    claim = (str(path.resolve(strict=True)), unique_name)
    with _claim_lock:
        if claim in _claimed_groups:
            raise BootstrapError("CCL bootstrap group was already consumed")
        _claimed_groups.add(claim)
    return {"product": product, "group": selected, "descriptor_path": path}


def build_engine(binding: Mapping[str, Any]):
    """Construct the shared device-backed all-reduce engine from one claim."""
    from .engine import AllReduceEngine, GroupSpec, RankBootstrap
    from .native import NativeCCL

    group = binding["group"]
    identity = group["identity"]
    rank = group["rank"]
    runtime_path = binding["product"]["runtime_library"]["path"]
    capability_fd = rank["capability_fd"]
    transferred = False
    try:
        native = NativeCCL(runtime_path)
        join_deadline = native.deadline_after(rank["join_timeout_ns"])
        group_spec = GroupSpec(
            world_size=identity["world_size"],
            epoch=identity["epoch"],
            group_generation=identity["group_generation"],
            job_uuid=bytes.fromhex(identity["job_uuid"]),
            group_uuid=bytes.fromhex(identity["group_uuid"]),
            model_identity_sha256=bytes.fromhex(
                identity["model_identity_sha256"]
            ),
        )
        bootstrap = RankBootstrap(
            rank=rank["rank"],
            capability_fd=capability_fd,
            broker_pid=rank["broker_pid"],
            broker_start_time_ticks=rank["broker_start_time_ticks"],
            absolute_deadline_ns=join_deadline,
            credits_per_peer=rank["credits_per_peer"],
        )
        # AllReduceEngine owns capability_fd on every path once join is called.
        transferred = True
        return AllReduceEngine.join(group_spec, bootstrap, native=native)
    finally:
        if not transferred:
            try:
                os.close(capability_fd)
            except OSError:
                pass


def _reset_claims_for_tests() -> None:
    with _claim_lock:
        _claimed_groups.clear()
