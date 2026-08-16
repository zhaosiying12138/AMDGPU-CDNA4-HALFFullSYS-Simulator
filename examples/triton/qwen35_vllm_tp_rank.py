#!/usr/bin/env python3
"""Run one real Qwen3.5 rank under the existing GemSim CCL supervisor.

This is a bring-up worker, not a second model implementation.  It only binds
the inherited product/bootstrap identity and then invokes the unchanged model
entrypoint with external tensor-parallel arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    runpy.run_path(str(ROOT / "examples/triton/_gemsim_bootstrap.py"))["bootstrap"](
        __file__, "qwen35-vllm-tp-rank"
    )


class WorkerError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def read_private(path: Path, maximum: int = 2 * 1024 * 1024) -> dict[str, Any]:
    path = path.resolve(strict=True)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise WorkerError("worker config is not a private regular file")
        payload = b""
        while len(payload) < metadata.st_size:
            chunk = os.read(descriptor, metadata.st_size - len(payload))
            if not chunk:
                raise WorkerError("worker config was truncated")
            payload += chunk
        if os.read(descriptor, 1):
            raise WorkerError("worker config changed while reading")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerError("worker config is invalid JSON") from error
    if not isinstance(value, dict) or payload != canonical_json(value):
        raise WorkerError("worker config is not canonical")
    return value


def socket_identity(descriptor: int) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISSOCK(metadata.st_mode):
        raise WorkerError("CCL capability is not a socket")
    return {
        "fd": descriptor,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "target": os.readlink(f"/proc/self/fd/{descriptor}"),
    }


def write_private(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_rank(config_path: Path, capability_fd: int) -> int:
    config = read_private(config_path)
    required = {
        "worker_mode", "model_script", "model_inference_mode", "model_stop_after_layer",
        "model_eager", "model_skip_oracle", "rank", "world_size", "dist_init_method",
        "result_path", "bootstrap_descriptor_path", "bootstrap_descriptor_sha256",
        "product",
    }
    if set(config) < required or config.get("worker_mode") != "qwen35-model":
        raise WorkerError("Qwen worker configuration is incomplete")
    rank = config["rank"]
    world = config["world_size"]
    if type(rank) is not int or type(world) is not int or not 2 <= world <= 16 or not 0 <= rank < world:
        raise WorkerError("invalid Qwen worker topology")
    if type(capability_fd) is not int or capability_fd < 0:
        raise WorkerError("invalid capability FD")
    inherited = socket_identity(capability_fd)
    descriptor = Path(config["bootstrap_descriptor_path"]).resolve(strict=True)
    descriptor_sha = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    if descriptor_sha != config["bootstrap_descriptor_sha256"]:
        raise WorkerError("bootstrap descriptor identity changed")
    debug_root = Path(config["result_path"]).resolve().parent / "qwen-debug"
    debug_root = debug_root.resolve()
    if debug_root.exists() or debug_root.is_symlink():
        raise WorkerError("Qwen debug output must be absent")
    debug_root.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(config["model_script"]),
        "--inference-mode", str(config["model_inference_mode"]),
        "--debug-output-dir", str(debug_root),
        "--debug-stop-after-layer", str(config["model_stop_after_layer"]),
        "--tensor-parallel-size", str(world),
        "--rank", str(rank),
        "--dist-init-method", str(config["dist_init_method"]),
    ]
    if bool(config["model_eager"]):
        argv.append("--eager")
    if bool(config["model_skip_oracle"]):
        argv.append("--skip-oracle")
    old_argv = sys.argv
    try:
        sys.argv = argv
        namespace = runpy.run_path(str(config["model_script"]), run_name="qwen35_model_rank")
        main = namespace.get("main")
        if not callable(main):
            raise WorkerError("Qwen model entrypoint does not expose main()")
        return_code = int(main())
    finally:
        sys.argv = old_argv
    result = {
        "schema": "amdgpu-sim.qwen35-vllm-tp-rank-result.v1",
        "status": "success" if return_code == 0 else "model_failure",
        "rank": rank,
        "world_size": world,
        "return_code": return_code,
        "capability_fd": inherited,
        "bootstrap_descriptor_sha256": descriptor_sha,
        "debug_output": str(debug_root),
        "model_script": str(Path(config["model_script"]).resolve(strict=True)),
        "product": config["product"],
    }
    write_private(Path(config["result_path"]), result)
    return return_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capability-fd", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        parsed = parse_args()
        raise SystemExit(run_rank(parsed.config, parsed.capability_fd))
    except (WorkerError, OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
