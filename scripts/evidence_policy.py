#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural policy shared by evidence producers and the resume verifier."""

from __future__ import annotations

import datetime as dt
from pathlib import PurePosixPath
import re
from typing import Any


ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidencePolicyError(RuntimeError):
    pass


def timestamp(value: Any, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidencePolicyError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise EvidencePolicyError(f"{label} has no timezone")
    return parsed


def argv(value: Any, label: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise EvidencePolicyError(f"{label} is not a non-empty string argv")


def external_descriptor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidencePolicyError(f"{label} is not an object")
    path = value.get("path")
    posix = PurePosixPath(path) if isinstance(path, str) else None
    if (
        posix is None
        or posix.is_absolute()
        or ".." in posix.parts
        or not path
        or not isinstance(value.get("size"), int)
        or isinstance(value.get("size"), bool)
        or value["size"] < 0
        or not isinstance(value.get("sha256"), str)
        or not SHA256_RE.fullmatch(value["sha256"])
        or not isinstance(value.get("required_at_acceptance"), bool)
        or not isinstance(value.get("required_for_resume"), bool)
        or value.get("tracked") is not False
    ):
        raise EvidencePolicyError(f"{label} has an invalid external artifact identity")
    return value


def validate_evidence(
    value: dict[str, Any], *, expected_id: str, checkpoint_id: str
) -> tuple[set[str], list[dict[str, Any]]]:
    if (
        value.get("schema") != "amdgpu-sim.evidence.v1"
        or value.get("id") != expected_id
        or value.get("checkpoint_id") != checkpoint_id
        or not isinstance(value.get("type"), str)
        or not value["type"]
        or not isinstance(value.get("claim"), str)
        or not value["claim"]
        or not isinstance(value.get("cwd"), str)
        or not value["cwd"]
        or not isinstance(value.get("exit_code"), int)
        or isinstance(value.get("exit_code"), bool)
    ):
        raise EvidencePolicyError(f"evidence identity is incomplete: {expected_id}")
    started = timestamp(value.get("started_at"), f"{expected_id}.started_at")
    ended = timestamp(value.get("ended_at"), f"{expected_id}.ended_at")
    if ended < started:
        raise EvidencePolicyError(f"evidence time range is reversed: {expected_id}")
    commands = value.get("command_argv")
    if not isinstance(commands, list) or not commands:
        raise EvidencePolicyError(f"evidence has no command argv: {expected_id}")
    for index, command in enumerate(commands):
        argv(command, f"{expected_id}.command_argv[{index}]")

    results = value.get("command_results")
    if not isinstance(results, list) or not results:
        raise EvidencePolicyError(f"evidence has no command results: {expected_id}")
    result_ids: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    for index, result in enumerate(results):
        label = f"{expected_id}.command_results[{index}]"
        if not isinstance(result, dict):
            raise EvidencePolicyError(f"{label} is not an object")
        result_id = result.get("id")
        if (
            not isinstance(result_id, str)
            or not ID_RE.fullmatch(result_id)
            or result_id in result_ids
            or not isinstance(result.get("cwd"), str)
            or not result["cwd"]
            or not isinstance(result.get("exit_code"), int)
            or isinstance(result.get("exit_code"), bool)
            or not isinstance(result.get("result"), str)
            or not result["result"]
        ):
            raise EvidencePolicyError(f"{label} has an invalid command identity")
        argv(result.get("argv"), f"{label}.argv")
        result_ids.add(result_id)
        retained = result.get("raw_streams_retained")
        if retained is True:
            command_started = timestamp(result.get("started_at"), f"{label}.started_at")
            command_ended = timestamp(result.get("ended_at"), f"{label}.ended_at")
            if command_ended < command_started:
                raise EvidencePolicyError(f"{label} has a reversed time range")
            for key in ("record", "stdout", "stderr"):
                descriptor = external_descriptor(result.get(key), f"{label}.{key}")
                if descriptor["path"] in artifact_paths:
                    raise EvidencePolicyError(
                        f"duplicate evidence artifact path: {descriptor['path']}"
                    )
                artifact_paths.add(descriptor["path"])
                artifacts.append(descriptor)
        elif retained is False:
            if (
                result.get("stdout_sha256") is not None
                or result.get("stderr_sha256") is not None
                or not isinstance(result.get("raw_streams_limitation"), str)
                or not result["raw_streams_limitation"]
            ):
                raise EvidencePolicyError(f"{label} has an invalid unretained-stream record")
        else:
            raise EvidencePolicyError(f"{label} does not declare raw stream retention")

    external = value.get("external_artifacts", [])
    if not isinstance(external, list):
        raise EvidencePolicyError(f"external_artifacts is not a list: {expected_id}")
    for index, item in enumerate(external):
        descriptor = external_descriptor(item, f"{expected_id}.external_artifacts[{index}]")
        if descriptor["path"] in artifact_paths:
            raise EvidencePolicyError(f"duplicate evidence artifact path: {descriptor['path']}")
        artifact_paths.add(descriptor["path"])
        artifacts.append(descriptor)
    return result_ids, artifacts
