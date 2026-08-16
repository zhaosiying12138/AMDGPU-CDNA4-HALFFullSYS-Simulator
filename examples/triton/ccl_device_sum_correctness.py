#!/usr/bin/env python3

from __future__ import annotations

__import__("runpy").run_path(
    __file__.replace("ccl_device_sum_correctness.py", "_gemsim_bootstrap.py")
)["bootstrap"](__file__, "ccl-device-sum")

import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import triton

from gemsim_ccl import DeviceSumExecutor


DEVICE = triton.runtime.driver.active.get_active_torch_device()
PREFIX_ELEMENTS = 17
SUFFIX_ELEMENTS = 19


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def managed_session_record() -> dict[str, object]:
    driver = triton.runtime.driver.active
    runtime = getattr(driver, "runtime", None)
    info = getattr(runtime, "session_info", None)
    if runtime is None or info is None or not runtime.session.value:
        raise RuntimeError("managed simulator session is unavailable")
    child_pid = int(info.child_pid)
    proc = Path("/proc") / str(child_pid)
    command = (proc / "cmdline").read_bytes().split(b"\0")
    argv = [item.decode("utf-8", "strict") for item in command if item]
    if len(argv) < 5 or argv[1] != "--listener-mode=on" or argv[2] != "--outdir":
        raise RuntimeError("managed gem5 command line is invalid")

    def option(name: str) -> str:
        if argv.count(name) != 1:
            raise RuntimeError(f"managed gem5 option {name} is not unique")
        index = argv.index(name)
        if index + 1 >= len(argv):
            raise RuntimeError(f"managed gem5 option {name} has no value")
        return argv[index + 1]

    output_directory = Path(option("--outdir"))
    trace_path = Path(option("--dispatch-trace-path"))
    config_path = Path(argv[4])
    gem5_path = Path(os.readlink(proc / "exe"))
    for path in (gem5_path, config_path, output_directory, trace_path):
        if not path.is_absolute() or path != Path(os.path.normpath(path)):
            raise RuntimeError("managed gem5 command contains a noncanonical path")
    run_directory = trace_path.parent
    if (
        output_directory.parent != run_directory
        or trace_path.name != "dispatch-trace.jsonl"
        or output_directory.name != "m5out"
    ):
        raise RuntimeError("managed gem5 artifact paths do not share one run directory")
    return {
        "child_pid": child_pid,
        "connection_id": int(info.connection_id),
        "epoch": int(info.epoch),
        "rank": int(info.rank),
        "world_size": int(info.world_size),
        "daemon_uuid": bytes(info.daemon_uuid).hex(),
        "job_uuid": bytes(info.job_uuid).hex(),
        "gem5_path": str(gem5_path),
        "config_path": str(config_path),
        "run_directory": str(run_directory),
        "output_directory": str(output_directory),
        "trace_path": str(trace_path),
        "stats_path": str(output_directory / "stats.txt"),
        "log_path": str(run_directory / "gem5.log"),
        "command_sha256": hashlib.sha256(b"\0".join(command)).hexdigest(),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "runtime_library": str(Path(runtime.path).resolve(strict=True)),
        "prefix": os.environ["ROCM_SIM_ROOT"],
        "triton_cache_directory": os.environ["TRITON_CACHE_DIR"],
    }


def deterministic_values(count: int, dtype: torch.dtype, phase: int) -> torch.Tensor:
    indices = torch.arange(count, dtype=torch.int64, device=DEVICE)
    numerators = ((indices * (phase * 2 + 3) + phase * 11) % 97) - 48
    values = numerators.to(torch.float32) / float(1 << (phase % 4 + 3))
    return values.to(dtype)


def guarded_tensor(
    values: torch.Tensor,
    *,
    prefix_value: float,
    suffix_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    storage = torch.empty(
        PREFIX_ELEMENTS + values.numel() + SUFFIX_ELEMENTS,
        dtype=values.dtype,
        device=DEVICE,
    )
    storage[:PREFIX_ELEMENTS].fill_(prefix_value)
    storage[PREFIX_ELEMENTS:-SUFFIX_ELEMENTS].copy_(values)
    storage[-SUFFIX_ELEMENTS:].fill_(suffix_value)
    view = storage[PREFIX_ELEMENTS:-SUFFIX_ELEMENTS]
    return storage, view, storage[:PREFIX_ELEMENTS].clone(), storage[-SUFFIX_ELEMENTS:].clone()


def compare_exact(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(actual.to(torch.float32)).all().item())
    raw_dtype = torch.uint16 if actual.dtype == torch.bfloat16 else torch.int32
    mismatch = int(torch.count_nonzero(actual.view(raw_dtype) != expected.view(raw_dtype)).item())
    maximum = float(
        torch.max(torch.abs(actual.to(torch.float32) - expected.to(torch.float32))).item()
    ) if actual.numel() else 0.0
    return {
        "correct": finite and mismatch == 0,
        "finite": finite,
        "mismatch_count": mismatch,
        "max_abs_error": maximum,
        "actual_sha256": tensor_sha256(actual),
        "expected_sha256": tensor_sha256(expected),
    }


def run_case(
    executor: DeviceSumExecutor,
    *,
    name: str,
    dtype: torch.dtype,
    extent: int,
    element_count: int,
) -> dict[str, object]:
    if extent == 0:
        left = torch.empty(0, dtype=dtype, device=DEVICE)
        right = torch.empty(0, dtype=dtype, device=DEVICE)
        left_storage = left
        right_storage = right
        left_prefix = left.clone()
        left_suffix = left.clone()
        right_prefix = right.clone()
        right_suffix = right.clone()
    else:
        left_values = deterministic_values(extent, dtype, 1)
        right_values = deterministic_values(extent, dtype, 2)
        if dtype == torch.bfloat16 and extent >= 4:
            left_values[:4] = torch.tensor(
                [1.0, 1.0078125, -1.0, -1.0078125], dtype=dtype, device=DEVICE
            )
            right_values[:4] = torch.tensor(
                [0.00390625, 0.00390625, -0.00390625, -0.00390625],
                dtype=dtype,
                device=DEVICE,
            )
        if dtype == torch.float32 and extent >= 8:
            left_values[:8] = torch.tensor(
                [
                    1.0,
                    16777216.0,
                    -16777216.0,
                    65536.25,
                    -65536.25,
                    -0.0,
                    torch.finfo(torch.float32).tiny,
                    -torch.finfo(torch.float32).tiny,
                ],
                dtype=dtype,
                device=DEVICE,
            )
            right_values[:8] = torch.tensor(
                [
                    2.0**-24,
                    1.0,
                    -1.0,
                    -65536.0,
                    65536.0,
                    0.0,
                    torch.finfo(torch.float32).tiny,
                    -torch.finfo(torch.float32).tiny,
                ],
                dtype=dtype,
                device=DEVICE,
            )
        left_storage, left, left_prefix, left_suffix = guarded_tensor(
            left_values, prefix_value=-13.0, suffix_value=11.0
        )
        right_storage, right, right_prefix, right_suffix = guarded_tensor(
            right_values, prefix_value=7.0, suffix_value=-9.0
        )
    left_before = left.clone()
    right_before = right.clone()

    output_storage = left_storage
    output = executor.sum_in_place(left, right, element_count=element_count)
    output_prefix = left_prefix
    output_suffix = left_suffix

    expected = (
        left_before[:element_count].to(torch.float32)
        + right_before[:element_count].to(torch.float32)
    ).to(dtype)
    comparison = compare_exact(output[:element_count], expected)
    tie_contract = None
    if dtype == torch.bfloat16 and element_count >= 4:
        actual_bits = [int(value) for value in output[:4].view(torch.uint16).tolist()]
        expected_bits = [0x3F80, 0x3F82, 0xBF80, 0xBF82]
        tie_contract = {
            "inputs": [
                [1.0, 0.00390625],
                [1.0078125, 0.00390625],
                [-1.0, -0.00390625],
                [-1.0078125, -0.00390625],
            ],
            "actual_bits": [f"0x{value:04x}" for value in actual_bits],
            "expected_bits": [f"0x{value:04x}" for value in expected_bits],
            "rounding": "round_to_nearest_even",
            "correct": actual_bits == expected_bits,
        }
    tail_unchanged = bool(torch.equal(output[element_count:], left_before[element_count:]))
    right_unchanged = bool(torch.equal(right, right_before))
    output_aliases_destination = bool(
        output is left
        and output.data_ptr() == left.data_ptr()
        and output.storage_offset() == left.storage_offset()
        and output.shape == left.shape
        and output.stride() == left.stride()
    )
    guards_unchanged = bool(
        torch.equal(output_storage[:PREFIX_ELEMENTS], output_prefix)
        and torch.equal(output_storage[-SUFFIX_ELEMENTS:], output_suffix)
        and torch.equal(right_storage[:PREFIX_ELEMENTS], right_prefix)
        and torch.equal(right_storage[-SUFFIX_ELEMENTS:], right_suffix)
    )
    correct = bool(
        comparison["correct"]
        and tail_unchanged
        and right_unchanged
        and output_aliases_destination
        and guards_unchanged
        and (tie_contract is None or tie_contract["correct"])
    )
    return {
        "name": name,
        "dtype": str(dtype).removeprefix("torch."),
        "extent": extent,
        "element_count": element_count,
        "mode": "in_place",
        "program_count": triton.cdiv(element_count, 256) if element_count else 0,
        "comparison": comparison,
        "bf16_tie_contract": tie_contract,
        "tail_unchanged": tail_unchanged,
        "right_unchanged": right_unchanged,
        "output_aliases_destination": output_aliases_destination,
        "guards_unchanged": guards_unchanged,
        "output_correct": correct,
    }


def negative_contracts(executor: DeviceSumExecutor) -> dict[str, bool]:
    good = torch.zeros(8, dtype=torch.float32, device=DEVICE)
    other = torch.ones(8, dtype=torch.float32, device=DEVICE)
    backing = bytearray(64)
    overlap_left = torch.frombuffer(backing, dtype=torch.float32, count=8, offset=0)
    overlap_right = torch.frombuffer(backing, dtype=torch.float32, count=8, offset=4)
    empty = torch.empty(0, dtype=torch.float32, device=DEVICE)

    def rejected(action) -> bool:
        try:
            action()
        except (TypeError, ValueError):
            return True
        return False

    def zero_alias_permitted() -> bool:
        before = executor.counters.device_reduction_launch_count
        output = executor.sum_in_place(empty, empty, element_count=0)
        return output is empty and executor.counters.device_reduction_launch_count == before

    return {
        "unsupported_dtype_rejected": rejected(
            lambda: executor.sum_in_place(good.to(torch.float64), other.to(torch.float64))
        ),
        "mismatched_dtype_rejected": rejected(
            lambda: executor.sum_in_place(good, other.to(torch.bfloat16))
        ),
        "noncontiguous_tensor_rejected": rejected(
            lambda: executor.sum_in_place(good, torch.ones(16)[::2])
        ),
        "multidimensional_tensor_rejected": rejected(
            lambda: executor.sum_in_place(torch.zeros((2, 4)), torch.ones((2, 4)))
        ),
        "source_alias_rejected": rejected(lambda: executor.sum_in_place(good, good)),
        "overlapping_storage_rejected": rejected(
            lambda: executor.sum_in_place(overlap_left, overlap_right)
        ),
        "boolean_count_rejected": rejected(
            lambda: executor.sum_in_place(good, other, element_count=True)
        ),
        "negative_count_rejected": rejected(
            lambda: executor.sum_in_place(good, other, element_count=-1)
        ),
        "oversized_count_rejected": rejected(
            lambda: executor.sum_in_place(good, other, element_count=9)
        ),
        "shape_mismatch_rejected": rejected(
            lambda: executor.sum_in_place(good, other[:-1])
        ),
        "zero_count_alias_permitted": zero_alias_permitted(),
    }


def main() -> int:
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "gemsim_amd" or target.arch != "gfx950":
        raise RuntimeError(f"unexpected Triton execution target: {target}")
    if DEVICE.type != "cpu":
        raise RuntimeError(f"gemsim_amd must expose CPU staging, got {DEVICE}")

    executor = DeviceSumExecutor()
    cases = []
    counts = (0, 1, 3, 127, 128, 129, 255, 256, 257, 1024, 1027, 2048, 7168)
    for dtype in (torch.bfloat16, torch.float32):
        dtype_name = str(dtype).removeprefix("torch.")
        for count in counts:
            cases.append(
                run_case(
                    executor,
                    name=f"{dtype_name}_in_place_{count}",
                    dtype=dtype,
                    extent=0 if count == 0 else max(8, count + 4),
                    element_count=count,
                )
            )
    negative = negative_contracts(executor)
    counters = executor.counters
    session = managed_session_record()
    correct = bool(
        all(case["output_correct"] for case in cases)
        and all(negative.values())
        and counters.device_reduction_launch_count == 24
        and counters.host_reduction_count == 0
    )
    payload = {
        "schema": "amdgpu-sim.ccl-device-sum.v1",
        "backend": target.backend,
        "arch": target.arch,
        "operation": "sum",
        "supported_dtypes": ["bfloat16", "float32"],
        "binary_compute_dtype": "float32",
        "persistent_workspace_dtype": "descriptor_dtype",
        "accumulation_mode": "pairwise_then_store_per_executor_invocation",
        "workspace_rounding": {
            "bfloat16": "round_to_nearest_even_per_executor_invocation",
            "float32": "binary_float32_add_per_executor_invocation",
            "round_after_each_invocation": True,
        },
        "block_size": 256,
        "cases": cases,
        "negative_contracts": negative,
        "device_reduction_launch_count": counters.device_reduction_launch_count,
        "self_reported_counters": {
            "host_reduction_count": counters.host_reduction_count,
            "fallback_count": 0,
            "cpu_fallback_count": 0,
            "nvidia_fallback_count": 0,
            "acceptance_authority": False,
        },
        "oracle": {
            "role": "post_target_external_host_check",
            "executed_after_target": True,
            "feedback_to_target": False,
        },
        "managed_session": session,
        "output_correct": correct,
        "claim_scope": "standalone_device_sum_primitive",
        "planner_binding_accepted": False,
        "trace_evidence_bound": False,
        "live_collective_accepted": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
