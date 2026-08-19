from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_identity_gate", ROOT / "tools/run_identity_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RunIdentityGateTest(unittest.TestCase):
    def test_schema_is_versioned(self) -> None:
        self.assertEqual(MODULE.SCHEMA, "amdgpu-sim.run-identity-gate.v1")

    def test_tracked_sonames_cover_the_owned_boundary(self) -> None:
        # ROCr is the KMD-replacement boundary this project owns; HIP decides
        # whether a copy ever reaches the ROCr eligibility helper. Both must be
        # resolved, or a stale-binary run cannot be detected.
        self.assertIn("libhsa-runtime64.so.1", MODULE.TRACKED_SONAMES)
        self.assertIn("libamdhip64.so.7", MODULE.TRACKED_SONAMES)

    def test_describe_handles_absent_and_missing_paths(self) -> None:
        self.assertEqual(MODULE.describe(None), {"resolved": None})
        record = MODULE.describe("/nonexistent/definitely/not/here.so")
        self.assertEqual(record["resolved"], "/nonexistent/definitely/not/here.so")
        self.assertEqual(record.get("error"), "unresolvable")

    def test_describe_hashes_a_real_file(self) -> None:
        target = ROOT / "tools/run_identity_gate.py"
        record = MODULE.describe(str(target))
        self.assertEqual(record["realpath"], str(target.resolve()))
        self.assertEqual(record["bytes"], target.stat().st_size)
        self.assertEqual(len(record["sha256"]), 64)

    def test_resolve_loaded_returns_none_for_unknown_soname(self) -> None:
        self.assertIsNone(MODULE.resolve_loaded("libdefinitely-not-present-xyz.so.9"))

    def test_resolve_loaded_finds_a_library_the_loader_can_bind(self) -> None:
        # libc is always bindable, so this exercises the real dlopen + maps path
        # without depending on any ROCm artifact being installed.
        resolved = MODULE.resolve_loaded("libm.so.6")
        self.assertIsNotNone(resolved)
        self.assertTrue(str(resolved).startswith("/"))

    def test_resolver_child_strips_runtime_and_diagnostic_injection(self) -> None:
        injected = {
            "LD_PRELOAD": "/tmp/should-not-reach-child.so",
            "PYTHONPATH": "/tmp/diagnostic-sitecustomize",
            "SAGR_QWEN35_SGLANG_LAYER_GATE_OUTPUT": "/tmp/layer-gate",
            "SAGR_QWEN35_SGLANG_LAYER_GATE_GOLDEN": "/tmp/golden",
            "SAGR_QWEN35_OPERATOR_GOLDEN": "/tmp/operator-golden",
            "SAGR_TRITON_LAUNCH_LOG": "/tmp/triton.jsonl",
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/usr/lib/libm.so.6\n", stderr=""
        )
        with mock.patch.dict(os.environ, injected, clear=False):
            with mock.patch.object(
                MODULE.subprocess, "run", return_value=completed
            ) as run:
                self.assertEqual(
                    MODULE.resolve_loaded("libm.so.6"), "/usr/lib/libm.so.6"
                )
        child_environment = run.call_args.kwargs["env"]
        for name in injected:
            self.assertNotIn(name, child_environment)

    def test_gate_does_not_start_a_simulator(self) -> None:
        # The gate must never *call* hsa_init or a HIP entry point; doing so
        # would launch a managed gem5 and violate the one-gem5-per-TP1-lane
        # rule. Match calls, not prose -- the module docstring legitimately
        # mentions hsa_init to say it is not called.
        import ast

        tree = ast.parse((ROOT / "tools/run_identity_gate.py").read_text(encoding="ascii"))
        forbidden = {"hsa_init", "hipGetDeviceCount", "hipInit", "hsa_amd_memory_lock"}
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = getattr(function, "attr", None) or getattr(function, "id", None)
                if name:
                    called.add(name)
        self.assertEqual(called & forbidden, set())

        # The child program the gate executes must also stay dlopen-only.
        child = MODULE.resolve_loaded.__doc__ or ""
        self.assertNotIn("hsa_init", child)
        for name in forbidden:
            self.assertNotIn(f"{name}(", (ROOT / "tools/run_identity_gate.py").read_text(encoding="ascii"))

    def test_json_output_is_parseable_and_versioned(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_identity_gate.py"), "--format", "json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        record = json.loads(completed.stdout)
        self.assertEqual(record["schema"], MODULE.SCHEMA)
        for key in ("repo_head", "libraries", "fastcopy", "matches_active_product"):
            self.assertIn(key, record)

    def test_require_fastcopy_fails_when_gates_are_off(self) -> None:
        environment = dict(os.environ)
        environment["HSA_ENABLE_DTIF_FAST_COPY"] = "0"
        environment["SAGR_HSAKMT_MODEL_FAST_COPY"] = "0"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_identity_gate.py"), "--require-fastcopy"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            check=False, env=environment,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("fast copy is not enabled", completed.stderr)

    def test_require_fastcopy_passes_when_both_gates_are_one(self) -> None:
        environment = dict(os.environ)
        environment["HSA_ENABLE_DTIF_FAST_COPY"] = "1"
        environment["SAGR_HSAKMT_MODEL_FAST_COPY"] = "1"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/run_identity_gate.py"), "--require-fastcopy"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            check=False, env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_one_gate_alone_is_not_enough(self) -> None:
        for present, absent in (
            ("HSA_ENABLE_DTIF_FAST_COPY", "SAGR_HSAKMT_MODEL_FAST_COPY"),
            ("SAGR_HSAKMT_MODEL_FAST_COPY", "HSA_ENABLE_DTIF_FAST_COPY"),
        ):
            environment = dict(os.environ)
            environment[present] = "1"
            environment[absent] = "0"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_identity_gate.py"), "--require-fastcopy"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
                check=False, env=environment,
            )
            self.assertEqual(completed.returncode, 1, f"{present}=1 {absent}=0 should fail")

    def test_unknown_required_soname_is_rejected(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, str(ROOT / "tools/run_identity_gate.py"),
                "--require-active-product", "libnot-tracked.so.1",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("unknown soname", completed.stderr)


if __name__ == "__main__":
    unittest.main()
