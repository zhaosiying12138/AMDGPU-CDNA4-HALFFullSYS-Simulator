# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-only contract tests for the in-sandbox Triton softmax smoke.

The smoke itself needs a simulated device, so these tests cover the parts that
can be checked without one: that the runner reproduces the library-ordering and
identity rules an accepted run depends on, and that the example is a real
oracle rather than a self-confirming print.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/sandbox_triton_smoke.sh"
EXAMPLE = ROOT / "examples/quickstart/triton_softmax.py"


class SandboxTritonSmokeRunnerTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.runner = RUNNER.read_text(encoding="utf-8")

    def test_runner_is_executable(self) -> None:
        self.assertTrue(RUNNER.stat().st_mode & 0o111, RUNNER)

    def _library_path(self) -> list[str]:
        for line in self.runner.splitlines():
            if line.startswith("export LD_LIBRARY_PATH="):
                return line.split("=", 1)[1].split(":")
        self.fail("the runner does not set LD_LIBRARY_PATH")

    def test_only_the_self_runtime_build_is_prepended(self) -> None:
        entries = self._library_path()
        self.assertEqual(entries[0], "$RT")
        self.assertNotIn("$PROD/lib", entries[:1])

    def test_product_lib_stays_behind_the_rocm_sysroot(self) -> None:
        """PyTorch's precompiled device code was built against ROCm 7.2.3.

        Moving the HEAD product's lib/ ahead of the sysroot makes HIP resolve to
        the product and invalidates every one of those code objects; both
        engines then die at the first device allocation with
        hipErrorInvalidImage. The ordering is the fix, so pin it.
        """

        entries = self._library_path()
        self.assertIn("$SYSROOT/lib", entries)
        self.assertIn("$PROD/lib", entries)
        self.assertLess(entries.index("$SYSROOT/lib"), entries.index("$PROD/lib"))

    def test_rocr_preload_is_prepended_not_assigned(self) -> None:
        self.assertIn(
            "export LD_PRELOAD=$PROD/lib/libhsa-runtime64.so.1${LD_PRELOAD:+:$LD_PRELOAD}",
            self.runner,
        )

    def test_identity_is_recorded_before_the_workload_runs(self) -> None:
        for key in (
            "managed_gem5_sha256",
            "rocr_sha256",
            "model_lib_sha256",
            "runtime_sha256",
        ):
            self.assertIn(f'echo "{key}=', self.runner, key)
        identity = self.runner.index("managed_gem5_sha256=")
        workload = self.runner.index("examples/quickstart/triton_softmax.py")
        self.assertLess(identity, workload)

    def test_identity_hashes_the_simulator_that_is_launched(self) -> None:
        """Hashing a fixed path certifies the wrong binary when the launcher is
        overridden, and SAGR_MANAGED_GEM5 is normally a host-loader shim rather
        than the simulator itself."""

        self.assertIn('sha256sum "$SAGR_MANAGED_GEM5"', self.runner)
        self.assertIn("gem5_elf_sha256=", self.runner)
        self.assertNotIn(
            'sha256sum "$R/projects/gem5/build/VEGA_X86/gem5.opt"', self.runner
        )

    def test_runner_selects_unchanged_upstream_triton(self) -> None:
        self.assertIn("unset TRITON_DEFAULT_BACKEND", self.runner)
        code = [
            line
            for line in self.runner.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertNotIn("gemsim_hip", "\n".join(code))

    def test_runner_fails_closed_without_a_retired_dispatch(self) -> None:
        self.assertIn("native_execution_retired", self.runner)
        self.assertIn("retired_dispatches=", self.runner)
        self.assertIn("not device evidence", self.runner)

    def test_runner_confines_its_simulator_to_a_private_run_root(self) -> None:
        self.assertIn("export SAGR_MANAGED_RUN_ROOT=", self.runner)


class SoftmaxExampleTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.source = EXAMPLE.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source, filename=str(EXAMPLE))

    def test_example_is_valid_python_with_a_jit_kernel(self) -> None:
        kernels = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(item, ast.Attribute) and item.attr == "jit"
                for item in node.decorator_list
            )
        ]
        self.assertEqual(kernels, ["softmax_kernel"])

    def test_reference_is_an_independent_wider_precision_oracle(self) -> None:
        self.assertIn("def reference_softmax", self.source)
        self.assertIn("torch.float64", self.source)
        # A device softmax compared against torch.softmax on the same tensor
        # library would hide a shared bug; the oracle recomputes it instead.
        self.assertNotIn("torch.softmax", self.source)
        self.assertNotIn("torch.nn.functional.softmax", self.source)

    def test_tolerances_are_explicit_and_tight(self) -> None:
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith("TOLERANCE")
        }
        self.assertEqual(
            sorted(constants), ["FLOAT32_RELATIVE_TOLERANCE", "ROW_SUM_TOLERANCE"]
        )
        for name, value in constants.items():
            self.assertLessEqual(value, 1e-6, name)

    def test_both_a_plain_and_a_masked_shape_are_covered(self) -> None:
        widths = sorted(
            ast.literal_eval(keyword.value)
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_case"
            for keyword in node.keywords
            if keyword.arg == "cols"
        )
        self.assertEqual(len(widths), 2, widths)
        powers = [width for width in widths if width & (width - 1) == 0]
        # One row width is a power of two (no masking) and one is not, so the
        # -inf masked tail is exercised and cannot poison the row maximum.
        self.assertEqual(len(powers), 1, f"expected one power-of-two width: {widths}")

    def test_example_refuses_a_project_triton_backend(self) -> None:
        self.assertIn('"triton.backends.amd.driver"', self.source)
        self.assertIn("unexpected upstream Triton driver", self.source)

    def test_example_checks_aliasing_and_input_immutability(self) -> None:
        for check in (
            "input_unchanged",
            "output_does_not_alias_input",
            "output_is_device_resident",
            "finite",
        ):
            self.assertIn(check, self.source, check)


if __name__ == "__main__":
    unittest.main()
