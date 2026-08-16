# SPDX-License-Identifier: GPL-3.0-or-later
"""Static gates for the upstream-zero-diff integration architecture."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


class LayeringContractTest(unittest.TestCase):
    def test_layering_migration_backup_is_complete_and_immutable(self) -> None:
        backup = (
            ROOT
            / "artifacts/source-backups"
            / "20260813T101500Z-layering-migration-baseline"
        )
        manifest = json.loads((backup / "manifest.json").read_text("utf-8"))
        self.assertEqual(
            manifest["schema"],
            "amdgpu-sim.layering-migration-source-backup.v1",
        )
        self.assertEqual(len(manifest["artifacts"]), 6)
        for artifact in manifest["artifacts"]:
            with self.subTest(path=artifact["path"]):
                relative = Path(artifact["path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                payload = (backup / relative).read_bytes()
                self.assertEqual(len(payload), artifact["bytes"])
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), artifact["sha256"]
                )

    def test_gem5_domain_core_is_protocol_free(self) -> None:
        directory = ROOT / "projects/gem5/src/dev/amdgpu"
        names = (
            "host_gpu_domain_types.hh",
            "host_gpu_memory_state.hh",
            "host_gpu_memory_state.cc",
            "host_native_memory_context.hh",
            "host_native_memory_context.cc",
            "host_gpu_code_object_image.hh",
            "host_gpu_code_object_image.cc",
            "host_gpu_code_object_state.hh",
            "host_gpu_code_object_state.cc",
            "host_native_code_object_mapper.hh",
            "host_native_code_object_mapper.cc",
            "host_gpu_native_command_processor_core.hh",
            "host_gpu_native_command_processor_core.cc",
            "host_gpu_native_dispatch_state_v2.hh",
            "host_gpu_native_dispatch_state_v2.cc",
        )
        forbidden = ("protocol::", "host_gpu_protocol.hh", "bridge_generic_v2")
        violations: list[str] = []
        for name in names:
            payload = (directory / name).read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in payload:
                    violations.append(f"{name}:{marker}")
        self.assertEqual(violations, [])

    def test_wire_domain_conversion_is_confined_to_bridge_adapter(self) -> None:
        directory = ROOT / "projects/gem5/src/dev/amdgpu"
        adapter = (directory / "host_gpu_protocol_adapter.cc").read_text(
            encoding="utf-8"
        )
        self.assertIn("namespace gem5::hostgpu::protocol_adapter", adapter)
        self.assertIn("protocol::", adapter)
        self.assertIn("domain::", adapter)

        violations: list[str] = []
        for path in sorted(directory.glob("*.cc")):
            if path.name == "host_gpu_protocol_adapter.cc" or \
                    path.name.endswith(".test.cc"):
                continue
            payload = path.read_text(encoding="utf-8")
            if "namespace gem5::hostgpu::protocol_adapter" in payload:
                violations.append(path.name)
        self.assertEqual(violations, [])

    def test_production_bridge_does_not_advertise_legacy_fixture(self) -> None:
        bridge = (
            ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_bridge.cc"
        ).read_text(encoding="utf-8")
        header = (
            ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_bridge.hh"
        ).read_text(encoding="utf-8")
        sconstruct = (
            ROOT / "projects/gem5/src/dev/amdgpu/SConscript"
        ).read_text(encoding="utf-8")
        advertised = (
            "identity.supported[protocol::PinnedDispatchCapabilityByte]"
        )
        self.assertNotIn(advertised, bridge)
        self.assertIn("protocol::GenericDispatchCapabilityMask", bridge)
        for legacy_marker in (
            "host_gpu_dispatch_fixture.hh",
            "dispatch_fixture::",
            "HostGPUDispatchState",
            "processDispatchRecord",
            "protocol::DispatchRequest",
            "protocol::DispatchResponse",
            "protocol::MessageType::DispatchRequest",
        ):
            self.assertNotIn(legacy_marker, bridge)
            self.assertNotIn(legacy_marker, header)
        self.assertNotIn(
            "Source('host_gpu_dispatch_state.cc', tags=['x86 isa'])",
            sconstruct,
        )
        self.assertIn("GTest('host_gpu_dispatch_state.test'", sconstruct)

    def test_native_memory_policy_covers_the_full_bound_lifecycle(self) -> None:
        bridge = (
            ROOT / "projects/gem5/src/dev/amdgpu/host_gpu_bridge.cc"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "!executionBound || requestId == 0 ||\n"
            "             requestId != executionRequestId",
            bridge,
        )
        self.assertIn(
            "if (executionBound && !packetRead && !kernargRead &&",
            bridge,
        )
        self.assertIn(
            "if (executionBound && !signalWrite && !allocationWrite && !segmentWrite)",
            bridge,
        )
        self.assertNotIn(
            "if (executionActive && !packetRead", bridge
        )
        self.assertNotIn(
            "if (executionActive && !allocationWrite", bridge
        )

    def test_pinned_framework_and_compiler_repositories_are_clean(self) -> None:
        for relative in ("projects/pytorch", "projects/triton", "projects/vllm"):
            repository = ROOT / relative
            with self.subTest(repository=relative):
                self.assertEqual(
                    git_output(
                        repository,
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    ),
                    "",
                )

    def test_triton_core_has_no_gemsim_coupling(self) -> None:
        roots = (
            ROOT / "projects/triton/python/triton/runtime",
            ROOT / "projects/triton/python/triton/compiler",
            ROOT / "projects/triton/python/triton/language",
            ROOT / "projects/triton/third_party/amd/backend",
        )
        forbidden = (b"gemsim", b"self_amdgpu", b"ROCM_SIM_ROOT", b"SAGR_")
        violations: list[str] = []
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                payload = path.read_bytes()
                if any(value.lower() in payload.lower() for value in forbidden):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_oot_backend_uses_standard_triton_base_classes(self) -> None:
        compiler_path = ROOT / "plugins/triton/gemsim_amd/backend/compiler.py"
        driver_path = ROOT / "plugins/triton/gemsim_amd/backend/driver.py"
        compiler = ast.parse(compiler_path.read_text(encoding="utf-8"))
        driver = ast.parse(driver_path.read_text(encoding="utf-8"))

        def bases(tree: ast.AST, class_name: str) -> set[str]:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return {ast.unparse(base) for base in node.bases}
            self.fail(f"missing class {class_name}")

        self.assertIn(
            "amd_compiler.HIPBackend", bases(compiler, "GemsimAMDBackend")
        )
        self.assertIn("DriverBase", bases(driver, "GemsimAMDDriver"))

    def test_framework_plugin_uses_official_entry_points(self) -> None:
        pyproject = (
            ROOT / "plugins/framework/gemsim_vllm/pyproject.toml"
        ).read_text(encoding="utf-8")
        self.assertIn('[project.entry-points."vllm.platform_plugins"]', pyproject)
        self.assertIn('[project.entry-points."vllm.general_plugins"]', pyproject)

        platform = ast.parse(
            (
                ROOT
                / "plugins/framework/gemsim_vllm/src/gemsim_vllm/platform.py"
            ).read_text(encoding="utf-8")
        )
        communicator = ast.parse(
            (
                ROOT
                / "plugins/framework/gemsim_vllm/src/gemsim_vllm/communicator.py"
            ).read_text(encoding="utf-8")
        )
        adapters = ast.parse(
            (
                ROOT
                / "plugins/framework/gemsim_vllm/src/gemsim_vllm/adapters.py"
            ).read_text(encoding="utf-8")
        )
        class_bases = {
            node.name: {ast.unparse(base) for base in node.bases}
            for tree in (platform, communicator)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("Platform", class_bases["GemsimPlatform"])
        self.assertIn(
            "DeviceCommunicatorBase", class_bases["GemsimDeviceCommunicator"]
        )
        decorators = {
            ast.unparse(decorator)
            for node in ast.walk(adapters)
            if isinstance(node, ast.ClassDef)
            for decorator in node.decorator_list
        }
        self.assertTrue(
            any("PluggableLayer.register_oot" in value for value in decorators)
        )
        self.assertTrue(any("CustomOp.register_oot" in value for value in decorators))

    def test_production_code_does_not_write_vllm_private_tp_or_methods(self) -> None:
        roots = tuple(
            ROOT / relative
            for relative in ("plugins", "examples", "scripts", "tools")
        )
        violations: list[str] = []
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    targets: list[ast.expr] = []
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        if isinstance(node, ast.Assign):
                            targets.extend(node.targets)
                        else:
                            targets.append(node.target)
                    for target in targets:
                        text = ast.unparse(target)
                        if text.endswith("._TP") or text in {
                            "GroupCoordinator.broadcast",
                            "GroupCoordinator.broadcast_tensor_dict",
                        }:
                            violations.append(
                                f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}"
                            )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
