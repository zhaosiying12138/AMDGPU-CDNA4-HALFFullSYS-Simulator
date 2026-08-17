# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused tests for the AgentENV .wslconfig renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentenv_wslconfig", ROOT / "tools" / "agentenv_wslconfig.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AgentEnvWslConfigTest(unittest.TestCase):
    def test_renderer_uses_literal_double_backslashes(self) -> None:
        current = "[wsl2]\nmemory=64424509440\n"
        rendered = MODULE.render_candidate(
            current, "/mnt/c/Users/Admin1/wsl-kernels/agentenv-6.18.40.1"
        )
        self.assertIn(
            r"kernel=C:\\Users\\Admin1\\wsl-kernels\\agentenv-6.18.40.1\\bzImage",
            rendered.splitlines(),
        )
        self.assertIn(
            r"kernelModules=C:\\Users\\Admin1\\wsl-kernels\\agentenv-6.18.40.1\\modules.vhdx",
            rendered.splitlines(),
        )

    def test_renderer_replaces_only_managed_wsl2_keys(self) -> None:
        current = (
            "[wsl2]\n"
            "memory=1024\n"
            "kernel=C:\\stale\\bzImage\n"
            "nestedVirtualization=false\n"
            "[experimental]\n"
            "sparseVhd=true\n"
        )
        rendered = MODULE.render_candidate(current, "/mnt/d/kernels/agentenv")
        self.assertEqual(rendered.count("kernel="), 1)
        self.assertEqual(rendered.count("nestedVirtualization="), 1)
        self.assertIn("memory=1024", rendered)
        self.assertIn("[experimental]\nsparseVhd=true", rendered)

    def test_renderer_rejects_non_drvfs_stage(self) -> None:
        with self.assertRaisesRegex(MODULE.ConfigError, "/mnt/<drive>"):
            MODULE.render_candidate("[wsl2]\n", "/tmp/kernel")


if __name__ == "__main__":
    unittest.main()
