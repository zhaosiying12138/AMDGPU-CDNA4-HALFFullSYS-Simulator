# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused tests for the AgentENV WSL kernel artifact tool."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_agentenv_wsl_kernel.sh"
RELEASE = "6.18.40.1-aenv-test"


@unittest.skipUnless(
    shutil.which("mke2fs") and shutil.which("qemu-img") and shutil.which("debugfs"),
    "mke2fs, qemu-img, and debugfs are required",
)
class AgentEnvKernelToolTest(unittest.TestCase):
    def test_pack_modules_builds_dual_compatible_vhd(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "kernel-out"
            module_root = output / "modules" / "lib" / "modules" / RELEASE
            driver = module_root / "kernel" / "drivers" / "block" / "ublk_drv.ko"
            driver.parent.mkdir(parents=True)
            driver.write_bytes(b"test module\n")
            (module_root / "modules.dep").write_text(
                "kernel/drivers/block/ublk_drv.ko:\n", encoding="utf-8"
            )
            (module_root / "build").symlink_to("/nonexistent/build-host")
            (output / "headers" / "include").mkdir(parents=True)
            (output / "headers" / "include" / "test.h").write_text(
                "/* test */\n", encoding="utf-8"
            )
            (output / "perf" / "bin").mkdir(parents=True)
            (output / "perf" / "bin" / "perf").write_text(
                "test perf\n", encoding="utf-8"
            )
            (output / "kernelrelease").write_text(RELEASE + "\n", encoding="utf-8")

            env = os.environ.copy()
            env["AGENTENV_KERNEL_OUT"] = str(output)
            subprocess.run(
                [str(TOOL), "pack-modules"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            raw = Path(temp) / "modules.raw"
            subprocess.run(
                [shutil.which("qemu-img"), "convert", "-O", "raw",
                 str(output / "modules.vhdx"), str(raw)],
                check=True,
                capture_output=True,
                text=True,
            )
            root_listing = subprocess.run(
                [shutil.which("debugfs"), "-R", "ls -p /", str(raw)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn(f"/{RELEASE}/", root_listing)
            self.assertIn("/modules.dep/", root_listing)
            self.assertIn("/kernel/", root_listing)
            self.assertNotIn("/build/", root_listing)
            for name in ("modules.dep", "kernel"):
                link_stat = subprocess.run(
                    [shutil.which("debugfs"), "-R", f"stat /{name}", str(raw)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertIn(
                    f'Fast link dest: "{RELEASE}/modules/{name}"', link_stat
                )
            driver_stat = subprocess.run(
                [shutil.which("debugfs"), "-R",
                 f"stat /{RELEASE}/modules/kernel/drivers/block/ublk_drv.ko",
                 str(raw)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("Size: 12", driver_stat)
            for path in (
                f"/{RELEASE}/modules/modules.dep",
                f"/{RELEASE}/linux-headers/include/test.h",
                f"/{RELEASE}/perf/bin/perf",
            ):
                result = subprocess.run(
                    [shutil.which("debugfs"), "-R", f"stat {path}", str(raw)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, (path, result.stderr))


if __name__ == "__main__":
    unittest.main()
