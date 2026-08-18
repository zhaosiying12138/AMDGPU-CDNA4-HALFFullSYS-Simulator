#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/sim_amdgpu_arch.sh"


class SimAmdgpuArchTest(unittest.TestCase):
    def run_with_rocminfo(
        self, output: str, returncode: int = 0
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            rocminfo = Path(directory) / "rocminfo"
            rocminfo.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%b' {output!r}\n"
                f"exit {returncode}\n",
                encoding="ascii",
            )
            rocminfo.chmod(0o755)
            environment = dict(os.environ)
            environment["SAGR_SIM_ROCMINFO"] = str(rocminfo)
            return subprocess.run(
                [str(SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

    def test_reports_each_simulated_gpu_target(self) -> None:
        completed = self.run_with_rocminfo(
            "  Name: AMD Instinct MI350X\n"
            "      Name: gfx950\n"
            "      Name: gfx1201\n"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "gfx950\ngfx1201\n")

    def test_fails_closed_when_rocminfo_reports_no_target(self) -> None:
        completed = self.run_with_rocminfo("  Name: AMD Instinct MI350X\n")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")

    def test_propagates_rocminfo_failure(self) -> None:
        completed = self.run_with_rocminfo("      Name: gfx950\n", returncode=7)
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
