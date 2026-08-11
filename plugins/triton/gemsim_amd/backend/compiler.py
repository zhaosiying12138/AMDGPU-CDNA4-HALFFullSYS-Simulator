from dataclasses import replace

from triton.backends.amd import compiler as amd_compiler


class GemsimAMDBackend(amd_compiler.HIPBackend):

    @staticmethod
    def supports_target(target):
        return target.backend == "gemsim_amd"

    def get_target_name(self, options):
        return f"gemsim_amd:{options.arch}"

    def parse_options(self, options):
        parsed = super().parse_options(dict(options))
        return replace(parsed, backend_name="gemsim_amd")
