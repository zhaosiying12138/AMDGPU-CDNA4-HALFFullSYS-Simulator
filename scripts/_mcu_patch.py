#!/usr/bin/env python3
"""Apply the multi-compute-unit wiring to a copy of host_dispatch.py.

Textual, idempotent, and fails loudly if any anchor is missing, so the edit
cannot half-apply to the config every lane reads at start-up.
"""
import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()


def swap(old: str, new: str) -> None:
    global text
    if text.count(old) != 1:
        raise SystemExit(
            f"anchor not unique ({text.count(old)}): {old[:80]!r}")
    text = text.replace(old, new)


# ---------------------------------------------------------------- 1. header
swap(
    'LDS_BYTES_PER_CU = int(os.environ.get("GEMSIM_LDS_BYTES_PER_CU", 160 * 1024))\n',
    'LDS_BYTES_PER_CU = int(os.environ.get("GEMSIM_LDS_BYTES_PER_CU", 160 * 1024))\n'
    '\n'
    '# Compute units the simulator provides.\n'
    '#\n'
    '# The topology this stack publishes advertises a full gfx950: simd_count\n'
    '# 1024 over simd_per_cu 4, i.e. 256 compute units, each with\n'
    '# max_waves_per_simd 8. gem5 built exactly one. That is not merely a slower\n'
    '# machine, it is a different one: a kernel whose forward progress depends on\n'
    '# several workgroups running at the same time cannot complete on it.\n'
    '#\n'
    "# AITER's split-K GEMM is such a kernel. One workgroup publishes a tile\n"
    '# counter and its peers poll it. With too few resident workgroups every\n'
    '# resident peer polls a flag whose producer was never scheduled, and the run\n'
    '# livelocks: scalar instructions retire forever, no wavefront completes, and\n'
    '# the host waits on a completion signal that can never arrive. This is the\n'
    '# same shape as the 64 KiB-versus-160 KiB group-segment defect -- the device\n'
    '# claiming a capability the simulator does not honour -- and it fails the\n'
    '# same silent way.\n'
    '#\n'
    "# The count is a knob because simulation cost scales with it and 256 is not\n"
    '# affordable; see docs/ and the capsule in tools/coop_capsule/ for the\n'
    '# measured cost per count. Whatever value is chosen below the advertised 256,\n'
    '# the mismatch is real: a kernel needing more co-resident workgroups than\n'
    '# NUM_COMPUTE_UNITS * (workgroups that fit in one CU) can still deadlock.\n'
    'NUM_COMPUTE_UNITS = int(os.environ.get("GEMSIM_NUM_COMPUTE_UNITS", 16))\n'
    'if NUM_COMPUTE_UNITS < 1:\n'
    '    raise SystemExit("GEMSIM_NUM_COMPUTE_UNITS must be >= 1")\n',
)

# ------------------------------------------------------- 2. per-CU factory
swap(
    'def create_compute_unit(shader, functional_fast=False):\n'
    '    cu = ComputeUnit(\n'
    '        cu_id=0,\n',
    'def create_compute_unit(shader, cu_id, functional_fast=False):\n'
    '    cu = ComputeUnit(\n'
    '        cu_id=cu_id,\n',
)

# --------------------------------------------------- 3. topology bookkeeping
swap(
    """    # The count is left configurable because simulation cost scales with it.
    args.num_compute_units = int(
        os.environ.get("GEMSIM_NUM_COMPUTE_UNITS", "1")
    )
    args.cu_per_sqc = args.num_compute_units
    args.cu_per_scalar_cache = args.num_compute_units
    args.num_sqc = 1
    args.num_scalar_cache = 1
""",
    """    # The count is left configurable because simulation cost scales with it.
    args.num_compute_units = NUM_COMPUTE_UNITS
    # One scalar/instruction cache shared by every compute unit. GPU_VIPER
    # builds ceil(num_compute_units / cu_per_sqc) SQC controllers and the same
    # again for the scalar cache, and each is a Ruby controller, a sequencer, a
    # TLB and a TLB coalescer. Sharing costs nothing architecturally here:
    # host-native instruction fetch never reaches the SQC (FetchUnit returns
    # from the native provider) and host-native scalar loads never reach the
    # scalar cache (ComputeUnit::sendScalarRequest services them functionally).
    # The only traffic either one carries is the per-kernel launch invalidate,
    # which Shader::prepareInvalidate issues once per SQC group.
    args.cu_per_sqc = args.num_compute_units
    args.cu_per_scalar_cache = args.num_compute_units
    args.num_sqc = 1
    args.num_scalar_cache = 1
""",
)

# ------------------------------------------------------------ 4. build CUs
swap(
    """    # One CU is built and wired below. Raising GEMSIM_NUM_COMPUTE_UNITS is
    # not just a matter of building more of them: the Ruby port wiring further
    # down binds a single VIPER coalescer, SQC port and scalar port by index,
    # and each of those becomes per-unit. See the fatal there.
    compute_unit = create_compute_unit(shader, args.functional_fast)
    shader.CUs = [compute_unit]
""",
    """    # Shader::Shader asserts cuList[i]->cu_id == i, and GPU_VIPER builds one
    # TCP (vector L1 + VIPER coalescer) per unit, so the identifiers have to be
    # dense and in order. The Ruby ports are bound after Ruby.create_system.
    compute_units = [
        create_compute_unit(shader, cu_id, args.functional_fast)
        for cu_id in range(args.num_compute_units)
    ]
    shader.CUs = compute_units
""",
)

# ----------------------------------------------------------- 5. port wiring
swap(
    """    gpu_port_idx = len(system.ruby._cpu_ports) - 3
    token_ports = [
        port
        for port in system.ruby._cpu_ports
        if isinstance(port, VIPERCoalescer)
    ]
    if len(token_ports) != args.num_compute_units:
        fatal(
            "expected one VIPER coalescer per compute unit: %d units but %d "
            "coalescers. Raising GEMSIM_NUM_COMPUTE_UNITS above 1 needs the "
            "port wiring below to be looped over the units as well."
            % (args.num_compute_units, len(token_ports))
        )
    compute_unit.gmTokenPort = token_ports[0].gmTokenPort
    for lane in range(64):
        compute_unit.memory_port[lane] = system.ruby._cpu_ports[
            gpu_port_idx
        ].in_ports[lane]
    gpu_port_idx += 1
    compute_unit.sqc_port = system.ruby._cpu_ports[gpu_port_idx].in_ports
    gpu_port_idx += 1
    compute_unit.scalar_port = system.ruby._cpu_ports[gpu_port_idx].in_ports
""",
    """    # Ruby.create_system publishes its sequencers in construction order:
    # the CPU core pair first, then one VIPER coalescer per compute unit, then
    # the SQC sequencers, then the scalar-cache sequencers, then any command
    # processors (num_cp is 0 here). Derive the first GPU index from that
    # composition rather than from a constant, and assert the composition.
    gpu_port_idx = (
        len(system.ruby._cpu_ports)
        - args.num_compute_units
        - args.num_sqc
        - args.num_scalar_cache
        - args.num_cp * 2
    )
    token_ports = [
        port
        for port in system.ruby._cpu_ports
        if isinstance(port, VIPERCoalescer)
    ]
    if len(token_ports) != args.num_compute_units:
        fatal(
            "expected one VIPER coalescer per compute unit: %d units but %d "
            "coalescers"
            % (args.num_compute_units, len(token_ports))
        )
    if gpu_port_idx < 0 or token_ports[0] is not (
        system.ruby._cpu_ports[gpu_port_idx]
    ):
        fatal(
            "GPU_VIPER sequencer order changed: expected the first VIPER "
            "coalescer at index %d of %d cpu ports"
            % (gpu_port_idx, len(system.ruby._cpu_ports))
        )

    for cu_id, compute_unit in enumerate(compute_units):
        # Only the TCP coalescers use a token port for back pressure.
        compute_unit.gmTokenPort = token_ports[cu_id].gmTokenPort

    # One vector L1 per compute unit; the pipeline issues wf_size uncoalesced
    # requests per issue cycle, hence 64 memory ports each.
    for compute_unit in compute_units:
        for lane in range(64):
            compute_unit.memory_port[lane] = system.ruby._cpu_ports[
                gpu_port_idx
            ].in_ports[lane]
        gpu_port_idx += 1

    # cu_per_sqc compute units share one SQC, and likewise for the scalar
    # cache: advance to the next sequencer only on a group boundary.
    for cu_id, compute_unit in enumerate(compute_units):
        if cu_id > 0 and not cu_id % args.cu_per_sqc:
            gpu_port_idx += 1
        compute_unit.sqc_port = system.ruby._cpu_ports[gpu_port_idx].in_ports
    gpu_port_idx += 1
    for cu_id, compute_unit in enumerate(compute_units):
        if cu_id > 0 and not cu_id % args.cu_per_scalar_cache:
            gpu_port_idx += 1
        compute_unit.scalar_port = system.ruby._cpu_ports[
            gpu_port_idx
        ].in_ports
    gpu_port_idx += 1
    if gpu_port_idx != len(system.ruby._cpu_ports) - args.num_cp * 2:
        fatal(
            "GPU port wiring consumed %d of %d ruby cpu ports"
            % (gpu_port_idx, len(system.ruby._cpu_ports))
        )
""",
)

# ------------------------------------------------------- 6. identity header
swap(
    '''    "host-gpu-dispatch-mode timing_fidelity={} functional_fast={} "
    "lds_bytes_per_cu={}".format(
        "approximate" if args.functional_fast else "cycle-accurate",
        int(args.functional_fast),
        LDS_BYTES_PER_CU,
    ),''',
    '''    "host-gpu-dispatch-mode timing_fidelity={} functional_fast={} "
    "lds_bytes_per_cu={} compute_units={} wave_slots_per_simd={}".format(
        "approximate" if args.functional_fast else "cycle-accurate",
        int(args.functional_fast),
        LDS_BYTES_PER_CU,
        args.num_compute_units,
        8,
    ),''',
)

target.write_text(text)
print(f"patched {target}")
