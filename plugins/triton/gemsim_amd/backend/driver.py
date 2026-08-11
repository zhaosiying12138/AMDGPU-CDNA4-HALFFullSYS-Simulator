import atexit
import ctypes
import os
import struct
import threading
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


_MANAGED_API_VERSION = 1
_ABI_MAJOR = 1
_WAVEFRONT_SIZE = 64
_BUFFER_ALIGNMENT = 4096


class _ErrorInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("wire_status", ctypes.c_int32),
        ("native_errno", ctypes.c_int32),
        ("message", ctypes.c_char * 128),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class _SessionOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("queue_depth", ctypes.c_uint32),
        ("startup_timeout_ns", ctypes.c_uint64),
        ("operation_timeout_ns", ctypes.c_uint64),
        ("run_timeout_ns", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * 24),
    ]


class _SessionInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("external_endpoint", ctypes.c_uint32),
        ("connection_id", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("rank", ctypes.c_uint32),
        ("world_size", ctypes.c_uint32),
        ("child_pid", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("daemon_uuid", ctypes.c_uint8 * 16),
        ("job_uuid", ctypes.c_uint8 * 16),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class _MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("allocation_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("simulated_va", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
        ("alignment_bytes", ctypes.c_uint64),
        ("connection_id", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("daemon_uuid", ctypes.c_uint8 * 16),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class _KernelInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("kernel_index", ctypes.c_uint32),
        ("kernarg_segment_size", ctypes.c_uint32),
        ("kernarg_segment_align", ctypes.c_uint32),
        ("group_segment_fixed_size", ctypes.c_uint32),
        ("private_segment_fixed_size", ctypes.c_uint32),
        ("max_flat_workgroup_size", ctypes.c_uint32),
        ("wavefront_size", ctypes.c_uint32),
        ("descriptor_preload_dwords", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("object_id", ctypes.c_uint64),
        ("object_generation", ctypes.c_uint64),
        ("mapping_id", ctypes.c_uint64),
        ("mapping_generation", ctypes.c_uint64),
        ("entry_va", ctypes.c_uint64),
        ("kernarg_va", ctypes.c_uint64),
        ("image_sha256", ctypes.c_uint8 * 32),
        ("connection_id", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("daemon_uuid", ctypes.c_uint8 * 16),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class _LaunchOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("grid_x", ctypes.c_uint32),
        ("grid_y", ctypes.c_uint32),
        ("grid_z", ctypes.c_uint32),
        ("workgroup_x", ctypes.c_uint32),
        ("workgroup_y", ctypes.c_uint32),
        ("workgroup_z", ctypes.c_uint32),
        ("num_warps", ctypes.c_uint32),
        ("num_ctas", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("wavefront_size", ctypes.c_uint32),
        ("launch_flags", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 16),
    ]


class _DispatchCompletion(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("wire_status", ctypes.c_int32),
        ("request_id", ctypes.c_uint64),
        ("object_id", ctypes.c_uint64),
        ("object_generation", ctypes.c_uint64),
        ("mapping_id", ctypes.c_uint64),
        ("mapping_generation", ctypes.c_uint64),
        ("kernarg_allocation_id", ctypes.c_uint64),
        ("kernarg_generation", ctypes.c_uint64),
        ("kernarg_va", ctypes.c_uint64),
        ("kernarg_size", ctypes.c_uint64),
        ("kernarg_alignment", ctypes.c_uint64),
        ("queue_id", ctypes.c_uint64),
        ("queue_generation", ctypes.c_uint64),
        ("queue_sequence", ctypes.c_uint64),
        ("signal_id", ctypes.c_uint64),
        ("signal_generation", ctypes.c_uint64),
        ("signal_value_bits", ctypes.c_uint64),
        ("ticket_id", ctypes.c_uint64),
        ("trace_id", ctypes.c_uint64),
        ("packet_va", ctypes.c_uint64),
        ("packet_crc32c", ctypes.c_uint32),
        ("output_crc32c", ctypes.c_uint32),
        ("sim_tick", ctypes.c_uint64),
        ("admission_tick", ctypes.c_uint64),
        ("start_tick", ctypes.c_uint64),
        ("end_tick", ctypes.c_uint64),
        ("retire_tick", ctypes.c_uint64),
        ("image_sha256", ctypes.c_uint8 * 32),
        ("connection_id", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("daemon_uuid", ctypes.c_uint8 * 16),
        ("reserved", ctypes.c_uint8 * 16),
    ]


_EXPECTED_STRUCTURE_SIZES = {
    _ErrorInfo: 160,
    _SessionOptions: 64,
    _SessionInfo: 96,
    _MemoryInfo: 96,
    _KernelInfo: 176,
    _LaunchOptions: 76,
    _DispatchCompletion: 304,
}


class _ManagedBuffer:

    def __init__(self, handle, info, host_address, host_size):
        self.handle = handle
        self.info = info
        self.host_address = host_address
        self.host_size = host_size


class _ManagedKernel:

    def __init__(self, runtime, name, handle, info):
        self.runtime = runtime
        self.name = name
        self.handle = handle
        self.info = info

    def close(self):
        if self.handle.value:
            self.runtime.unload_kernel(self)


def _runtime_library_path():
    prefix = os.environ.get("ROCM_SIM_ROOT")
    if not prefix:
        raise RuntimeError(
            "ROCM_SIM_ROOT is unset; activate the repository-local ROCm "
            "environment before importing Triton"
        )
    path = Path(prefix).resolve() / "lib" / "libself_amdgpu_runtime.so.1"
    if not path.is_file():
        raise RuntimeError(f"repository-local managed runtime is missing: {path}")
    return path


class _ManagedRuntime:

    def __init__(self):
        for structure, expected in _EXPECTED_STRUCTURE_SIZES.items():
            actual = ctypes.sizeof(structure)
            if actual != expected:
                raise RuntimeError(
                    f"ctypes ABI mismatch for {structure.__name__}: "
                    f"expected {expected}, got {actual}"
                )
        self.path = _runtime_library_path()
        self.lib = None
        self.lock = threading.RLock()
        self.session = ctypes.c_void_p()
        self.session_info = None
        atexit.register(self._close_at_exit)

    def _ensure_library(self):
        if self.lib is not None:
            return
        lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        lib.sagr_abi_version.argtypes = []
        lib.sagr_abi_version.restype = ctypes.c_uint32
        lib.sagr_status_string.argtypes = [ctypes.c_int32]
        lib.sagr_status_string.restype = ctypes.c_char_p
        lib.sagr_managed_session_options_init.argtypes = [
            ctypes.POINTER(_SessionOptions),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_options_init.restype = ctypes.c_int32
        lib.sagr_managed_launch_options_init.argtypes = [
            ctypes.POINTER(_LaunchOptions),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_launch_options_init.restype = ctypes.c_int32
        lib.sagr_managed_session_open.argtypes = [
            ctypes.POINTER(_SessionOptions),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_SessionInfo),
            ctypes.c_uint32,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_open.restype = ctypes.c_int32
        lib.sagr_managed_session_close.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_close.restype = ctypes.c_int32
        lib.sagr_managed_buffer_allocate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_MemoryInfo),
            ctypes.c_uint32,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_buffer_allocate.restype = ctypes.c_int32
        copy_args = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_buffer_copy_from_host.argtypes = copy_args
        lib.sagr_managed_buffer_copy_from_host.restype = ctypes.c_int32
        lib.sagr_managed_buffer_copy_to_host.argtypes = copy_args
        lib.sagr_managed_buffer_copy_to_host.restype = ctypes.c_int32
        lib.sagr_managed_buffer_free.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_buffer_free.restype = ctypes.c_int32
        lib.sagr_managed_kernel_load.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_KernelInfo),
            ctypes.c_uint32,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_kernel_load.restype = ctypes.c_int32
        lib.sagr_managed_kernel_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_LaunchOptions),
            ctypes.POINTER(_DispatchCompletion),
            ctypes.c_uint32,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_kernel_launch.restype = ctypes.c_int32
        lib.sagr_managed_kernel_unload.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_kernel_unload.restype = ctypes.c_int32
        abi_version = lib.sagr_abi_version()
        if abi_version >> 16 != _ABI_MAJOR:
            raise RuntimeError(
                f"managed runtime ABI mismatch in {self.path}: expected major "
                f"{_ABI_MAJOR}, got 0x{abi_version:08x}"
            )
        self.lib = lib

    def _check(self, status, operation, error=None):
        if status == 0:
            return
        status_text = self.lib.sagr_status_string(status)
        status_text = status_text.decode("utf-8", "replace") if status_text else "unknown"
        detail = ""
        if error is not None:
            detail = bytes(error.message).split(b"\0", 1)[0].decode("utf-8", "replace")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"gemsim_amd {operation} failed with status {status} "
            f"({status_text}){suffix}"
        )

    def _ensure_session(self):
        self._ensure_library()
        if self.session.value:
            return
        options = _SessionOptions()
        status = self.lib.sagr_managed_session_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        )
        self._check(status, "session options initialization")
        info = _SessionInfo()
        error = _ErrorInfo()
        status = self.lib.sagr_managed_session_open(
            ctypes.byref(options),
            ctypes.byref(self.session),
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(error),
            ctypes.sizeof(error),
        )
        self._check(status, "session open", error)
        self.session_info = info

    def load_kernel(self, name, image):
        with self.lock:
            self._ensure_session()
            image_buffer = ctypes.create_string_buffer(image)
            handle = ctypes.c_void_p()
            info = _KernelInfo()
            error = _ErrorInfo()
            status = self.lib.sagr_managed_kernel_load(
                self.session,
                ctypes.cast(image_buffer, ctypes.c_void_p),
                len(image),
                name.encode("utf-8"),
                ctypes.byref(handle),
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
            self._check(status, f"kernel load ({name})", error)
            if (
                info.version != _MANAGED_API_VERSION
                or info.wavefront_size != _WAVEFRONT_SIZE
            ):
                self._unload_handle(handle)
                raise RuntimeError(
                    f"gemsim_amd kernel {name} has unsupported managed metadata"
                )
            return _ManagedKernel(self, name, handle, info)

    def unload_kernel(self, kernel):
        with self.lock:
            self._unload_handle(kernel.handle)

    def _unload_handle(self, handle):
        if not handle.value:
            return
        if not self.session.value:
            handle.value = None
            return
        error = _ErrorInfo()
        status = self.lib.sagr_managed_kernel_unload(
            ctypes.byref(handle), ctypes.byref(error), ctypes.sizeof(error)
        )
        self._check(status, "kernel unload", error)

    def allocate_buffer(self, host_address, host_size, alignment=_BUFFER_ALIGNMENT):
        with self.lock:
            self._ensure_session()
            allocation_size = max(host_size, 1)
            handle = ctypes.c_void_p()
            info = _MemoryInfo()
            error = _ErrorInfo()
            status = self.lib.sagr_managed_buffer_allocate(
                self.session,
                allocation_size,
                max(alignment, _BUFFER_ALIGNMENT),
                ctypes.byref(handle),
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
            self._check(status, "buffer allocation", error)
            buffer = _ManagedBuffer(handle, info, host_address, host_size)
            try:
                if host_size:
                    self.copy_from_host(buffer)
            except BaseException as copy_error:
                try:
                    self.free_buffer(buffer)
                except Exception as cleanup_error:
                    raise cleanup_error from copy_error
                raise
            return buffer

    def copy_from_host(self, buffer):
        error = _ErrorInfo()
        status = self.lib.sagr_managed_buffer_copy_from_host(
            buffer.handle,
            0,
            ctypes.c_void_p(buffer.host_address),
            buffer.host_size,
            ctypes.byref(error),
            ctypes.sizeof(error),
        )
        self._check(status, "host-to-simulator copy", error)

    def copy_to_host(self, buffer):
        if not buffer.host_size:
            return
        error = _ErrorInfo()
        status = self.lib.sagr_managed_buffer_copy_to_host(
            buffer.handle,
            0,
            ctypes.c_void_p(buffer.host_address),
            buffer.host_size,
            ctypes.byref(error),
            ctypes.sizeof(error),
        )
        self._check(status, "simulator-to-host copy", error)

    def free_buffer(self, buffer):
        if not buffer.handle.value:
            return
        error = _ErrorInfo()
        status = self.lib.sagr_managed_buffer_free(
            ctypes.byref(buffer.handle), ctypes.byref(error), ctypes.sizeof(error)
        )
        self._check(status, "buffer free", error)

    def launch(self, kernel, kernarg, options):
        with self.lock:
            packed = (ctypes.c_uint8 * len(kernarg)).from_buffer_copy(kernarg)
            completion = _DispatchCompletion()
            error = _ErrorInfo()
            status = self.lib.sagr_managed_kernel_launch(
                kernel.handle,
                ctypes.cast(packed, ctypes.c_void_p),
                len(kernarg),
                ctypes.byref(options),
                ctypes.byref(completion),
                ctypes.sizeof(completion),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
            self._check(status, f"kernel launch ({kernel.name})", error)
            if completion.status != 0:
                self._check(completion.status, f"kernel completion ({kernel.name})")
            return completion

    def close(self):
        with self.lock:
            if self.lib is None or not self.session.value:
                return
            error = _ErrorInfo()
            status = self.lib.sagr_managed_session_close(
                ctypes.byref(self.session),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
            self._check(status, "session close", error)

    def _close_at_exit(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown must not obscure the program's actual result.
            pass


class _GemsimUtils:

    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def get_device_properties(device):
        if device != 0:
            raise ValueError("gemsim_amd exposes exactly one device")
        return {
            "arch": "gfx950",
            "warpSize": _WAVEFRONT_SIZE,
            "max_shared_mem": 65_536,
            "multiprocessor_count": 1,
        }

    def load_binary(self, name, image, shared, device):
        if device != 0:
            raise ValueError("gemsim_amd exposes exactly one device")
        kernel = self.runtime.load_kernel(name, image)
        if shared > 65_536:
            kernel.close()
            raise RuntimeError(f"gemsim_amd shared-memory request is too large: {shared}")
        return (
            kernel,
            kernel,
            0,
            0,
            kernel.info.max_flat_workgroup_size,
        )

    @staticmethod
    def unload_module(module):
        if isinstance(module, _ManagedKernel):
            module.close()


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _unwrap_tensor(value):
    base = getattr(value, "base", value)
    return base if hasattr(base, "untyped_storage") else None


def _storage_description(value):
    tensor = _unwrap_tensor(value)
    if tensor is None:
        raise TypeError(
            "gemsim_amd pointer arguments must be CPU torch tensors; raw host "
            "or accelerator pointers are not accepted"
        )
    if getattr(tensor.device, "type", None) != "cpu":
        raise ValueError(
            f"gemsim_amd accepts CPU tensors for synchronous staging, got {tensor.device}"
        )
    storage = tensor.untyped_storage()
    host_address = int(storage.data_ptr())
    host_size = int(storage.nbytes())
    argument_address = int(value.data_ptr())
    offset = argument_address - host_address
    if offset < 0 or offset > host_size:
        raise ValueError("tensor data pointer is outside its CPU storage")
    return (host_address, host_size), offset


_SCALAR_FORMATS = {
    "i1": (4, 4, "i"),
    "i8": (1, 1, "b"),
    "i16": (2, 2, "h"),
    "i32": (4, 4, "i"),
    "i64": (8, 8, "q"),
    "u1": (4, 4, "I"),
    "u8": (1, 1, "B"),
    "u16": (2, 2, "H"),
    "u32": (4, 4, "I"),
    "u64": (8, 8, "Q"),
    "fp16": (2, 2, "e"),
    "fp32": (4, 4, "f"),
    "f32": (4, 4, "f"),
    "fp64": (8, 8, "d"),
}


def _pack_scalar(type_name, value):
    if type_name == "bf16":
        raw = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return 2, 2, struct.pack("<H", raw >> 16)
    try:
        size, alignment, fmt = _SCALAR_FORMATS[type_name]
    except KeyError as exc:
        raise TypeError(f"gemsim_amd does not support scalar argument type {type_name}") from exc
    return size, alignment, struct.pack("<" + fmt, value)


class _GemsimLauncher:

    def __init__(self, src, metadata):
        if any(len(path) != 1 for path in getattr(src, "constants", {})):
            raise TypeError("gemsim_amd does not yet support nested constexpr arguments")
        constant_indices = {path[0] for path in getattr(src, "constants", {})}
        self.arguments = [
            (index, type_name)
            for index, type_name in enumerate(src.signature.values())
            if index not in constant_indices and type_name != "constexpr"
        ]
        if any(not isinstance(type_name, str) for _, type_name in self.arguments):
            raise TypeError("gemsim_amd does not yet support nested kernel arguments")
        self.metadata = metadata

    def __call__(self, *args, **kwargs):
        if len(args) < 5 or not isinstance(args[4], _ManagedKernel):
            raise TypeError("gemsim_amd received an invalid managed kernel handle")
        with args[4].runtime.lock:
            return self._launch(*args, **kwargs)

    def _launch(
        self,
        grid_x,
        grid_y,
        grid_z,
        stream,
        function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        if stream not in (None, 0):
            raise ValueError("gemsim_amd exposes only its synchronous default stream")
        if not isinstance(function, _ManagedKernel) or not function.handle.value:
            raise TypeError("gemsim_amd received an invalid managed kernel handle")
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        runtime = function.runtime
        staged = {}
        buffers = []
        packed_values = []
        packed_size = 0
        operation_failed = False
        try:
            for index, type_name in self.arguments:
                if index >= len(args):
                    raise TypeError("gemsim_amd launcher received too few kernel arguments")
                value = args[index]
                if type_name.startswith("*"):
                    packed_size = _align_up(packed_size, 8)
                    if value is None or (isinstance(value, int) and value == 0):
                        simulated_va = 0
                    elif isinstance(value, int):
                        raise TypeError(
                            "gemsim_amd rejects nonzero raw pointer arguments; "
                            "pass a CPU torch tensor for managed staging"
                        )
                    else:
                        key, storage_offset = _storage_description(value)
                        buffer = staged.get(key)
                        if buffer is None:
                            buffer = runtime.allocate_buffer(*key)
                            staged[key] = buffer
                            buffers.append(buffer)
                        simulated_va = buffer.info.simulated_va + storage_offset
                    packed_values.append((packed_size, struct.pack("<Q", simulated_va)))
                    packed_size += 8
                    continue
                size, alignment, raw = _pack_scalar(type_name, value)
                packed_size = _align_up(packed_size, alignment)
                packed_values.append((packed_size, raw))
                packed_size += size

            grid_programs = int(grid_x) * int(grid_y) * int(grid_z)
            for scratch_size, scratch_alignment in (
                (self.metadata.global_scratch_size, self.metadata.global_scratch_align),
                (self.metadata.profile_scratch_size, self.metadata.profile_scratch_align),
            ):
                packed_size = _align_up(packed_size, 8)
                simulated_va = 0
                total_size = grid_programs * int(scratch_size)
                if total_size:
                    scratch = ctypes.create_string_buffer(total_size)
                    buffer = runtime.allocate_buffer(
                        ctypes.addressof(scratch), total_size, int(scratch_alignment)
                    )
                    buffer.scratch_owner = scratch
                    buffers.append(buffer)
                    simulated_va = buffer.info.simulated_va
                packed_values.append((packed_size, struct.pack("<Q", simulated_va)))
                packed_size += 8

            required_size = int(function.info.kernarg_segment_size)
            if packed_size > required_size:
                raise RuntimeError(
                    f"gemsim_amd packed {packed_size} kernarg bytes, but {function.name} "
                    f"declares only {required_size}"
                )
            kernarg = bytearray(required_size)
            for offset, raw in packed_values:
                kernarg[offset:offset + len(raw)] = raw

            num_warps, num_ctas, shared = map(int, kernel_metadata)
            workgroup_x = num_warps * _WAVEFRONT_SIZE
            if workgroup_x > function.info.max_flat_workgroup_size:
                raise RuntimeError(
                    f"gemsim_amd workgroup {workgroup_x} exceeds kernel limit "
                    f"{function.info.max_flat_workgroup_size}"
                )
            options = _LaunchOptions()
            status = runtime.lib.sagr_managed_launch_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            )
            runtime._check(status, "launch options initialization")
            options.grid_x = int(grid_x) * num_ctas * workgroup_x
            options.grid_y = int(grid_y)
            options.grid_z = int(grid_z)
            options.workgroup_x = workgroup_x
            options.workgroup_y = 1
            options.workgroup_z = 1
            options.num_warps = num_warps
            options.num_ctas = num_ctas
            options.shared_memory_bytes = shared
            options.wavefront_size = _WAVEFRONT_SIZE
            runtime.launch(function, kernarg, options)
            for buffer in staged.values():
                runtime.copy_to_host(buffer)
        except BaseException:
            operation_failed = True
            raise
        finally:
            cleanup_error = None
            for buffer in reversed(buffers):
                try:
                    runtime.free_buffer(buffer)
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and not operation_failed:
                raise cleanup_error

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class GemsimAMDDriver(DriverBase):

    def __init__(self):
        self.runtime = _ManagedRuntime()
        self.utils = _GemsimUtils(self.runtime)
        self.launcher_cls = _GemsimLauncher
        self.get_current_device = lambda: 0
        self.set_current_device = self._set_current_device
        self.get_current_stream = lambda device: 0

    @classmethod
    def is_active(cls):
        try:
            _runtime_library_path()
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _set_current_device(device):
        if device != 0:
            raise ValueError("gemsim_amd exposes exactly one device")

    def map_python_to_cpp_type(self, ty):
        if ty.startswith("*"):
            return "uint64_t"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint32_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "uint16_t",
            "bf16": "uint16_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def get_current_target(self):
        return GPUTarget("gemsim_amd", "gfx950", _WAVEFRONT_SIZE)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_benchmarker(self):
        def unsupported(*args, **kwargs):
            raise RuntimeError(
                "gemsim_amd profiling is deferred until a retained workload "
                "demonstrates a material bottleneck"
            )

        return unsupported
