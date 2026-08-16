import atexit
import ctypes
import hashlib
import json
import os
import stat
import struct
import threading
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


_MANAGED_API_VERSION = 1
_MANAGED_SESSION_OPTIONS_V2_VERSION = 2
_MANAGED_SESSION_V2_FLAG_PRIVATE_NAMESPACE = 2
_MANAGED_ENDPOINT_BYTES = 108
_MANAGED_MAX_WORLD_SIZE = 16
_RANK_DESCRIPTOR_MAX_BYTES = 64 * 1024
_ABI_MAJOR = 1
_ABI_MINOR = 7
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


class _SessionOptionsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("queue_depth", ctypes.c_uint32),
        ("startup_timeout_ns", ctypes.c_uint64),
        ("operation_timeout_ns", ctypes.c_uint64),
        ("run_timeout_ns", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
        ("rank", ctypes.c_uint32),
        ("world_size", ctypes.c_uint32),
        ("job_uuid", ctypes.c_uint8 * 16),
        ("endpoint", ctypes.c_uint8 * _MANAGED_ENDPOINT_BYTES),
        ("reserved", ctypes.c_uint8 * 20),
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
    _SessionOptionsV2: 200,
    _SessionInfo: 96,
    _MemoryInfo: 96,
    _KernelInfo: 176,
    _LaunchOptions: 76,
    _DispatchCompletion: 304,
}


def _canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def _read_private_file(path):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            "gemsim rank launch descriptor could not be opened safely"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or metadata.st_size <= 0
            or metadata.st_size > _RANK_DESCRIPTOR_MAX_BYTES
        ):
            raise RuntimeError("gemsim rank launch descriptor is not private")
        chunks = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RuntimeError("gemsim rank launch descriptor was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("gemsim rank launch descriptor changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _rank_launch_descriptor():
    value = os.environ.get("GEMSIM_RANK_LAUNCH_DESCRIPTOR")
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise RuntimeError("gemsim rank launch descriptor path is invalid")
    data = _read_private_file(path)
    try:
        document = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("gemsim rank launch descriptor is invalid JSON") from error
    if not isinstance(document, dict) or data != _canonical_json(document):
        raise RuntimeError("gemsim rank launch descriptor is not canonical")
    if set(document) != {
        "schema",
        "job_uuid",
        "epoch",
        "rank",
        "world_size",
        "paths",
    } or document["schema"] != "amdgpu-sim.gemsim-rank-launch.v1":
        raise RuntimeError("gemsim rank launch descriptor schema is invalid")
    paths = document["paths"]
    expected_path_keys = {
        "instance_directory",
        "triton_cache_directory",
        "runtime_directory",
        "endpoint",
        "gem5_output_directory",
        "dispatch_trace_path",
        "gem5_log_path",
        "gem5_cache_directory",
    }
    if not isinstance(paths, dict) or set(paths) != expected_path_keys:
        raise RuntimeError("gemsim rank launch descriptor paths are invalid")
    normalized_paths = {}
    for name in expected_path_keys:
        path_value = paths[name]
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or Path(path_value) != Path(os.path.normpath(path_value))
        ):
            raise RuntimeError(f"gemsim rank launch path {name} is invalid")
        normalized_paths[name] = Path(path_value)
    instance_directory = normalized_paths["instance_directory"]
    runtime_directory = normalized_paths["runtime_directory"]
    if runtime_directory.parent != instance_directory:
        raise RuntimeError("gemsim rank launch runtime namespace is invalid")
    for name in (
        "endpoint",
        "gem5_output_directory",
        "dispatch_trace_path",
        "gem5_log_path",
        "gem5_cache_directory",
    ):
        if normalized_paths[name].parent != runtime_directory:
            raise RuntimeError("gemsim rank launch runtime paths are not isolated")
    job_uuid = document["job_uuid"]
    endpoint = paths["endpoint"]
    if (
        not isinstance(job_uuid, str)
        or len(job_uuid) != 32
        or job_uuid == "0" * 32
        or any(character not in "0123456789abcdef" for character in job_uuid)
        or not isinstance(document["epoch"], int)
        or isinstance(document["epoch"], bool)
        or document["epoch"] <= 0
        or not isinstance(document["rank"], int)
        or isinstance(document["rank"], bool)
        or not isinstance(document["world_size"], int)
        or isinstance(document["world_size"], bool)
        or document["world_size"] < 2
        or document["world_size"] > _MANAGED_MAX_WORLD_SIZE
        or not 0 <= document["rank"] < document["world_size"]
        or len(endpoint.encode("utf-8")) >= _MANAGED_ENDPOINT_BYTES
    ):
        raise RuntimeError("gemsim rank launch descriptor identity is invalid")
    ambient_cache = os.environ.get("TRITON_CACHE_DIR")
    expected_cache = str(normalized_paths["triton_cache_directory"])
    if ambient_cache is None or str(Path(ambient_cache).resolve()) != expected_cache:
        raise RuntimeError(
            "TRITON_CACHE_DIR does not match the immutable rank launch descriptor"
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "job_uuid": bytes.fromhex(job_uuid),
        "epoch": document["epoch"],
        "rank": document["rank"],
        "world_size": document["world_size"],
        "endpoint": endpoint.encode("utf-8"),
        "triton_cache_directory": expected_cache,
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
        self.owner_pid = os.getpid()
        self.forked_child = False
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)
        atexit.register(self._close_at_exit)

    def _after_fork_child(self):
        self.lock = threading.RLock()
        self.forked_child = True
        self.session = ctypes.c_void_p()
        self.session_info = None

    def _check_owner(self):
        if self.forked_child or os.getpid() != self.owner_pid:
            raise RuntimeError(
                "gemsim_amd managed runtime cannot use a session inherited across fork"
            )

    def _ensure_library(self):
        self._check_owner()
        if self.lib is not None:
            return
        lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        lib.sagr_abi_version.argtypes = []
        lib.sagr_abi_version.restype = ctypes.c_uint32
        abi_version = lib.sagr_abi_version()
        abi_major = abi_version >> 16
        abi_minor = abi_version & 0xFFFF
        if abi_major != _ABI_MAJOR or abi_minor < _ABI_MINOR:
            raise RuntimeError(
                f"managed runtime ABI mismatch in {self.path}: expected "
                f"{_ABI_MAJOR}.{_ABI_MINOR} or newer compatible minor, got "
                f"0x{abi_version:08x}"
            )
        lib.sagr_status_string.argtypes = [ctypes.c_int32]
        lib.sagr_status_string.restype = ctypes.c_char_p
        lib.sagr_managed_session_options_init.argtypes = [
            ctypes.POINTER(_SessionOptions),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_options_init.restype = ctypes.c_int32
        lib.sagr_managed_session_options_v2_init.argtypes = [
            ctypes.POINTER(_SessionOptionsV2),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_options_v2_init.restype = ctypes.c_int32
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
        lib.sagr_managed_session_open_v2.argtypes = [
            ctypes.POINTER(_SessionOptionsV2),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_SessionInfo),
            ctypes.c_uint32,
            ctypes.POINTER(_ErrorInfo),
            ctypes.c_uint32,
        ]
        lib.sagr_managed_session_open_v2.restype = ctypes.c_int32
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
        self._check_owner()
        self._ensure_library()
        if self.session.value:
            return
        descriptor = _rank_launch_descriptor()
        info = _SessionInfo()
        error = _ErrorInfo()
        if descriptor is None:
            options = _SessionOptions()
            status = self.lib.sagr_managed_session_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            )
            self._check(status, "session options initialization")
            status = self.lib.sagr_managed_session_open(
                ctypes.byref(options),
                ctypes.byref(self.session),
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
        else:
            options_v2 = _SessionOptionsV2()
            status = self.lib.sagr_managed_session_options_v2_init(
                ctypes.byref(options_v2), ctypes.sizeof(options_v2)
            )
            self._check(status, "exact-topology session options initialization")
            options_v2.flags = _MANAGED_SESSION_V2_FLAG_PRIVATE_NAMESPACE
            options_v2.epoch = descriptor["epoch"]
            options_v2.rank = descriptor["rank"]
            options_v2.world_size = descriptor["world_size"]
            options_v2.job_uuid[:] = descriptor["job_uuid"]
            endpoint = descriptor["endpoint"]
            options_v2.endpoint[: len(endpoint)] = endpoint
            status = self.lib.sagr_managed_session_open_v2(
                ctypes.byref(options_v2),
                ctypes.byref(self.session),
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(error),
                ctypes.sizeof(error),
            )
        self._check(status, "session open", error)
        if descriptor is not None and (
            info.epoch != descriptor["epoch"]
            or info.rank != descriptor["rank"]
            or info.world_size != descriptor["world_size"]
            or bytes(info.job_uuid) != descriptor["job_uuid"]
        ):
            close_error = _ErrorInfo()
            close_status = self.lib.sagr_managed_session_close(
                ctypes.byref(self.session),
                ctypes.byref(close_error),
                ctypes.sizeof(close_error),
            )
            self._check(close_status, "mismatched session close", close_error)
            raise RuntimeError("gemsim exact-topology session identity mismatch")
        self.session_info = info

    def load_kernel(self, name, image):
        with self.lock:
            self._check_owner()
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
            self._check_owner()
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
            self._check_owner()
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
        self._check_owner()
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
        self._check_owner()
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
        self._check_owner()
        if not buffer.handle.value:
            return
        error = _ErrorInfo()
        status = self.lib.sagr_managed_buffer_free(
            ctypes.byref(buffer.handle), ctypes.byref(error), ctypes.sizeof(error)
        )
        self._check(status, "buffer free", error)

    def launch(self, kernel, kernarg, options):
        with self.lock:
            self._check_owner()
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
            self._check_owner()
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
        if self.forked_child or os.getpid() != self.owner_pid:
            return
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
