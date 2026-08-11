import ctypes
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from triton.backends.compiler import GPUTarget
from triton.backends.gemsim_amd import compiler
from triton.backends.gemsim_amd import driver


class _LaunchOptionsInit:

    def __call__(self, options_pointer, size):
        options = ctypes.cast(
            options_pointer, ctypes.POINTER(driver._LaunchOptions)
        ).contents
        options.struct_size = size
        options.version = 1
        return 0


class _BufferAllocate:

    def __call__(
        self,
        session,
        size,
        alignment,
        handle_pointer,
        info_pointer,
        info_size,
        error_pointer,
        error_size,
    ):
        handle = ctypes.cast(
            handle_pointer, ctypes.POINTER(ctypes.c_void_p)
        ).contents
        handle.value = 11
        info = ctypes.cast(
            info_pointer, ctypes.POINTER(driver._MemoryInfo)
        ).contents
        info.simulated_va = 0x100000
        info.size_bytes = size
        info.alignment_bytes = alignment
        return 0


class _FakeRuntime:

    def __init__(self, launch_error=None):
        self.lock = threading.RLock()
        self.lib = SimpleNamespace(
            sagr_managed_launch_options_init=_LaunchOptionsInit()
        )
        self.launch_error = launch_error
        self.allocations = []
        self.copies_to_host = []
        self.frees = []
        self.launches = []
        self._next_va = 0x100000

    def allocate_buffer(self, host_address, host_size, alignment=4096):
        info = driver._MemoryInfo()
        info.simulated_va = self._next_va
        info.size_bytes = host_size
        info.alignment_bytes = alignment
        self._next_va += 0x100000
        buffer = driver._ManagedBuffer(
            ctypes.c_void_p(len(self.allocations) + 1),
            info,
            host_address,
            host_size,
        )
        self.allocations.append(buffer)
        return buffer

    def copy_to_host(self, buffer):
        self.copies_to_host.append(buffer)

    def free_buffer(self, buffer):
        self.frees.append(buffer)
        buffer.handle.value = None

    def launch(self, kernel, kernarg, options):
        copied_options = driver._LaunchOptions.from_buffer_copy(bytes(options))
        self.launches.append((kernel, bytes(kernarg), copied_options))
        if self.launch_error is not None:
            raise self.launch_error

    @staticmethod
    def _check(status, operation):
        if status != 0:
            raise RuntimeError(f"{operation} failed")

    @staticmethod
    def unload_kernel(kernel):
        kernel.handle.value = None


def _source(*types, constant_indices=()):
    return SimpleNamespace(
        signature={index: type_name for index, type_name in enumerate(types)},
        constants={(index,): None for index in constant_indices},
    )


def _metadata():
    return SimpleNamespace(
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
    )


def _kernel(runtime, kernarg_size):
    info = driver._KernelInfo()
    info.version = 1
    info.kernarg_segment_size = kernarg_size
    info.kernarg_segment_align = 8
    info.max_flat_workgroup_size = 256
    info.wavefront_size = 64
    return driver._ManagedKernel(
        runtime, "add_kernel", ctypes.c_void_p(7), info
    )


def _launch(launcher, kernel, *arguments, stream=0):
    return launcher(
        97,
        1,
        1,
        stream,
        kernel,
        (4, 1, 0),
        None,
        None,
        None,
        *arguments,
    )


class ManagedAbiTest(unittest.TestCase):

    def test_structure_sizes_and_offsets(self):
        expected_sizes = {
            driver._ErrorInfo: 160,
            driver._SessionOptions: 64,
            driver._SessionInfo: 96,
            driver._MemoryInfo: 96,
            driver._KernelInfo: 176,
            driver._LaunchOptions: 76,
            driver._DispatchCompletion: 304,
        }
        self.assertEqual(driver._EXPECTED_STRUCTURE_SIZES, expected_sizes)
        for structure, expected in expected_sizes.items():
            self.assertEqual(ctypes.sizeof(structure), expected)
        self.assertEqual(driver._MemoryInfo.simulated_va.offset, 24)
        self.assertEqual(driver._KernelInfo.kernarg_segment_size.offset, 16)
        self.assertEqual(driver._KernelInfo.entry_va.offset, 80)
        self.assertEqual(driver._LaunchOptions.grid_x.offset, 12)
        self.assertEqual(driver._DispatchCompletion.status.offset, 12)
        self.assertEqual(driver._DispatchCompletion.request_id.offset, 24)
        self.assertEqual(driver._DispatchCompletion.sim_tick.offset, 184)
        self.assertEqual(driver._DispatchCompletion.image_sha256.offset, 224)

    def test_runtime_dso_binding_is_lazy(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "lib" / "libself_amdgpu_runtime.so.1"
            library.parent.mkdir()
            library.touch()
            with mock.patch.dict("os.environ", {"ROCM_SIM_ROOT": directory}), \
                    mock.patch.object(driver.ctypes, "CDLL") as load, \
                    mock.patch.object(driver.atexit, "register") as register:
                runtime = driver._ManagedRuntime()
                self.assertEqual(runtime.path, library)
                self.assertIsNone(runtime.lib)
                load.assert_not_called()
                register.assert_called_once_with(runtime._close_at_exit)
                runtime.close()

    def test_initial_copy_failure_frees_buffer_and_chains_cleanup(self):
        runtime = object.__new__(driver._ManagedRuntime)
        runtime.lock = threading.RLock()
        runtime.session = ctypes.c_void_p(1)
        runtime.lib = SimpleNamespace(
            sagr_managed_buffer_allocate=_BufferAllocate(),
            sagr_status_string=lambda status: b"injected",
        )
        runtime._ensure_session = mock.Mock()
        copy_error = RuntimeError("injected H2D failure")
        runtime.copy_from_host = mock.Mock(side_effect=copy_error)

        def release(buffer):
            buffer.handle.value = None

        runtime.free_buffer = mock.Mock(side_effect=release)
        with self.assertRaisesRegex(RuntimeError, "injected H2D failure"):
            runtime.allocate_buffer(0x1000, 16)
        runtime.free_buffer.assert_called_once()
        self.assertFalse(runtime.free_buffer.call_args.args[0].handle.value)

        cleanup_error = RuntimeError("injected cleanup failure")
        runtime.free_buffer = mock.Mock(side_effect=cleanup_error)
        with self.assertRaisesRegex(RuntimeError, "injected cleanup failure") as raised:
            runtime.allocate_buffer(0x1000, 16)
        self.assertIs(raised.exception.__cause__, copy_error)

    def test_explicit_close_is_strict_and_atexit_close_is_best_effort(self):
        runtime = object.__new__(driver._ManagedRuntime)
        runtime.lock = threading.RLock()
        runtime.session = ctypes.c_void_p(1)
        session_close = mock.Mock(return_value=7)
        runtime.lib = SimpleNamespace(
            sagr_managed_session_close=session_close,
            sagr_status_string=lambda status: b"injected",
        )

        with self.assertRaisesRegex(RuntimeError, "session close failed"):
            runtime.close()
        runtime._close_at_exit()
        self.assertEqual(session_close.call_count, 2)


class PackingAndStagingTest(unittest.TestCase):

    def test_scalar_encoding(self):
        self.assertEqual(driver._pack_scalar("i32", -7), (4, 4, struct.pack("<i", -7)))
        self.assertEqual(driver._pack_scalar("u64", 9), (8, 8, struct.pack("<Q", 9)))
        self.assertEqual(driver._pack_scalar("fp32", 1.5), (4, 4, struct.pack("<f", 1.5)))
        self.assertEqual(driver._pack_scalar("bf16", 1.5), (2, 2, b"\xc0?"))
        with self.assertRaisesRegex(TypeError, "scalar argument type"):
            driver._pack_scalar("fp8", 1)

    def test_tutorial_kernarg_grid_and_cpu_staging(self):
        runtime = _FakeRuntime()
        launcher = driver._GemsimLauncher(
            _source("*fp32", "*fp32", "*fp32", "i32", "constexpr",
                    constant_indices=(4,)),
            _metadata(),
        )
        kernel = _kernel(runtime, 48)
        x = torch.arange(16, dtype=torch.float32)
        y = torch.arange(16, dtype=torch.float32) + 1
        output = torch.empty_like(x)

        _launch(launcher, kernel, x, y, output, 98432, 1024)

        self.assertEqual(len(runtime.allocations), 3)
        self.assertEqual(runtime.copies_to_host, runtime.allocations)
        self.assertEqual(runtime.frees, list(reversed(runtime.allocations)))
        self.assertEqual(len(runtime.launches), 1)
        _, kernarg, options = runtime.launches[0]
        self.assertEqual(len(kernarg), 48)
        self.assertEqual(
            struct.unpack_from("<QQQi", kernarg),
            (0x100000, 0x200000, 0x300000, 98432),
        )
        self.assertEqual(kernarg[28:], bytes(20))
        self.assertEqual(options.grid_x, 97 * 4 * 64)
        self.assertEqual(options.grid_y, 1)
        self.assertEqual(options.grid_z, 1)
        self.assertEqual(options.workgroup_x, 256)
        self.assertEqual(options.num_warps, 4)
        self.assertEqual(options.num_ctas, 1)

    def test_storage_alias_is_deduplicated_and_invalid_alias_is_rejected(self):
        runtime = _FakeRuntime()
        launcher = driver._GemsimLauncher(
            _source("*fp32", "*fp32", "i32"), _metadata()
        )
        kernel = _kernel(runtime, 40)
        base = torch.arange(16, dtype=torch.float32)
        view = base[4:]

        _launch(launcher, kernel, base, view, 12)

        self.assertEqual(len(runtime.allocations), 1)
        _, kernarg, _ = runtime.launches[0]
        first, second = struct.unpack_from("<QQ", kernarg)
        self.assertEqual(first, 0x100000)
        self.assertEqual(second, first + 4 * base.element_size())

        invalid = SimpleNamespace(
            base=base,
            data_ptr=lambda: base.untyped_storage().data_ptr()
            + base.untyped_storage().nbytes()
            + 4,
        )
        with self.assertRaisesRegex(ValueError, "outside its CPU storage"):
            driver._storage_description(invalid)


class LauncherFailureTest(unittest.TestCase):

    def setUp(self):
        self.launcher = driver._GemsimLauncher(
            _source("*fp32", "*fp32", "*fp32", "i32"), _metadata()
        )
        self.tensors = tuple(
            torch.arange(8, dtype=torch.float32) + index
            for index in range(3)
        )

    def test_invalid_stream_and_raw_pointer_are_rejected(self):
        runtime = _FakeRuntime()
        kernel = _kernel(runtime, 48)
        with self.assertRaisesRegex(ValueError, "synchronous default stream"):
            _launch(self.launcher, kernel, *self.tensors, 8, stream=1)
        self.assertEqual(runtime.allocations, [])

        with self.assertRaisesRegex(TypeError, "nonzero raw pointer"):
            _launch(self.launcher, kernel, 0x1234, *self.tensors[1:], 8)
        self.assertEqual(runtime.allocations, [])
        self.assertEqual(runtime.launches, [])

    def test_buffers_are_freed_when_launch_fails(self):
        runtime = _FakeRuntime(RuntimeError("injected launch failure"))
        kernel = _kernel(runtime, 48)
        with self.assertRaisesRegex(RuntimeError, "injected launch failure"):
            _launch(self.launcher, kernel, *self.tensors, 8)
        self.assertEqual(len(runtime.allocations), 3)
        self.assertEqual(runtime.copies_to_host, [])
        self.assertEqual(runtime.frees, list(reversed(runtime.allocations)))


class ActiveTargetTest(unittest.TestCase):

    def test_only_gemsim_target_is_supported_and_exposes_cpu(self):
        target = GPUTarget("gemsim_amd", "gfx950", 64)
        self.assertTrue(compiler.GemsimAMDBackend.supports_target(target))
        self.assertFalse(
            compiler.GemsimAMDBackend.supports_target(
                GPUTarget("hip", "gfx950", 64)
            )
        )
        backend = compiler.GemsimAMDBackend(target)
        self.assertEqual(backend.parse_options({}).backend_name, "gemsim_amd")

        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "lib" / "libself_amdgpu_runtime.so.1"
            library.parent.mkdir()
            library.touch()
            with mock.patch.dict("os.environ", {"ROCM_SIM_ROOT": directory}):
                active_driver = driver.GemsimAMDDriver()
                current = active_driver.get_current_target()
                self.assertEqual(current.backend, "gemsim_amd")
                self.assertEqual(current.arch, "gfx950")
                self.assertEqual(current.warp_size, 64)
                self.assertEqual(
                    active_driver.get_active_torch_device(), torch.device("cpu")
                )
                self.assertTrue(driver.GemsimAMDDriver.is_active())
                self.assertIsNone(active_driver.runtime.lib)


if __name__ == "__main__":
    unittest.main()
