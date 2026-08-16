"""Device arithmetic used by the standalone CCL transport."""

from __future__ import annotations

import dataclasses

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 256
_MAX_ELEMENT_COUNT = (1 << 31) - 1
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


@triton.jit
def _sum_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    element_count,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    left = tl.load(left_ptr + offsets, mask=mask).to(tl.float32)
    right = tl.load(right_ptr + offsets, mask=mask).to(tl.float32)
    result = left + right
    if IS_BF16:
        result = result.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(output_ptr + offsets, result, mask=mask)


@dataclasses.dataclass(frozen=True)
class DeviceSumCounters:
    device_reduction_launch_count: int
    host_reduction_count: int


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    start = int(storage.data_ptr())
    size = int(storage.nbytes())
    end = start + size
    if size < 0 or end < start:
        raise ValueError("invalid staged storage address range")
    return start, end


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must use the gemsim CPU staging device")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use bfloat16 or float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")


def _validate_operands(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    element_count: int,
) -> None:
    for name, tensor in (("left", left), ("right", right), ("output", output)):
        _validate_tensor(name, tensor)
    if left.dtype != right.dtype or left.dtype != output.dtype:
        raise TypeError("left, right, and output must use the same dtype")
    if left.numel() != right.numel() or left.numel() != output.numel():
        raise ValueError("left, right, and output must have the same element count")
    if type(element_count) is not int:
        raise TypeError("element_count must be an integer")
    if element_count < 0 or element_count > output.numel():
        raise ValueError("element_count is outside the tensor extent")
    if element_count > _MAX_ELEMENT_COUNT:
        raise ValueError("element_count exceeds the proven 32-bit kernel index limit")

    # A zero-size planner chunk accesses no storage, so aliasing is immaterial.
    if element_count == 0:
        return

    if output is not left or (
        output.data_ptr() != left.data_ptr()
        or output.storage_offset() != left.storage_offset()
        or output.shape != left.shape
        or output.stride() != left.stride()
    ):
        raise ValueError("in-place output must alias the complete left tensor exactly")

    left_range = _storage_range(left)
    right_range = _storage_range(right)
    if _ranges_overlap(left_range, right_range):
        raise ValueError("source and destination staged storage must not overlap")


class DeviceSumExecutor:
    """Synchronous device-only SUM primitive for the generic CCL data plane.

    The caller must pass a private collective workspace as ``destination``.
    This primitive guarantees the arithmetic and alias contract for one
    synchronous launch; it does not make a multi-step collective or driver
    cleanup failure atomic.  A public collective commits a fresh output only
    after every carrier step and device launch has completed successfully.
    """

    def __init__(self) -> None:
        target = triton.runtime.driver.active.get_current_target()
        device = triton.runtime.driver.active.get_active_torch_device()
        if target.backend != "gemsim_amd" or target.arch != "gfx950":
            raise RuntimeError(f"unexpected Triton execution target: {target}")
        if device.type != "cpu":
            raise RuntimeError(f"gemsim_amd must expose CPU staging, got {device}")
        self._launch_count = 0

    @property
    def counters(self) -> DeviceSumCounters:
        return DeviceSumCounters(
            device_reduction_launch_count=self._launch_count,
            host_reduction_count=0,
        )

    def sum_in_place(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        element_count: int | None = None,
    ) -> torch.Tensor:
        count = destination.numel() if element_count is None else element_count
        _validate_operands(destination, source, destination, count)
        if count == 0:
            return destination
        # count <= INT32_MAX proves the last int32 offset
        # floor((count - 1) / 256) * 256 + 255 cannot overflow.
        grid = (triton.cdiv(count, _BLOCK_SIZE),)
        _sum_kernel[grid](
            destination,
            source,
            destination,
            count,
            IS_BF16=destination.dtype == torch.bfloat16,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
        )
        self._launch_count += 1
        return destination
