"""Host-only generic contracts and a separate Qwen RowParallel acceptance anchor."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = ROOT / "plugins/framework/gemsim_vllm/src"
sys.path.insert(0, str(PLUGIN_SRC))

from gemsim_vllm.row_parallel import (  # noqa: E402
    row_parallel_shard,
    validate_row_parallel_contract,
)


QWEN_HIDDEN_SIZE = 1024
QWEN_MLP_INTERMEDIATE_SIZE = 3584


def oracle_shard_row_parallel_weight(
    loaded_weight: torch.Tensor, *, tp_size: int, tp_rank: int
) -> torch.Tensor:
    shard = row_parallel_shard(
        loaded_weight.shape[1], tp_size=tp_size, tp_rank=tp_rank
    )
    return loaded_weight[:, shard.start : shard.stop].clone().contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


def dense_oracle(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.float().matmul(weight.float().t()).to(torch.bfloat16).contiguous()


@contextmanager
def model_parallel_world_one():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    descriptor, store_path = tempfile.mkstemp(prefix="gemsim-row-host-", suffix=".store")
    os.close(descriptor)
    initialized = False
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{store_path}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1, backend="gloo")
            initialized = True
            yield
    finally:
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
        Path(store_path).unlink(missing_ok=True)


class GemsimRowParallelContractTest(unittest.TestCase):
    def test_generic_shards_cover_2_through_16_without_overlap(self) -> None:
        for tp_size in range(1, 17):
            input_size = 240 * tp_size
            shards = [
                row_parallel_shard(input_size, tp_size=tp_size, tp_rank=rank)
                for rank in range(tp_size)
            ]
            self.assertEqual(shards[0].start, 0)
            self.assertEqual(shards[-1].stop, input_size)
            self.assertTrue(all(shard.size == 240 for shard in shards))
            self.assertTrue(
                all(left.stop == right.start for left, right in zip(shards, shards[1:]))
            )

        for input_size, tp_size, tp_rank in (
            (3584, 17, 0),
            (3584, 3, 0),
            (3584, 2, 2),
        ):
            with self.assertRaises(ValueError):
                row_parallel_shard(input_size, tp_size=tp_size, tp_rank=tp_rank)

    def test_qwen_mlp_tp2_oracle_ranges_reconstruct_exactly(self) -> None:
        full = torch.arange(
            QWEN_HIDDEN_SIZE * QWEN_MLP_INTERMEDIATE_SIZE,
            dtype=torch.int32,
        ).reshape(QWEN_HIDDEN_SIZE, QWEN_MLP_INTERMEDIATE_SIZE)
        ranges = [
            row_parallel_shard(3584, tp_size=2, tp_rank=rank) for rank in range(2)
        ]
        shards = [
            oracle_shard_row_parallel_weight(full, tp_size=2, tp_rank=rank)
            for rank in range(2)
        ]
        self.assertEqual([(value.start, value.stop) for value in ranges], [
            (0, 1792),
            (1792, 3584),
        ])
        self.assertEqual([tuple(value.shape) for value in shards], [(1024, 1792)] * 2)
        reconstructed = torch.cat(shards, dim=1)
        self.assertTrue(torch.equal(reconstructed, full))
        self.assertEqual(tensor_sha256(reconstructed), tensor_sha256(full))
        self.assertTrue(all(value.is_contiguous() for value in shards))
        self.assertTrue(
            all(
                value.untyped_storage().data_ptr()
                != full.untyped_storage().data_ptr()
                for value in shards
            )
        )

    def test_pinned_loader_and_oot_inheritance_remain_authoritative(self) -> None:
        try:
            from gemsim_vllm.adapters import GemsimRowParallelLinear
            from vllm.model_executor.layers.linear import RowParallelLinear
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        self.assertNotIn("forward", GemsimRowParallelLinear.__dict__)
        self.assertIs(GemsimRowParallelLinear.forward, RowParallelLinear.forward)
        self.assertIs(GemsimRowParallelLinear.weight_loader, RowParallelLinear.weight_loader)
        self.assertIs(
            GemsimRowParallelLinear.weight_loader_v2,
            RowParallelLinear.weight_loader_v2,
        )
        source = inspect.getsource(RowParallelLinear.weight_loader)
        self.assertIn("start_idx = self.tp_rank * shard_size", source)
        self.assertIn("loaded_weight.narrow(input_dim, start_idx, shard_size)", source)

    def test_real_world_one_uses_param_loader_and_auto_weights_loader(self) -> None:
        try:
            from gemsim_vllm.adapters import (
                GemsimRowParallelLinear,
                GemsimUnquantizedRowParallelMethod,
            )
            from vllm.model_executor.layers.linear import RowParallelLinear
            from vllm.model_executor.models.utils import AutoWeightsLoader
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        with model_parallel_world_one():
            layer = RowParallelLinear(
                16,
                8,
                bias=False,
                input_is_parallel=True,
                params_dtype=torch.bfloat16,
            )
            self.assertIsInstance(layer, GemsimRowParallelLinear)
            self.assertIsInstance(layer.quant_method, GemsimUnquantizedRowParallelMethod)
            loaded_weight = torch.arange(128, dtype=torch.float32).reshape(8, 16).to(
                torch.bfloat16
            )
            loaded = AutoWeightsLoader(layer).load_weights([("weight", loaded_weight)])
            self.assertEqual(loaded, {"weight"})
            self.assertTrue(torch.equal(layer.weight.detach(), loaded_weight))
            self.assertIs(layer.weight.weight_loader.__self__, layer)

    def test_real_pinned_forward_calls_local_method_then_fresh_all_reduce(self) -> None:
        try:
            from gemsim_vllm.adapters import GemsimUnquantizedRowParallelMethod
            from vllm.model_executor.layers.linear import RowParallelLinear
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        tokens = 2
        full_input = torch.zeros((tokens, 3584), dtype=torch.bfloat16)
        full_weight = torch.zeros((1024, 3584), dtype=torch.bfloat16)
        rows = torch.arange(1024)
        full_weight[rows, rows] = 0.5
        full_weight[rows, 1792 + rows] = 0.25
        full_input[:, :1024] = torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16)
        full_input[:, 1792:2816] = torch.tensor([[4.0], [8.0]], dtype=torch.bfloat16)
        inputs = [value.contiguous() for value in full_input.chunk(2, dim=-1)]
        weights = [
            oracle_shard_row_parallel_weight(full_weight, tp_size=2, tp_rank=rank)
            for rank in range(2)
        ]
        locals_ = [dense_oracle(inputs[rank], weights[rank]) for rank in range(2)]
        reduced = (locals_[0].float() + locals_[1].float()).to(torch.bfloat16)
        single_device = dense_oracle(full_input, full_weight)
        self.assertTrue(torch.equal(reduced, single_device))

        for rank in range(2):
            layer = SimpleNamespace(
                input_size=3584,
                output_size=1024,
                input_size_per_partition=1792,
                tp_size=2,
                tp_rank=rank,
                input_is_parallel=True,
                reduce_results=True,
                skip_bias_add=False,
                return_bias=True,
                bias=None,
                weight=weights[rank],
                quant_method=GemsimUnquantizedRowParallelMethod(),
            )
            before = inputs[rank].view(torch.uint8).clone()
            dense = mock.Mock(side_effect=dense_oracle)
            collective_inputs: list[torch.Tensor] = []

            def all_reduce(value: torch.Tensor) -> torch.Tensor:
                collective_inputs.append(value)
                return reduced.clone()

            with (
                mock.patch(
                    "gemsim_vllm.adapters.torch.ops.gemsim.dense_linear",
                    dense,
                    create=True,
                ),
                mock.patch(
                    "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
                    side_effect=all_reduce,
                ) as collective,
            ):
                output, output_bias = RowParallelLinear.forward(layer, inputs[rank])
            dense.assert_called_once()
            collective.assert_called_once()
            self.assertTrue(torch.equal(collective_inputs[0], locals_[rank]))
            self.assertTrue(torch.equal(output, single_device))
            self.assertIsNone(output_bias)
            self.assertTrue(torch.equal(inputs[rank].view(torch.uint8), before))
            self.assertNotEqual(
                output.untyped_storage().data_ptr(),
                collective_inputs[0].untyped_storage().data_ptr(),
            )

    def test_generic_contract_and_local_method_fail_closed(self) -> None:
        try:
            from gemsim_vllm.adapters import GemsimUnquantizedRowParallelMethod
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        layer = SimpleNamespace(
            input_size=3584,
            output_size=1024,
            input_size_per_partition=1792,
            tp_size=2,
            tp_rank=0,
            input_is_parallel=True,
            reduce_results=True,
            bias=torch.zeros(1024, dtype=torch.bfloat16),
            weight=torch.zeros((1024, 1792), dtype=torch.bfloat16),
        )
        method = GemsimUnquantizedRowParallelMethod()
        with self.assertRaises(NotImplementedError):
            method.apply(
                layer,
                torch.zeros((1, 1792), dtype=torch.bfloat16),
                bias=layer.bias,
            )
        with self.assertRaises(RuntimeError):
            validate_row_parallel_contract(
                input_size=3584,
                output_size=1024,
                input_size_per_partition=895,
                tp_size=4,
                tp_rank=0,
                weight=torch.empty((1024, 896), dtype=torch.bfloat16),
            )
        with self.assertRaises(ValueError):
            validate_row_parallel_contract(
                input_size=2048,
                output_size=1024,
                input_size_per_partition=1024,
                tp_size=2,
                tp_rank=0,
                weight=torch.empty((1024, 1023), dtype=torch.bfloat16),
            )

        deferred_bias_layer = SimpleNamespace(
            input_size=64,
            output_size=7,
            input_size_per_partition=4,
            tp_size=16,
            tp_rank=15,
            bias=torch.zeros(7, dtype=torch.bfloat16),
            weight=torch.zeros((7, 4), dtype=torch.bfloat16),
        )
        with mock.patch(
            "gemsim_vllm.adapters.torch.ops.gemsim.dense_linear",
            side_effect=dense_oracle,
            create=True,
        ):
            output = method.apply(
                deferred_bias_layer,
                torch.ones((3, 4), dtype=torch.bfloat16),
                bias=None,
            )
        self.assertEqual(tuple(output.shape), (3, 7))

    def test_upstream_forward_owns_split_no_reduce_and_deferred_bias(self) -> None:
        try:
            from vllm.model_executor.layers.linear import RowParallelLinear
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        full_input = torch.arange(24, dtype=torch.float32).reshape(2, 12).to(
            torch.bfloat16
        )
        local_output = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(
            torch.bfloat16
        )
        deferred_bias = torch.arange(5, dtype=torch.float32).to(torch.bfloat16)
        method = mock.Mock()
        method.apply.return_value = local_output
        layer = SimpleNamespace(
            input_is_parallel=False,
            tp_size=2,
            tp_rank=1,
            skip_bias_add=True,
            bias=deferred_bias,
            reduce_results=False,
            return_bias=True,
            quant_method=method,
        )
        with mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce"
        ) as collective:
            output, output_bias = RowParallelLinear.forward(layer, full_input)
        collective.assert_not_called()
        called_input, called_bias = method.apply.call_args.args[1:]
        self.assertTrue(torch.equal(called_input, full_input[:, 6:].contiguous()))
        self.assertTrue(called_input.is_contiguous())
        self.assertIsNone(called_bias)
        self.assertIs(output, local_output)
        self.assertIs(output_bias, deferred_bias)

    def test_upstream_forward_tp1_skips_collective_and_honors_return_type(self) -> None:
        try:
            from vllm.model_executor.layers.linear import RowParallelLinear
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        input_ = torch.ones((3, 4), dtype=torch.bfloat16)
        local_output = torch.zeros((3, 7), dtype=torch.bfloat16)
        method = mock.Mock()
        method.apply.return_value = local_output
        layer = SimpleNamespace(
            input_is_parallel=True,
            tp_size=1,
            tp_rank=0,
            skip_bias_add=False,
            bias=None,
            reduce_results=True,
            return_bias=False,
            quant_method=method,
        )
        with mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce"
        ) as collective:
            output = RowParallelLinear.forward(layer, input_)
        collective.assert_not_called()
        method.apply.assert_called_once_with(layer, input_, None)
        self.assertIs(output, local_output)

    def test_other_tp_adapters_remain_explicitly_unsupported(self) -> None:
        try:
            from gemsim_vllm.adapters import (
                GemsimColumnParallelLinear,
                GemsimMergedColumnParallelLinear,
                GemsimQKVParallelLinear,
                GemsimQwenGatedDeltaNetAttention,
                GemsimVocabParallelEmbedding,
            )
        except ModuleNotFoundError:
            self.skipTest("pinned vLLM is not installed in this Python environment")

        layer = SimpleNamespace(
            tp_size=2,
            quant_config=None,
            bias=None,
            return_bias=True,
            weight=torch.zeros((4, 2), dtype=torch.bfloat16),
        )
        input_ = torch.ones((3, 2), dtype=torch.bfloat16)
        for adapter in (
            GemsimColumnParallelLinear,
            GemsimMergedColumnParallelLinear,
            GemsimQKVParallelLinear,
        ):
            with self.assertRaises(NotImplementedError):
                adapter._gemsim_forward(layer, input_)
        with self.assertRaises(NotImplementedError):
            GemsimVocabParallelEmbedding.forward(layer, torch.ones(1, dtype=torch.long))
        with self.assertRaises(NotImplementedError):
            GemsimQwenGatedDeltaNetAttention.forward(layer, input_)


if __name__ == "__main__":
    unittest.main()
