"""Focused tests for the formal out-of-tree framework plugin boundary."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = ROOT / "plugins/framework/gemsim_vllm/src"
sys.path.insert(0, str(PLUGIN_SRC))

# vLLM caches its platform during the first import. Import under the explicit
# product environment without leaking test-only values into later test modules.
with mock.patch.dict(
    os.environ,
    {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
):
    import gemsim_vllm  # noqa: E402


class GemsimVllmPluginTest(unittest.TestCase):
    def test_attention_gate_preserves_upstream_bf16_boundary(self) -> None:
        attention = torch.tensor([-127.0], dtype=torch.bfloat16)
        gate = torch.tensor([-12.0], dtype=torch.bfloat16)

        # qwen3_next.py evaluates torch.sigmoid on a BF16 gate tensor, so that
        # result is stored as BF16 before the following BF16 multiplication.
        sigmoid_bf16 = torch.sigmoid(gate)
        upstream = (
            attention.to(torch.float32) * sigmoid_bf16.to(torch.float32)
        ).to(torch.bfloat16)
        fused_fp32 = (
            attention.to(torch.float32)
            * torch.sigmoid(gate.to(torch.float32))
        ).to(torch.bfloat16)

        self.assertEqual(sigmoid_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(upstream, torch.tensor(
            [-0.0007781982421875], dtype=torch.bfloat16
        )))
        self.assertTrue(torch.equal(fused_fp32, torch.tensor(
            [-0.000782012939453125], dtype=torch.bfloat16
        )))
        self.assertFalse(torch.equal(upstream, fused_fp32))

    def test_platform_activation_is_explicit(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(gemsim_vllm.platform_plugin())
        with mock.patch.dict(
            os.environ,
            {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
            clear=True,
        ):
            self.assertEqual(
                gemsim_vllm.platform_plugin(),
                "gemsim_vllm.platform.GemsimPlatform",
            )

    def test_general_registration_is_idempotent(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
            clear=True,
        ):
            gemsim_vllm.register_ops()
            gemsim_vllm.register_ops()
            from vllm import ModelRegistry

            self.assertIn(
                "GemsimQwen3_5ForCausalLM",
                ModelRegistry.get_supported_archs(),
            )
            for name in (
                "dense_linear",
                "embedding",
                "fused_add_gemma_rms_norm",
                "gdn_conv_decode",
                "gdn_recurrent_decode",
                "gemma_rms_norm",
                "rotary_embedding",
                "rms_norm_gated",
                "sigmoid_output_gate",
                "silu_and_mul",
            ):
                self.assertTrue(hasattr(torch.ops.gemsim, name), name)

    def test_registration_does_not_modify_vllm_coordinator_methods(self) -> None:
        from vllm.distributed.parallel_state import GroupCoordinator

        methods = (
            GroupCoordinator.broadcast,
            GroupCoordinator.broadcast_tensor_dict,
        )
        with mock.patch.dict(
            os.environ,
            {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
            clear=True,
        ):
            gemsim_vllm.register_ops()
            gemsim_vllm.register_ops()
        self.assertIs(GroupCoordinator.broadcast, methods[0])
        self.assertIs(GroupCoordinator.broadcast_tensor_dict, methods[1])

    def test_platform_preserves_upstream_compile_and_rejects_graphs(self) -> None:
        from gemsim_vllm.platform import GemsimPlatform
        from vllm.config import CUDAGraphMode, CompilationMode

        compilation = SimpleNamespace(
            custom_ops=[],
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[1, 2, 4],
        )
        parallel = SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_elastic_ep=False,
            enable_eplb=False,
            enable_dbo=False,
            nnodes=1,
            node_rank=0,
        )
        config = SimpleNamespace(
            compilation_config=compilation,
            parallel_config=parallel,
            model_config=SimpleNamespace(is_moe=False),
            speculative_config=None,
        )
        compilation.cudagraph_mode = CUDAGraphMode.NONE
        compilation.cudagraph_capture_sizes = []
        GemsimPlatform.check_and_update_config(config)

        self.assertEqual(compilation.custom_ops, [])
        self.assertEqual(compilation.mode, CompilationMode.VLLM_COMPILE)
        self.assertEqual(compilation.backend, "inductor")
        self.assertEqual(compilation.cudagraph_mode, CUDAGraphMode.NONE)
        self.assertEqual(compilation.cudagraph_capture_sizes, [])
        self.assertEqual(GemsimPlatform.simple_compile_backend, "inductor")
        self.assertEqual(GemsimPlatform.dist_backend, "gloo")
        self.assertEqual(GemsimPlatform.device_name, "cpu")
        self.assertEqual(
            GemsimPlatform.get_device_communicator_cls(),
            "gemsim_vllm.communicator.GemsimDeviceCommunicator",
        )
        self.assertEqual(GemsimPlatform.get_device_uuid(15), "gemsim-gfx950-rank15")
        with self.assertRaises(ValueError):
            GemsimPlatform.get_device_uuid(16)
        self.assertIsNone(GemsimPlatform.import_kernels())

        parallel.pipeline_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "pure tensor parallelism"):
            GemsimPlatform.check_and_update_config(config)
        parallel.pipeline_parallel_size = 1
        compilation.cudagraph_mode = CUDAGraphMode.FULL
        compilation.cudagraph_capture_sizes = [1, 2, 4]
        with self.assertRaisesRegex(ValueError, "CUDA Graph capture"):
            GemsimPlatform.check_and_update_config(config)
        parallel.nnodes = 2
        with self.assertRaisesRegex(ValueError, "single node"):
            GemsimPlatform.check_and_update_config(config)

    def test_attention_construction_backend_is_explicit(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
            clear=True,
        ):
            # Import vLLM first so its plugin discovery completes before the
            # plugin imports vllm.platforms.interface (avoids a package-cycle
            # during direct unit-test imports).
            import vllm

            from gemsim_vllm.platform import GemsimPlatform
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            selector = SimpleNamespace(
                use_mla=False,
                use_sparse=False,
                dtype=torch.bfloat16,
            )
            self.assertEqual(
                GemsimPlatform.get_attn_backend_cls(None, selector),
                "gemsim_vllm.attention.GemsimAttentionBackend",
            )

    def test_attention_impl_populates_upstream_output_buffer(self) -> None:
        from gemsim_vllm import attention as attention_module

        implementation = attention_module.GemsimAttentionImpl.__new__(
            attention_module.GemsimAttentionImpl
        )
        implementation.num_heads = 2
        implementation.num_kv_heads = 1
        implementation.head_size = 256
        implementation.scale = 0.0625
        query = torch.zeros((2, 2, 256), dtype=torch.bfloat16)
        key = torch.zeros((2, 1, 256), dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        cache = torch.zeros((1, 16, 1, 512), dtype=torch.bfloat16)
        output = torch.full_like(query, float("nan"))
        metadata = SimpleNamespace(
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
            num_decode_tokens=0,
            max_query_len=2,
            max_seq_len=2,
        )

        def write_output(*args):
            args[3].fill_(3.0)

        with mock.patch.object(
            attention_module, "_prefill_attention_op", side_effect=write_output
        ):
            returned = implementation.forward(
                SimpleNamespace(), query, key, value, cache, metadata, output
            )

        self.assertIs(returned, output)
        self.assertTrue(torch.equal(output, torch.full_like(output, 3.0)))

    def test_fake_tensor_shapes(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ROCM_SIM_ROOT": "/private/prefix", "TRITON_DEFAULT_BACKEND": "gemsim_amd"},
            clear=True,
        ):
            gemsim_vllm.register_ops()
        with FakeTensorMode():
            x = torch.empty((7, 1024), dtype=torch.bfloat16)
            weight = torch.empty((3584, 1024), dtype=torch.bfloat16)
            linear = torch.ops.gemsim.dense_linear(x, weight)
            self.assertEqual(linear.shape, (7, 3584))

            norm_weight = torch.empty((1024,), dtype=torch.bfloat16)
            norm = torch.ops.gemsim.gemma_rms_norm(x, norm_weight, 1.0e-6)
            self.assertEqual(norm.shape, x.shape)
            fused = torch.ops.gemsim.fused_add_gemma_rms_norm(
                x, torch.empty_like(x), norm_weight, 1.0e-6
            )
            self.assertEqual([value.shape for value in fused], [x.shape, x.shape])

            activated = torch.ops.gemsim.silu_and_mul(
                torch.empty((7, 7168), dtype=torch.bfloat16)
            )
            self.assertEqual(activated.shape, (7, 3584))
            embedded = torch.ops.gemsim.embedding(
                torch.empty((7,), dtype=torch.int64),
                torch.empty((32, 1024), dtype=torch.bfloat16),
            )
            self.assertEqual(embedded.shape, (7, 1024))
            rotated_q, rotated_k = torch.ops.gemsim.rotary_embedding(
                torch.empty((7,), dtype=torch.int64),
                torch.empty((7, 2048), dtype=torch.bfloat16),
                torch.empty((7, 512), dtype=torch.bfloat16),
                torch.empty((32, 64), dtype=torch.bfloat16),
                256,
                64,
                True,
            )
            self.assertEqual(rotated_q.shape, (7, 2048))
            self.assertEqual(rotated_k.shape, (7, 512))
            gated = torch.ops.gemsim.sigmoid_output_gate(
                torch.empty((7, 2048), dtype=torch.bfloat16),
                torch.empty((7, 2048), dtype=torch.bfloat16),
            )
            self.assertEqual(gated.shape, (7, 2048))
            gdn_norm = torch.ops.gemsim.rms_norm_gated(
                torch.empty((7, 16, 128), dtype=torch.bfloat16),
                torch.empty((7, 16, 128), dtype=torch.bfloat16),
                torch.empty((128,), dtype=torch.float32),
                1.0e-6,
            )
            self.assertEqual(gdn_norm.shape, (7, 16, 128))
            conv = torch.ops.gemsim.gdn_conv_decode(
                torch.empty((1, 6144), dtype=torch.bfloat16),
                torch.empty((6144, 4), dtype=torch.bfloat16),
                torch.empty((3, 3, 6144), dtype=torch.bfloat16),
                torch.empty((1,), dtype=torch.int32),
            )
            self.assertEqual(conv.shape, (1, 6144))
            recurrent = torch.ops.gemsim.gdn_recurrent_decode(
                conv,
                torch.empty((1, 16), dtype=torch.bfloat16),
                torch.empty((1, 16), dtype=torch.bfloat16),
                torch.empty((16,), dtype=torch.float32),
                torch.empty((16,), dtype=torch.bfloat16),
                torch.empty((3, 16, 128, 128), dtype=torch.float32),
                torch.empty((1,), dtype=torch.int32),
            )
            self.assertEqual(recurrent.shape, (1, 16, 128))

            prefill_conv = torch.ops.gemsim.gdn_conv_decode(
                torch.empty((7, 6144), dtype=torch.bfloat16),
                torch.empty((6144, 4), dtype=torch.bfloat16),
                torch.empty((3, 3, 6144), dtype=torch.bfloat16),
                torch.empty((1,), dtype=torch.int32),
            )
            self.assertEqual(prefill_conv.shape, (7, 6144))
            prefill_recurrent = torch.ops.gemsim.gdn_recurrent_decode(
                prefill_conv,
                torch.empty((7, 16), dtype=torch.bfloat16),
                torch.empty((7, 16), dtype=torch.bfloat16),
                torch.empty((16,), dtype=torch.float32),
                torch.empty((16,), dtype=torch.bfloat16),
                torch.empty((3, 16, 128, 128), dtype=torch.float32),
                torch.empty((1,), dtype=torch.int32),
            )
            self.assertEqual(prefill_recurrent.shape, (7, 16, 128))


if __name__ == "__main__":
    unittest.main()
