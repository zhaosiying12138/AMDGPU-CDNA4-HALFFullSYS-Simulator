# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic source-contract tests for the Qwen3.5 text-only path."""

from __future__ import annotations

import importlib.util
import importlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen35_operator_manifest", ROOT / "tools" / "qwen35_operator_manifest.py"
)
assert SPEC and SPEC.loader
manifest_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_tool)
work_queue_tool = importlib.import_module("qwen35_operator_work_queue")


class Qwen35OperatorManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = manifest_tool.build_manifest()
        cls.smoke_spec = importlib.util.spec_from_file_location(
            "qwen35_operator_smoke", ROOT / "tools" / "qwen35_operator_smoke.py"
        )
        assert cls.smoke_spec and cls.smoke_spec.loader
        cls.smoke = importlib.util.module_from_spec(cls.smoke_spec)
        cls.smoke_spec.loader.exec_module(cls.smoke)

    def test_pinned_text_topology_is_explicit(self) -> None:
        model = self.manifest["model"]
        topology = self.manifest["topology"]
        self.assertEqual(model["id"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(model["revision"], "2fc06364715b967f1860aea9cf38778875588b17")
        self.assertEqual(model["text_model_type"], "qwen3_5_text")
        self.assertEqual(topology["num_hidden_layers"], 24)
        self.assertEqual(topology["layer_type_counts"], {"full_attention": 6, "linear_attention": 18})
        self.assertEqual(topology["linear_attention_layers"], [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22])
        self.assertEqual(topology["full_attention_layers"], [3, 7, 11, 15, 19, 23])

    def test_every_contract_has_source_evidence(self) -> None:
        summary = self.manifest["summary"]
        self.assertTrue(summary["static_source_ok"])
        self.assertEqual(summary["contract_count"], 15)
        self.assertTrue(all(item["static_source_ok"] for item in self.manifest["contracts"]))
        self.assertTrue(all(item["source_evidence"] for item in self.manifest["contracts"]))
        self.assertTrue(all(item["cpu_fallback_forbidden"] for item in self.manifest["contracts"]))
        self.assertTrue(all(item["nvidia_runtime_is_not_pass"] for item in self.manifest["contracts"]))
        self.assertTrue(
            all(
                item["completion_rule"] == "all_required_items_accepted"
                for item in self.manifest["contracts"]
            )
        )
        self.assertTrue(
            all(item["required_work_item_ids"] for item in self.manifest["contracts"])
        )

    def test_expected_triton_kernel_families_are_discovered(self) -> None:
        by_id = {item["id"]: item for item in self.manifest["contracts"]}
        self.assertIn("_causal_conv1d_fwd_kernel", by_id["gdn.conv.prefill"]["triton_symbols"])
        self.assertIn("_causal_conv1d_update_kernel", by_id["gdn.conv.decode"]["triton_symbols"])
        self.assertIn("chunk_scaled_dot_kkt_fwd_kernel", by_id["gdn.chunk_prefill"]["triton_symbols"])
        self.assertIn("fused_recurrent_gated_delta_rule_packed_decode_kernel", by_id["gdn.recurrent_decode"]["triton_symbols"])
        self.assertIn("fused_gdn_gating_kernel", by_id["gdn.auxiliary_triton_variants"]["triton_symbols"])
        self.assertIn("l2norm_fwd_kernel", by_id["gdn.auxiliary_triton_variants"]["triton_symbols"])
        self.assertIn("_fused_qk_rmsnorm_rope_gate_kernel", by_id["full_attention.qkv_qk_norm_rope"]["triton_symbols"])
        self.assertIn("_triton_mrope_forward", by_id["full_attention.qkv_qk_norm_rope"]["alternate_triton_symbols"])
        self.assertIn("_swiglustep_and_mul_kernel", by_id["mlp.gate_up_silu_down"]["alternate_triton_symbols"])
        full_attn = by_id["full_attention.kv_cache_attention"]["triton_symbols"]
        for symbol in (
            "kernel_paged_attention_2d",
            "_paged_kv_cache_offsets",
            "_fwd_kernel_alibi",
            "kernel_unified_attention",
            "reduce_segments",
            "softmax_step",
        ):
            self.assertIn(symbol, full_attn)
        counts = self.manifest["counts"]
        self.assertEqual(counts["configured_layer_count"], 24)
        self.assertEqual(counts["configured_layer_type_counts"], {
            "full_attention": 6,
            "linear_attention": 18,
        })
        self.assertEqual(counts["contract_count"], 15)
        self.assertEqual(counts["contract_static_source_ok_count"], 15)
        self.assertGreaterEqual(counts["triton_source_symbol_count"], 40)
        self.assertGreaterEqual(counts["custom_registration_count"], 30)

        backend = self.manifest["backend_support"]
        self.assertEqual(backend["target"]["backend"], "gemsim_amd")
        self.assertFalse(backend["target"]["counts_as_pass"])
        unsupported = {
            item["backend"]: item for item in self.manifest["unsupported_backends"]
        }
        for name in ("cpu", "cuda_nvidia", "gdn_flashinfer", "gdn_cutedsl"):
            self.assertIn(name, unsupported)
            self.assertFalse(unsupported[name]["counts_as_pass"])
        self.assertTrue(backend["external_unverified"])
        self.assertEqual(
            backend["unsupported_backend_names"],
            ["cpu", "cuda_nvidia", "gdn_flashinfer", "gdn_cutedsl"],
        )

    def test_manifest_is_reproducible_and_materialized(self) -> None:
        self.assertEqual(self.manifest, manifest_tool.build_manifest())
        materialized = json.loads(
            (ROOT / "tools" / "qwen35_operator_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(materialized, self.manifest)

    def test_work_queue_is_valid_but_no_complete_contract_is_accepted(self) -> None:
        self.assertEqual(self.manifest["schema"], work_queue_tool.SCHEMA)
        self.assertEqual(work_queue_tool.validate_manifest(self.manifest), [])
        summary = work_queue_tool.queue_summary(self.manifest)
        self.assertEqual(summary["contract_count"], 15)
        self.assertEqual(summary["work_item_count"], 32)
        self.assertEqual(summary["configured_work_item_count"], 2)
        self.assertEqual(summary["accepted_contract_count"], 0)
        self.assertEqual(summary["accepted_work_item_count"], 0)
        self.assertEqual(summary["ready_work_item_count"], 2)
        self.assertEqual(summary["result_errors"], [])
        self.assertFalse(summary["all_contracts_accepted"])

    def test_manifest_is_source_only_and_contains_no_preaccepted_results(self) -> None:
        items = self.manifest["work_items"]
        self.assertEqual(len(items), 32)
        self.assertTrue(all(item["required_for_contract"] for item in items))
        self.assertTrue(all(item["acceptance_role"] == "required" for item in items))
        self.assertTrue(
            all(item["runtime_evidence"] == work_queue_tool.runtime_evidence_policy()
                for item in items)
        )
        self.assertTrue(
            all(contract["partial_work_item_ids"] == []
                for contract in self.manifest["contracts"])
        )
        configured = [
            item for item in items if item["configuration_status"] == "configured"
        ]
        self.assertTrue(configured)
        self.assertTrue(
            all(
                item["kernel"]["code_objects"]
                == {"count": 1, "identity_policy": "recorded_sha256"}
                for item in configured
            )
        )

    def test_smoke_never_promotes_nvidia_or_cpu_to_pass(self) -> None:
        result = self.smoke.run_smoke()
        self.assertTrue(result["static_contract"]["passed"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["fallback_audit"]["cpu_fallback_counted_as_pass"])
        self.assertFalse(result["fallback_audit"]["nvidia_fallback_counted_as_pass"])
        self.assertEqual(result["runtime"]["status"], "work_queue_incomplete")
        self.assertTrue(result["runtime"]["source_only_manifest"])
        self.assertTrue(result["runtime"]["external_results_required"])
        self.assertEqual(result["work_queue"]["accepted_contract_count"], 0)
        self.assertEqual(result["work_queue"]["work_item_count"], 32)


if __name__ == "__main__":
    unittest.main()
