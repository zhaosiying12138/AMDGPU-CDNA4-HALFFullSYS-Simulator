"""Host-only tests for fail-closed Qwen3.5 per-layer checkpoints."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen35_layer_checkpoint_for_tests",
    ROOT / "examples/triton/_qwen35_layer_checkpoint.py",
)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


class Qwen35LayerCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()

    def tearDown(self) -> None:
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)

    @staticmethod
    def _identity(layer_count: int = 4) -> dict:
        layer_types = [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(layer_count)
        ]
        return {
            "model": {
                "id": "Qwen/Qwen3.5-0.8B",
                "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "artifacts": {
                    "config.json": {"bytes": 2907, "sha256": "1" * 64},
                    "model.safetensors": {"bytes": 1024, "sha256": "2" * 64},
                },
            },
            "implementation": {
                "architecture": "GemsimQwen3_5ForCausalLM",
                "runner_sha256": "3" * 64,
                "plugin_sha256": "4" * 64,
                "vllm_git_head": "5" * 40,
                "vllm_tree_sha256": "6" * 64,
                "gem5_binary_sha256": "7" * 64,
                "runtime_dso_sha256": "8" * 64,
                "prefix_manifest_sha256": "9" * 64,
            },
            "target": {
                "backend": "gemsim_amd",
                "arch": "gfx950",
                "device": "cpu",
                "fallback_allowed": False,
                "stochastic_ops": False,
            },
            "decoder": {
                "layer_count": layer_count,
                "hidden_size": 5,
                "activation_dtype": "torch.bfloat16",
                "layer_types": layer_types,
                "gdn_conv_state_shape": [1, 2, 3],
                "gdn_recurrent_state_shape": [1, 2, 2, 2],
                "kv_cache_shape": [1, 4, 2, 4],
                "gdn_conv_state_dtype": "torch.bfloat16",
                "gdn_recurrent_state_dtype": "torch.float32",
                "kv_cache_dtype": "torch.bfloat16",
            },
            "parallelism": {
                "world_size": 1,
                "rank": 0,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "weights": {
                "checkpoint_tensor_count": 488,
                "source_tensor_count": 320,
                "loaded_tensor_count": 320,
                "source_names_sha256": "a" * 64,
            },
        }

    @staticmethod
    def _request() -> dict:
        return {
            "sequence_id": "request-17",
            "phase": "prefill",
            "step_index": 0,
            "context_length_before": 0,
            "context_length_after": 2,
            "scheduler": {
                "metadata_sha256": "b" * 64,
                "slot_mapping_sha256": "c" * 64,
                "cache_block_ids": [0],
            },
        }

    @staticmethod
    def _lineage(after_layer: int) -> dict:
        return {
            "run_id": "run-20260812-17",
            "previous_after_layer": None if after_layer == -1 else after_layer - 1,
            "previous_manifest_sha256": None if after_layer == -1 else "d" * 64,
        }

    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor([248044, 266], dtype=torch.int64)
        positions = torch.tensor([0, 1], dtype=torch.int64)
        hidden = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(torch.bfloat16)
        residual = (hidden.float() + 0.5).to(torch.bfloat16)
        return input_ids, positions, hidden, residual

    @staticmethod
    def _caches(identity: dict, *, fill: float = 0.0) -> list[tuple]:
        records = []
        for index, layer_type in enumerate(identity["decoder"]["layer_types"]):
            module_identity = f"model.layers.{index}.attention"
            if layer_type == "linear_attention":
                records.append(
                    (
                        module_identity,
                        torch.full((1, 2, 3), fill + index, dtype=torch.bfloat16),
                        torch.full((1, 2, 2, 2), fill + index, dtype=torch.float32),
                    )
                )
            else:
                records.append(
                    (
                        module_identity,
                        torch.full((1, 4, 2, 4), fill + index, dtype=torch.bfloat16),
                    )
                )
        return records

    def _publish(
        self,
        output: Path,
        *,
        identity: dict,
        caches: list[tuple],
        after_layer: int = 0,
        generator: torch.Generator | None = None,
    ) -> dict:
        input_ids, positions, hidden, residual = self._inputs()
        return checkpoint.publish_layer_checkpoint(
            output,
            identity=identity,
            request_identity=self._request(),
            lineage=self._lineage(after_layer),
            after_layer=after_layer,
            hidden_states=hidden,
            residual=residual,
            input_ids=input_ids,
            positions=positions,
            caches=caches,
            named_generators={} if generator is None else {"sampler": generator},
        )

    def test_atomic_round_trip_restores_all_state_and_rng(self) -> None:
        identity = self._identity()
        source_caches = self._caches(identity, fill=1.0)
        target_caches = self._caches(identity, fill=100.0)
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        generator = torch.Generator().manual_seed(14)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            publication = self._publish(
                output,
                identity=identity,
                caches=source_caches,
                generator=generator,
            )
            self.assertEqual(
                set(entry.name for entry in output.iterdir()),
                {"manifest.json", "state.safetensors"},
            )
            expected_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=generator),
            )
            random.seed(101)
            np.random.seed(102)
            torch.manual_seed(103)
            generator.manual_seed(104)

            input_ids, positions, hidden, residual = self._inputs()
            restored = checkpoint.restore_layer_checkpoint(
                output,
                expected_identity=identity,
                expected_request_identity=self._request(),
                expected_input_ids=input_ids,
                expected_positions=positions,
                expected_after_layer=0,
                caches=target_caches,
                named_generators={"sampler": generator},
                expected_manifest_sha256=publication["manifest_sha256"],
            )
            self.assertTrue(torch.equal(restored.hidden_states, hidden))
            self.assertTrue(torch.equal(restored.residual, residual))
            self.assertEqual(restored.next_layer, 1)
            self.assertEqual(restored.result["resume_action"], "decoder_layers_then_final_norm")
            for source, target in zip(source_caches, target_caches):
                self.assertEqual(source[0], target[0])
                for source_tensor, target_tensor in zip(source[1:], target[1:]):
                    self.assertTrue(torch.equal(source_tensor, target_tensor))
            observed_rng = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=generator),
            )
            self.assertEqual(observed_rng[0], expected_rng[0])
            self.assertEqual(observed_rng[1], expected_rng[1])
            self.assertTrue(torch.equal(observed_rng[2], expected_rng[2]))
            self.assertTrue(torch.equal(observed_rng[3], expected_rng[3]))
            self.assertTrue(restored.result["strict_validation_passed"])
            self.assertTrue(publication["atomic_publish"])

    def test_implementation_identity_drift_is_rejected_before_mutation(self) -> None:
        identity = self._identity()
        source_caches = self._caches(identity, fill=1.0)
        target_caches = self._caches(identity, fill=100.0)
        target_before = [
            tuple([record[0], *(tensor.clone() for tensor in record[1:])])
            for record in target_caches
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            self._publish(output, identity=identity, caches=source_caches)
            drifted = copy.deepcopy(identity)
            drifted["implementation"]["gem5_binary_sha256"] = "e" * 64
            input_ids, positions, _hidden, _residual = self._inputs()
            torch.manual_seed(177)
            rng_before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError,
                "model/runtime identity mismatch",
            ):
                checkpoint.restore_layer_checkpoint(
                    output,
                    expected_identity=drifted,
                    expected_request_identity=self._request(),
                    expected_input_ids=input_ids,
                    expected_positions=positions,
                    expected_after_layer=0,
                    caches=target_caches,
                )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            for before, after in zip(target_before, target_caches):
                for before_tensor, after_tensor in zip(before[1:], after[1:]):
                    self.assertTrue(torch.equal(before_tensor, after_tensor))

    def test_tampered_state_is_rejected(self) -> None:
        identity = self._identity()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            self._publish(output, identity=identity, caches=self._caches(identity))
            state_path = output / "state.safetensors"
            value = bytearray(state_path.read_bytes())
            value[-1] ^= 1
            state_path.write_bytes(value)
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError, "state file SHA-256 mismatch"
            ):
                checkpoint.load_layer_checkpoint(output)

    def test_existing_destination_is_never_overwritten(self) -> None:
        identity = self._identity()
        source_caches = self._caches(identity)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            publication = self._publish(
                output, identity=identity, caches=source_caches
            )
            manifest_before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self._publish(output, identity=identity, caches=source_caches)
            self.assertEqual((output / "manifest.json").read_bytes(), manifest_before)
            self.assertEqual(
                publication["manifest_sha256"],
                checkpoint.load_layer_checkpoint(output).manifest_sha256,
            )

    def test_layer_23_resumes_only_at_final_norm(self) -> None:
        identity = self._identity(layer_count=24)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-23"
            publication = self._publish(
                output,
                identity=identity,
                caches=self._caches(identity),
                after_layer=23,
            )
            loaded = checkpoint.load_layer_checkpoint(output)
            self.assertEqual(publication["next_layer"], 24)
            self.assertEqual(publication["remaining_decoder_layer_count"], 0)
            self.assertEqual(publication["resume_action"], "final_norm")
            self.assertEqual(loaded.manifest["boundary"]["next_layer"], 24)
            self.assertEqual(
                loaded.manifest["boundary"]["resume_action"], "final_norm"
            )

    def test_initial_boundary_restores_layer_zero_inputs_with_absent_residual(self) -> None:
        identity = self._identity()
        source_caches = self._caches(identity, fill=1.0)
        target_caches = self._caches(identity, fill=100.0)
        input_ids, positions, hidden, _residual = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-before-00"
            publication = checkpoint.publish_layer_checkpoint(
                output,
                identity=identity,
                request_identity=self._request(),
                lineage=self._lineage(-1),
                after_layer=-1,
                hidden_states=hidden,
                residual=None,
                input_ids=input_ids,
                positions=positions,
                caches=source_caches,
            )
            restored = checkpoint.restore_layer_checkpoint(
                output,
                expected_identity=identity,
                expected_request_identity=self._request(),
                expected_input_ids=input_ids,
                expected_positions=positions,
                expected_after_layer=-1,
                caches=target_caches,
            )
            self.assertEqual(publication["after_layer"], -1)
            self.assertEqual(publication["next_layer"], 0)
            self.assertFalse(publication["residual_present"])
            self.assertEqual(
                publication["resume_action"], "decoder_layers_then_final_norm"
            )
            self.assertEqual(restored.next_layer, 0)
            self.assertIsNone(restored.residual)
            self.assertTrue(torch.equal(restored.hidden_states, hidden))

    def test_after_layer_minus_two_is_rejected(self) -> None:
        identity = self._identity()
        input_ids, positions, hidden, _residual = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "invalid-layer"
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError, r"must be an integer in \[-1,3\]"
            ):
                checkpoint.publish_layer_checkpoint(
                    output,
                    identity=identity,
                    request_identity=self._request(),
                    lineage={
                        "run_id": "run-20260812-17",
                        "previous_after_layer": None,
                        "previous_manifest_sha256": None,
                    },
                    after_layer=-2,
                    hidden_states=hidden,
                    residual=None,
                    input_ids=input_ids,
                    positions=positions,
                    caches=self._caches(identity),
                )
            self.assertFalse(output.exists())

    def test_layer_zero_checkpoint_cannot_break_initial_lineage(self) -> None:
        identity = self._identity()
        input_ids, positions, hidden, residual = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError,
                "immediately preceding layer",
            ):
                checkpoint.publish_layer_checkpoint(
                    output,
                    identity=identity,
                    request_identity=self._request(),
                    lineage={
                        "run_id": "run-20260812-17",
                        "previous_after_layer": None,
                        "previous_manifest_sha256": None,
                    },
                    after_layer=0,
                    hidden_states=hidden,
                    residual=residual,
                    input_ids=input_ids,
                    positions=positions,
                    caches=self._caches(identity),
                )
            self.assertFalse(output.exists())

    def test_nonfinite_hidden_is_rejected_without_publishing(self) -> None:
        identity = self._identity()
        input_ids, positions, hidden, residual = self._inputs()
        hidden[0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError, "nonfinite"
            ):
                checkpoint.publish_layer_checkpoint(
                    output,
                    identity=identity,
                    request_identity=self._request(),
                    lineage=self._lineage(0),
                    after_layer=0,
                    hidden_states=hidden,
                    residual=residual,
                    input_ids=input_ids,
                    positions=positions,
                    caches=self._caches(identity),
                )
            self.assertFalse(output.exists())

    def test_manifest_with_extra_field_is_rejected(self) -> None:
        identity = self._identity()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layer-00"
            self._publish(output, identity=identity, caches=self._caches(identity))
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["untrusted_extension"] = True
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                checkpoint.LayerCheckpointError, "manifest keys mismatch"
            ):
                checkpoint.load_layer_checkpoint(output)


if __name__ == "__main__":
    unittest.main()
