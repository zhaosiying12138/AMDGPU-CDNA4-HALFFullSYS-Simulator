# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed CPU tests for the Qwen3.5 per-layer oracle protocol."""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


ROOT = Path(__file__).resolve().parents[1]
FINAL_NORM_CHECKPOINT = (
    ROOT
    / "artifacts/qwen35-layer-diff/layer2023-resume-v1/checkpoint-after-layer-23"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_layer_oracle_protocol",
    ROOT / "examples/triton/_qwen35_layer_oracle_protocol.py",
)
assert SPEC and SPEC.loader
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def make_private_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="qwen35-layer-protocol-test-"))
    os.chmod(path, 0o700)
    return path


def remove_private_root(path: Path) -> None:
    for directory, subdirectories, filenames in os.walk(path, topdown=False):
        for filename in filenames:
            candidate = Path(directory) / filename
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)
        for subdirectory in subdirectories:
            candidate = Path(directory) / subdirectory
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        os.chmod(directory, 0o700)
    shutil.rmtree(path)


class Qwen35LayerOracleProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = protocol.current_layer_oracle_request_identity(Path(__file__))
        cls.oracle_identity = protocol.expected_local_oracle_identity()

    def setUp(self) -> None:
        self.private_root = make_private_root()

    def tearDown(self) -> None:
        remove_private_root(self.private_root)

    def publish_gdn_request(
        self,
        name: str,
        *,
        layer_index: int = 0,
        operation: str = "prefill_2",
        nonzero_state: bool = False,
    ) -> protocol.LayerOracleRequest:
        tokens = 2 if operation == "prefill_2" else 1
        conv = torch.zeros((1, 3, 6144), dtype=torch.bfloat16)
        recurrent = torch.zeros((1, 16, 128, 128), dtype=torch.float32)
        if nonzero_state:
            conv[0, :, :4] = torch.tensor(
                [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
                dtype=torch.bfloat16,
            )
            recurrent[0, 0, 0, :4] = torch.tensor([0.25, -0.5, 1.0, 2.0])
        return protocol.publish_layer_oracle_request(
            self.private_root / name,
            identity=self.identity,
            layer_index=layer_index,
            operation=operation,
            token_positions=[0, 1] if operation == "prefill_2" else [2],
            hidden_before=torch.arange(tokens * 1024, dtype=torch.float32)
            .reshape(tokens, 1024)
            .to(torch.bfloat16),
            residual_before=(
                None
                if layer_index == 0
                else torch.full((tokens, 1024), 0.5, dtype=torch.bfloat16)
            ),
            mutable_state_before={
                "conv_state": conv,
                "recurrent_state": recurrent,
            },
        )

    def publish_attention_request(
        self, name: str
    ) -> protocol.LayerOracleRequest:
        cache = torch.zeros((1, 16, 2, 512), dtype=torch.bfloat16)
        cache[:, :2] = 0.25
        return protocol.publish_layer_oracle_request(
            self.private_root / name,
            identity=self.identity,
            layer_index=3,
            operation="decode_1",
            token_positions=[2],
            hidden_before=torch.zeros((1, 1024), dtype=torch.bfloat16),
            residual_before=torch.ones((1, 1024), dtype=torch.bfloat16),
            mutable_state_before={"kv_cache": cache},
        )

    def publish_response(
        self,
        request: protocol.LayerOracleRequest,
        name: str,
        *,
        mutate_document=None,
        nonfinite: bool = False,
    ) -> Path:
        tokens = 2 if request.operation in ("prefill_2", "final_norm") else 1
        hidden = torch.full((tokens, 1024), 0.75, dtype=torch.bfloat16)
        if nonfinite:
            hidden[0, 0] = float("nan")
        if request.operation == "final_norm":
            tensors = {"final_hidden_after": hidden}
        else:
            tensors = {
                "hidden_after": hidden,
                "residual_after": torch.full(
                    (tokens, 1024), -0.25, dtype=torch.bfloat16
                ),
            }
        if request.operation != "final_norm" and request.layer_type == "linear_attention":
            conv = torch.zeros((1, 3, 6144), dtype=torch.bfloat16)
            conv[0, :, :4] = torch.tensor(
                [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
                dtype=torch.bfloat16,
            )
            recurrent = torch.zeros((1, 16, 128, 128), dtype=torch.float32)
            recurrent[0, 0, 0, :4] = torch.tensor([0.25, -0.5, 1.0, 2.0])
            tensors.update(
                {
                    "gdn_conv_state_after": conv,
                    "gdn_recurrent_state_after": recurrent,
                }
            )
        elif request.operation != "final_norm":
            cache = request.mutable_state_before["kv_cache"].clone()
            cache[:, 2] = 0.5
            tensors["full_attention_kv_cache_after"] = cache

        metadata = protocol._embedded_response_metadata(
            request.request_id, request.operation
        )
        tensor_bytes = save_safetensors(tensors, metadata=metadata)
        artifacts = {
            filename: {"expected": expected, "observed": expected}
            for filename, expected in protocol.PINNED_CHECKPOINT_ARTIFACTS.items()
        }
        document = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostic_only": True,
            "environment": {
                "cuda_runtime_version": "fixture",
                "cudnn_version": 0,
                "deterministic_algorithms": True,
                "float32_matmul_precision": "highest",
                "gpu": deepcopy(protocol.PINNED_GPU),
                "platform": "fixture",
                "python_executable": str(protocol.PINNED_ORACLE_PYTHON),
                "python_version": "fixture",
                "tf32_cudnn": False,
                "tf32_matmul": False,
                "torch_version": "fixture",
            },
            "input_package": {
                "directory": str(request.path),
                "package_sha256": request.package_sha256,
                "request_json_sha256": request.request_json_sha256,
                "tensor_file_sha256": request.payload_sha256,
            },
            "kind": (
                protocol.FINAL_NORM_RESPONSE_KIND
                if request.operation == "final_norm"
                else protocol.RESPONSE_KIND
            ),
            "layer_index": request.layer_index,
            "layer_type": request.layer_type,
            "operation": request.operation,
            "oracle": {
                "checkpoint": {
                    "artifacts": artifacts,
                    "directory": "/fixture/pinned-checkpoint",
                    "id": protocol.MODEL_ID,
                    "index_total_size": 1746882752,
                    "revision": protocol.MODEL_REVISION,
                    "shard": protocol.MODEL_SHARD,
                    "shard_manifest": deepcopy(
                        protocol.PINNED_CHECKPOINT_ARTIFACTS[protocol.MODEL_SHARD]
                    ),
                },
                "execution_boundary": deepcopy(protocol.EXECUTION_BOUNDARY),
                "formula": deepcopy(request.identities["formula"]),
                "plugin_identity_from_request": deepcopy(
                    request.identities["plugin"]
                ),
                "request_identities": deepcopy(request.identities),
                "runner_identity_from_request": deepcopy(
                    request.identities["runner"]
                ),
                "script": deepcopy(self.oracle_identity["script"]),
            },
            "payload": {
                "bytes": len(tensor_bytes),
                "embedded_metadata": metadata,
                "filename": protocol.TENSOR_FILENAME,
                "sha256": protocol._sha256_bytes(tensor_bytes),
            },
            "request_id": request.request_id,
            "result_roles": list(tensors),
            "schema": protocol.RESPONSE_SCHEMA,
            "state_snapshot": (
                "after_final_norm"
                if request.operation == "final_norm"
                else "after_layer"
            ),
            "target_feedback": protocol.TARGET_FEEDBACK_POLICY,
            "tensors": {
                key: protocol._tensor_descriptor(value)
                for key, value in tensors.items()
            },
            "timing": {
                "compute_seconds": 0.01,
                "model_checkpoint_validate_and_load_seconds": 1.0,
            },
            "token_positions": list(request.token_positions),
        }
        if mutate_document is not None:
            mutate_document(document)
        response_bytes = protocol._canonical_json(document) + b"\n"
        return protocol._publish_package(
            self.private_root / name,
            json_filename=protocol.RESPONSE_JSON_FILENAME,
            json_bytes=response_bytes,
            tensor_bytes=tensor_bytes,
        )

    def rewrite_request_document(self, request: protocol.LayerOracleRequest, mutate) -> None:
        package = request.path
        os.chmod(package, 0o700)
        json_path = package / protocol.REQUEST_JSON_FILENAME
        os.chmod(json_path, 0o600)
        document = json.loads(json_path.read_text(encoding="ascii"))
        mutate(document)
        json_path.write_bytes(protocol._canonical_json(document) + b"\n")
        os.chmod(json_path, 0o400)
        os.chmod(package, 0o500)

    def test_gdn_request_roundtrip_preserves_formal_sd_conv_layout(self) -> None:
        first = self.publish_gdn_request("request-a")
        second = self.publish_gdn_request("request-b")
        self.assertEqual(first.request_id, protocol.derive_request_id(first.document))
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(stat.S_IMODE(first.path.stat().st_mode), 0o500)
        self.assertEqual(
            {entry.name for entry in os.scandir(first.path)},
            {protocol.REQUEST_JSON_FILENAME, protocol.TENSOR_FILENAME},
        )
        raw = load_safetensors((first.path / protocol.TENSOR_FILENAME).read_bytes())
        self.assertEqual(tuple(raw["gdn_conv_state_before"].shape), (1, 3, 6144))
        self.assertEqual(
            tuple(first.mutable_state_before["conv_state"].shape), (1, 3, 6144)
        )

        live = self.publish_gdn_request(
            "request-live", layer_index=1, operation="decode_1", nonzero_state=True
        )
        raw_live = load_safetensors(
            (live.path / protocol.TENSOR_FILENAME).read_bytes()
        )
        self.assertTrue(
            torch.equal(
                raw_live["gdn_conv_state_before"],
                live.mutable_state_before["conv_state"],
            )
        )

    def test_attention_request_and_both_response_state_contracts(self) -> None:
        attention_request = self.publish_attention_request("attention-request")
        self.assertEqual(
            tuple(attention_request.mutable_state_before["kv_cache"].shape),
            (1, 16, 2, 512),
        )
        attention_response = protocol.load_layer_oracle_response(
            self.publish_response(attention_request, "attention-response"),
            request=attention_request,
            expected_oracle_identity=self.oracle_identity,
        )
        self.assertEqual(
            tuple(attention_response.mutable_state_after["kv_cache"].shape),
            (1, 16, 2, 512),
        )
        self.assertIsNone(attention_response.execution_record)

        gdn_request = self.publish_gdn_request("gdn-request")
        gdn_response = protocol.load_layer_oracle_response(
            self.publish_response(gdn_request, "gdn-response"),
            request=gdn_request,
            expected_oracle_identity=self.oracle_identity,
        )
        self.assertEqual(
            set(gdn_response.mutable_state_after), {"conv_state", "recurrent_state"}
        )
        self.assertEqual(
            tuple(gdn_response.mutable_state_after["conv_state"].shape),
            (1, 3, 6144),
        )
        self.assertEqual(
            gdn_response.mutable_state_after["conv_state"][0, :, :4]
            .float()
            .tolist(),
            [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]],
        )

    @unittest.skipUnless(
        FINAL_NORM_CHECKPOINT.is_dir(),
        "post-layer-23 checkpoint is unavailable",
    )
    def test_final_norm_request_and_response_roundtrip(self) -> None:
        request = protocol.publish_final_norm_oracle_request(
            self.private_root / "final-norm-request",
            identity=self.identity,
            source_checkpoint=FINAL_NORM_CHECKPOINT,
        )
        self.assertEqual(request.operation, "final_norm")
        self.assertEqual(request.layer_index, 23)
        self.assertEqual(request.layer_type, "final_norm")
        self.assertEqual(request.token_positions, (0, 1))
        self.assertEqual(tuple(request.hidden_before.shape), (2, 1024))
        self.assertEqual(tuple(request.residual_before.shape), (2, 1024))
        self.assertEqual(request.mutable_state_before, {})
        self.assertEqual(
            request.document["tensor_roles"],
            ["hidden_before", "residual_before"],
        )
        self.assertEqual(
            set(request.document["source_checkpoint"]),
            protocol._SOURCE_CHECKPOINT_KEYS,
        )
        self.assertEqual(request.request_id, protocol.derive_request_id(request.document))

        changed_binding = deepcopy(request.document)
        changed_binding["source_checkpoint"]["state_sha256"] = "0" * 64
        self.assertNotEqual(
            request.request_id,
            protocol.derive_request_id(changed_binding),
        )

        response = protocol.load_layer_oracle_response(
            self.publish_response(request, "final-norm-response"),
            request=request,
            expected_oracle_identity=self.oracle_identity,
        )
        self.assertIsNone(response.hidden_after)
        self.assertIsNone(response.residual_after)
        self.assertEqual(tuple(response.final_hidden_after.shape), (2, 1024))
        self.assertEqual(response.mutable_state_after, {})
        self.assertEqual(response.document["result_roles"], ["final_hidden_after"])
        self.assertEqual(response.document["state_snapshot"], "after_final_norm")

        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError,
            "only valid after layer 23",
        ):
            protocol.publish_layer_oracle_request(
                self.private_root / "wrong-final-norm-layer",
                identity=self.identity,
                layer_index=22,
                operation="final_norm",
                token_positions=[0, 1],
                hidden_before=request.hidden_before,
                residual_before=request.residual_before,
                mutable_state_before={},
                source_checkpoint=FINAL_NORM_CHECKPOINT,
            )

    def test_request_tamper_and_request_id_rederivation_are_rejected(self) -> None:
        request = self.publish_gdn_request("payload-tamper")
        os.chmod(request.path, 0o700)
        payload = request.path / protocol.TENSOR_FILENAME
        os.chmod(payload, 0o600)
        content = bytearray(payload.read_bytes())
        content[-1] ^= 1
        payload.write_bytes(content)
        os.chmod(payload, 0o400)
        os.chmod(request.path, 0o500)
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "payload hash mismatch"
        ):
            protocol.load_layer_oracle_request(request.path)

        request_id = self.publish_gdn_request("request-id-tamper")
        self.rewrite_request_document(
            request_id,
            lambda document: document.__setitem__("request_id", "0" * 64),
        )
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "request_id derivation mismatch"
        ):
            protocol.load_layer_oracle_request(request_id.path)

    def test_response_exact_bindings_and_nonfinite_are_rejected(self) -> None:
        mutations = {
            "request-id": lambda value: value.__setitem__("request_id", "0" * 64),
            "package": lambda value: value["input_package"].__setitem__(
                "package_sha256", "0" * 64
            ),
            "layer": lambda value: value.__setitem__("layer_index", 1),
            "operation": lambda value: value.__setitem__("operation", "decode_1"),
            "positions": lambda value: value.__setitem__("token_positions", [2]),
            "identities": lambda value: value["oracle"]["request_identities"][
                "model"
            ].__setitem__("num_layers", 23),
            "target-feedback": lambda value: value.__setitem__(
                "target_feedback", "allowed"
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                request = self.publish_gdn_request(f"binding-request-{name}")
                response = self.publish_response(
                    request, f"binding-response-{name}", mutate_document=mutation
                )
                with self.assertRaises(protocol.LayerOracleProtocolError):
                    protocol.load_layer_oracle_response(
                        response,
                        request=request,
                        expected_oracle_identity=self.oracle_identity,
                    )

        request = self.publish_gdn_request("nonfinite-response-request")
        response = self.publish_response(
            request, "nonfinite-response", nonfinite=True
        )
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "nonfinite"
        ):
            protocol.load_layer_oracle_response(
                response,
                request=request,
                expected_oracle_identity=self.oracle_identity,
            )
        bad_hidden = torch.zeros((2, 1024), dtype=torch.bfloat16)
        bad_hidden[0, 0] = float("inf")
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "nonfinite"
        ):
            protocol.publish_layer_oracle_request(
                self.private_root / "nonfinite-request",
                identity=self.identity,
                layer_index=0,
                operation="prefill_2",
                token_positions=[0, 1],
                hidden_before=bad_hidden,
                residual_before=None,
                mutable_state_before={
                    "conv_state": torch.zeros(
                        (1, 6144, 3), dtype=torch.bfloat16
                    ),
                    "recurrent_state": torch.zeros(
                        (1, 16, 128, 128), dtype=torch.float32
                    ),
                },
            )

    def test_publish_never_overwrites_and_read_rejects_non_regular_entries(self) -> None:
        request = self.publish_gdn_request("no-overwrite")
        original = {
            name: (request.path / name).read_bytes()
            for name in (protocol.REQUEST_JSON_FILENAME, protocol.TENSOR_FILENAME)
        }
        with self.assertRaises(FileExistsError):
            self.publish_gdn_request("no-overwrite")
        self.assertEqual(
            original,
            {name: (request.path / name).read_bytes() for name in original},
        )

        extra = self.publish_gdn_request("extra-entry")
        os.chmod(extra.path, 0o700)
        (extra.path / "extra").write_text("forbidden", encoding="ascii")
        os.chmod(extra.path / "extra", 0o400)
        os.chmod(extra.path, 0o500)
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "exactly two"
        ):
            protocol.load_layer_oracle_request(extra.path)

        linked = self.publish_gdn_request("symlink-entry")
        original_json = linked.path / protocol.REQUEST_JSON_FILENAME
        backup = self.private_root / "request-backup.json"
        shutil.copyfile(original_json, backup)
        os.chmod(backup, 0o400)
        os.chmod(linked.path, 0o700)
        original_json.unlink()
        original_json.symlink_to(backup)
        os.chmod(linked.path, 0o500)
        with self.assertRaisesRegex(
            protocol.LayerOracleProtocolError, "regular files"
        ):
            protocol.load_layer_oracle_request(linked.path)

    def test_subprocess_record_is_bound_to_validated_response(self) -> None:
        request = self.publish_gdn_request("execution-request")
        response_path = self.publish_response(request, "execution-response")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="oracle stdout\n", stderr="oracle stderr\n"
        )
        with mock.patch.object(
            protocol.subprocess, "run", return_value=completed
        ) as run_mock:
            response = protocol.run_layer_oracle(
                request=request,
                response_dir=response_path,
                expected_oracle_identity=self.oracle_identity,
            )
        assert response.execution_record is not None
        self.assertEqual(response.execution_record.exit_code, 0)
        self.assertEqual(response.execution_record.stdout, "oracle stdout\n")
        self.assertEqual(response.execution_record.stderr, "oracle stderr\n")
        self.assertEqual(
            response.execution_record.argv[0], str(protocol.PINNED_ORACLE_PYTHON)
        )
        self.assertEqual(
            response.execution_record.environment,
            self.oracle_identity["launcher"]["environment"],
        )
        self.assertEqual(
            response.execution_record.launcher_identity,
            self.oracle_identity["launcher"],
        )
        run_kwargs = run_mock.call_args.kwargs
        self.assertEqual(run_kwargs["env"], self.oracle_identity["launcher"]["environment"])
        self.assertNotIn("ROCM_SIM_ROOT", run_kwargs["env"])
        self.assertNotIn("TRITON_DEFAULT_BACKEND", run_kwargs["env"])
        self.assertNotIn("TRITON_CACHE_DIR", run_kwargs["env"])
        self.assertNotIn("LD_LIBRARY_PATH", run_kwargs["env"])
        self.assertEqual(
            run_kwargs["env"]["PATH"],
            "/usr/lib/wsl/lib:/usr/bin:/bin",
        )

        error = subprocess.CalledProcessError(
            9, ["oracle"], output="partial stdout", stderr="failure stderr"
        )
        with mock.patch.object(protocol.subprocess, "run", side_effect=error):
            with self.assertRaises(protocol.LayerOracleExecutionError) as raised:
                protocol.run_layer_oracle(
                    request=request,
                    response_dir=self.private_root / "unused-response",
                    expected_oracle_identity=self.oracle_identity,
                )
        self.assertEqual(raised.exception.execution_record.exit_code, 9)
        self.assertEqual(
            raised.exception.execution_record.stderr, "failure stderr"
        )

    def test_oracle_independence_and_triton_softplus_formula_are_locked(self) -> None:
        source_path = ROOT / "tools/qwen35_nvidia_layer_oracle.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        self.assertTrue(
            roots.isdisjoint({"gem5", "gemsim_vllm", "m5", "triton"})
        )
        self.assertEqual(
            protocol.EXECUTION_BOUNDARY["allowed_framework_internal_modules"],
            ["triton"],
        )
        self.assertTrue(protocol.EXECUTION_BOUNDARY["non_target_execution"])
        self.assertIn("torch.log(1.0 + torch.exp(softplus_input))", source)
        self.assertNotIn("torch.log1p(torch.exp(softplus_input))", source)
        self.assertIn("backbone_golden.fused_gemma_rms_formula(", source)
        self.assertIn("self.final_norm", source)
        value = torch.tensor(-1.0, dtype=torch.float32)
        triton_formula = torch.log(1.0 + torch.exp(value))
        log1p_formula = torch.log1p(torch.exp(value))
        self.assertNotEqual(
            triton_formula.view(torch.int32).item(),
            log1p_formula.view(torch.int32).item(),
        )


if __name__ == "__main__":
    unittest.main()
