# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("download_model", ROOT / "scripts/download_model.py")
assert SPEC and SPEC.loader
download_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_model)


def locked_qwen_source() -> dict[str, object]:
    lock = json.loads((ROOT / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    return next(source for source in lock["sources"] if source["id"] == "qwen3.5-0.8b")


def normalized_model_manifest(source: dict[str, object]) -> bytes:
    files = []
    for item in source["files"]:
        entry = {
            "path": item["path"],
            "blobId": item["blob_id"],
            "size": item["size"],
        }
        if "lfs" in item:
            lfs = item["lfs"]
            entry["lfs"] = {
                "sha256": lfs["sha256"],
                "size": lfs["size"],
                "pointerSize": lfs["pointer_size"],
            }
        files.append(entry)
    files.sort(key=lambda item: item["path"])
    return json.dumps(files, sort_keys=True, separators=(",", ":")).encode()


class ModelVerificationTest(unittest.TestCase):
    def test_frozen_qwen_inventory_regression(self) -> None:
        source = locked_qwen_source()
        paths = [item["path"] for item in source["files"]]

        self.assertEqual(len(paths), 13)
        self.assertEqual(len(set(paths)), 13)
        self.assertEqual(sum(item["size"] for item in source["files"]), 1769980465)
        self.assertEqual(source["materialized_size"], 1769980465)
        self.assertEqual(
            hashlib.sha256(normalized_model_manifest(source)).hexdigest(),
            "a75f357a84fbec475d54f7b4e2eedc808af21405c23206e16cc6057453b2c9d6",
        )
        self.assertEqual(
            source["mirror_metadata"]["normalized_revision_manifest_sha256"],
            "a75f357a84fbec475d54f7b4e2eedc808af21405c23206e16cc6057453b2c9d6",
        )

    def test_frozen_lfs_pointers_reproduce_git_blob_ids(self) -> None:
        source = locked_qwen_source()
        inventory = {item["path"]: item for item in source["files"]}
        expected = {
            "model.safetensors-00001-of-00001.safetensors": {
                "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
                "size": 1746942600,
                "pointer_size": 135,
                "blob_id": "969da4e6aa85b4e224739020212b7cb0d09cee14",
            },
            "tokenizer.json": {
                "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
                "size": 12807982,
                "pointer_size": 133,
                "blob_id": "a73a846725794819aa6e1c9e97d8dc9671c2006d",
            },
        }

        with tempfile.TemporaryDirectory() as temp:
            for index, (relative, identity) in enumerate(expected.items()):
                item = inventory[relative]
                self.assertEqual(item["size"], identity["size"])
                self.assertEqual(item["blob_id"], identity["blob_id"])
                self.assertEqual(
                    {key: item["lfs"][key] for key in ("sha256", "size", "pointer_size")},
                    {
                        key: identity[key]
                        for key in ("sha256", "size", "pointer_size")
                    },
                )
                pointer = (
                    "version https://git-lfs.github.com/spec/v1\n"
                    f"oid sha256:{identity['sha256']}\n"
                    f"size {identity['size']}\n"
                ).encode()
                self.assertEqual(len(pointer), identity["pointer_size"])
                path = Path(temp) / f"pointer-{index}"
                path.write_bytes(pointer)
                self.assertEqual(download_model.git_blob_sha1(path), identity["blob_id"])

    def test_official_download_passes_exact_locked_revision(self) -> None:
        source = locked_qwen_source()
        revision = source["official_revision"]
        self.assertRegex(revision, r"^[0-9a-f]{40}$")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            (root / "cache").mkdir()
            payload = b"{}\n"
            payload_path = root / "payload"
            payload_path.write_bytes(payload)
            small_source = {
                "repo_id": source["repo_id"],
                "official_revision": revision,
                "files": [
                    {
                        "path": "config.json",
                        "size": len(payload),
                        "blob_id": download_model.git_blob_sha1(payload_path),
                    }
                ],
            }
            snapshot_download = mock.Mock(
                side_effect=lambda **kwargs: (
                    Path(kwargs["local_dir"]) / "config.json"
                ).write_bytes(payload)
            )
            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.snapshot_download = snapshot_download
            args = argparse.Namespace(
                endpoint=download_model.OFFICIAL_ENDPOINT,
                allow_mirror=False,
                output="models/Qwen3.5-0.8B",
            )
            with (
                mock.patch.object(download_model, "ROOT", root),
                mock.patch.object(download_model, "load_lock", return_value=small_source),
                mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}),
                mock.patch.dict(os.environ),
            ):
                manifest = download_model.download(args)

            snapshot_download.assert_called_once_with(
                repo_id="Qwen/Qwen3.5-0.8B",
                revision="2fc06364715b967f1860aea9cf38778875588b17",
                allow_patterns=["config.json"],
                local_dir=str(
                    root
                    / "models"
                    / ".Qwen3.5-0.8B.partial-2fc06364715b967f1860aea9cf38778875588b17"
                ),
                cache_dir=str(root / "cache" / "huggingface"),
            )
            self.assertEqual(manifest["revision"], revision)
            self.assertEqual(manifest["endpoint"], download_model.OFFICIAL_ENDPOINT)
            self.assertEqual(manifest["provenance"], "official")

    def test_mirror_requires_authorization_and_preserves_provenance(self) -> None:
        source = locked_qwen_source()
        revision = source["official_revision"]
        mirror = "https://mirror.invalid"
        small_source = {
            "repo_id": source["repo_id"],
            "official_revision": revision,
            "files": [],
        }
        args = argparse.Namespace(
            endpoint=mirror,
            allow_mirror=False,
            output="models/Qwen3.5-0.8B",
        )
        snapshot_download = mock.Mock()
        huggingface_hub = types.ModuleType("huggingface_hub")
        huggingface_hub.snapshot_download = snapshot_download
        with (
            mock.patch.object(download_model, "load_lock", return_value=small_source),
            mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}),
        ):
            with self.assertRaisesRegex(download_model.ModelError, "explicit --allow-mirror"):
                download_model.download(args)
        snapshot_download.assert_not_called()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            (root / "cache").mkdir()
            payload = b"{}\n"
            payload_path = root / "payload"
            payload_path.write_bytes(payload)
            small_source["files"] = [
                {
                    "path": "config.json",
                    "size": len(payload),
                    "blob_id": download_model.git_blob_sha1(payload_path),
                }
            ]
            snapshot_download.side_effect = lambda **kwargs: (
                Path(kwargs["local_dir"]) / "config.json"
            ).write_bytes(payload)
            args.allow_mirror = True
            with (
                mock.patch.object(download_model, "ROOT", root),
                mock.patch.object(download_model, "load_lock", return_value=small_source),
                mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}),
                mock.patch.dict(os.environ),
            ):
                manifest = download_model.download(args)

            self.assertEqual(snapshot_download.call_args.kwargs["revision"], revision)
            self.assertEqual(manifest["revision"], revision)
            self.assertEqual(manifest["endpoint"], mirror)
            self.assertEqual(manifest["provenance"], "explicit-mirror")
            marker = json.loads(
                (
                    root
                    / "models"
                    / "Qwen3.5-0.8B"
                    / ".amdgpu-sim-download.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker,
                {
                    "endpoint": mirror,
                    "repo_id": "Qwen/Qwen3.5-0.8B",
                    "revision": revision,
                },
            )

    def test_model_output_is_confined_below_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            with mock.patch.object(download_model, "ROOT", root):
                self.assertEqual(
                    download_model.model_output_path("models/Qwen/test"),
                    root / "models" / "Qwen" / "test",
                )
                with self.assertRaises(download_model.ModelError):
                    download_model.model_output_path("artifacts/model")

    def test_ignored_storage_roots_must_not_be_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "external"
            external.mkdir()
            (root / "models").symlink_to(external, target_is_directory=True)
            (root / "cache").symlink_to(external, target_is_directory=True)
            with mock.patch.object(download_model, "ROOT", root):
                with self.assertRaisesRegex(download_model.ModelError, "must not be a symlink"):
                    download_model.model_output_path("models/Qwen/test")
                with self.assertRaisesRegex(download_model.ModelError, "must not be a symlink"):
                    download_model.ignored_storage_root("cache")

    def test_download_refuses_unfrozen_source_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "SOURCE_LOCK.json").write_text(
                json.dumps({"status": "observed-not-fetched", "sources": []}),
                encoding="utf-8",
            )
            with mock.patch.object(download_model, "ROOT", root):
                with self.assertRaises(download_model.ModelError):
                    download_model.load_lock()

    def test_safetensors_header_validates_tensor_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            header = json.dumps(
                {
                    "weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
                separators=(",", ":"),
            ).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            self.assertEqual(download_model.verify_safetensors_header(path), 1)

            broken = header.replace(b"[0,4]", b"[0,5]")
            path.write_bytes(struct.pack("<Q", len(broken)) + broken + b"\0" * 4)
            with self.assertRaises(download_model.ModelError):
                download_model.verify_safetensors_header(path)

    def test_git_blob_sha1(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.txt"
            path.write_bytes(b"hello\n")
            self.assertEqual(
                download_model.git_blob_sha1(path),
                "ce013625030ba8dba906f756967f9e9ca394464a",
            )

    def test_verify_small_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "config.json"
            path.write_bytes(b"{}\n")
            source = {
                "repo_id": "Qwen/test",
                "official_revision": "a" * 40,
                "files": [
                    {
                        "path": "config.json",
                        "size": 3,
                        "blob_id": download_model.git_blob_sha1(path),
                    }
                ],
            }
            result = download_model.verify_snapshot(
                directory, source, endpoint=download_model.OFFICIAL_ENDPOINT
            )
            self.assertEqual(result["revision"], "a" * 40)
            self.assertEqual(result["files"][0]["size"], 3)

    def test_inventory_rejects_escape(self) -> None:
        with self.assertRaises(download_model.ModelError):
            download_model.expected_inventory({"files": [{"path": "../weights"}]})

    def test_verify_requires_provenance_marker_or_explicit_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "config.json"
            path.write_bytes(b"{}\n")
            source = {
                "repo_id": "Qwen/test",
                "official_revision": "a" * 40,
                "files": [
                    {
                        "path": "config.json",
                        "size": 3,
                        "blob_id": download_model.git_blob_sha1(path),
                    }
                ],
            }
            with self.assertRaises(download_model.ModelError):
                download_model.verify_snapshot(directory, source)

    def test_snapshot_marker_prevents_provenance_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "config.json"
            path.write_bytes(b"{}\n")
            source = {
                "repo_id": "Qwen/test",
                "official_revision": "a" * 40,
                "files": [
                    {
                        "path": "config.json",
                        "size": 3,
                        "blob_id": download_model.git_blob_sha1(path),
                    }
                ],
            }
            (directory / ".amdgpu-sim-download.json").write_text(
                json.dumps(
                    {
                        "repo_id": "Qwen/test",
                        "revision": "a" * 40,
                        "endpoint": "https://mirror.invalid",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(download_model.ModelError):
                download_model.verify_snapshot(
                    directory, source, endpoint=download_model.OFFICIAL_ENDPOINT
                )

    def test_snapshot_rejects_a_symlinked_provenance_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "snapshot"
            directory.mkdir()
            external = root / "external.json"
            external.write_text("{}\n", encoding="utf-8")
            (directory / ".amdgpu-sim-download.json").symlink_to(external)
            with self.assertRaisesRegex(download_model.ModelError, "must not be a symlink"):
                download_model.verify_snapshot(directory, {"files": []})

    def test_huggingface_cache_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "snapshot"
            cache = directory / ".cache"
            cache.mkdir(parents=True)
            external = root / "external-cache"
            external.mkdir()
            (cache / "huggingface").symlink_to(external, target_is_directory=True)
            config = directory / "config.json"
            config.write_bytes(b"{}\n")
            with self.assertRaisesRegex(download_model.ModelError, "contains a symlink"):
                download_model.actual_files(directory)


if __name__ == "__main__":
    unittest.main()


class MultiModelSourceSelectionTest(unittest.TestCase):
    """The ladder spans two models: 0.8B for TP1/TP2, 9B for TP16."""

    def test_lock_entry_is_selected_by_id(self) -> None:
        nine = download_model.load_lock("qwen3.5-9b")
        self.assertEqual(nine["repo_id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(nine["path"], "models/Qwen3.5-9B")
        self.assertEqual(len(nine["official_revision"]), 40)

    def test_default_id_stays_on_the_original_model(self) -> None:
        self.assertEqual(download_model.DEFAULT_SOURCE_ID, "qwen3.5-0.8b")
        self.assertEqual(download_model.load_lock()["repo_id"], "Qwen/Qwen3.5-0.8B")

    def test_unknown_id_is_rejected_by_name(self) -> None:
        with self.assertRaises(download_model.ModelError) as caught:
            download_model.load_lock("qwen3.5-nonexistent")
        self.assertIn("qwen3.5-nonexistent", str(caught.exception))

    def test_every_locked_model_pins_payload_hashes(self) -> None:
        # An entry without per-file hashes cannot detect a corrupted or
        # substituted download, which is the whole point of freezing it.
        for source_id in ("qwen3.5-0.8b", "qwen3.5-9b"):
            source = download_model.load_lock(source_id)
            weights = [
                f for f in source["files"] if f["path"].endswith(".safetensors")
            ]
            self.assertTrue(weights, f"{source_id} locks no weight files")
            for entry in weights:
                self.assertEqual(
                    len(entry["lfs"]["sha256"]), 64, f"{source_id}:{entry['path']}"
                )
