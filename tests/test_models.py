import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from edvi_worker.models import prepare_model, complete_model, ModelDownloadError, DEFAULT_MODEL_REPO


class LocalEntryNotFoundError(Exception):
    pass


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "snapshot"
        self.path.mkdir()
        self.download = MagicMock()
        self.modules = patch.dict("sys.modules", {
            "huggingface_hub": SimpleNamespace(snapshot_download=self.download),
            "huggingface_hub.errors": SimpleNamespace(LocalEntryNotFoundError=LocalEntryNotFoundError)})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        runtime = patch("edvi_worker.models.check_runtime")
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def populate(self):
        (self.path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (self.path / "model.safetensors").write_bytes(b"mock-weights")
        (self.path / "tokenizer_config.json").write_text("{}")
        (self.path / "tokenizer.json").write_text("{}")

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return prepare_model(Path(self.directory.name))

    def test_first_start_downloads_then_reuses_offline(self):
        def download(**kwargs):
            if kwargs.get("local_files_only") and not complete_model(self.path):
                raise LocalEntryNotFoundError
            self.populate()
            return str(self.path)

        self.download.side_effect = download
        self.assertEqual(self.prepare(), str(self.path))
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(self.download.call_args.kwargs["repo_id"], DEFAULT_MODEL_REPO)
        self.download.reset_mock()
        self.assertEqual(self.prepare(), str(self.path))
        self.download.assert_called_once()
        self.assertTrue(self.download.call_args.kwargs["local_files_only"])

    def test_partial_cache_is_repaired(self):
        def download(**kwargs):
            if not kwargs.get("local_files_only"):
                self.populate()
            return str(self.path)
        self.download.side_effect = download
        self.assertEqual(self.prepare(), str(self.path))
        self.assertEqual(self.download.call_count, 2)

    def test_missing_shard_is_incomplete(self):
        self.populate()
        (self.path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
        self.assertFalse(complete_model(self.path))

    def test_runtime_failure_prevents_download(self):
        self.runtime.side_effect = ImportError
        with self.assertRaises(ModelDownloadError):
            self.prepare()
        self.download.assert_not_called()

    def test_download_failure_is_actionable_and_sanitized(self):
        self.download.side_effect = [LocalEntryNotFoundError(), RuntimeError("secret-token")]
        with self.assertRaises(ModelDownloadError) as error:
            self.prepare()
        self.assertNotIn("secret-token", str(error.exception))
        self.assertIn("internet", str(error.exception))

    def test_incomplete_download_is_rejected(self):
        self.download.return_value = str(self.path)
        with self.assertRaises(ModelDownloadError):
            self.prepare()
