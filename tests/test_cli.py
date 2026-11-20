import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

from edvi_worker import cli
from edvi_worker.system_info import SystemInfo


class CliTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["edvi", *arguments]), redirect_stdout(output):
            cli.main()
        return output.getvalue()

    def test_help_without_command(self):
        self.assertIn("info", self.invoke())

    def test_version(self):
        with self.assertRaises(SystemExit) as exit_result:
            output = io.StringIO()
            with patch("sys.argv", ["edvi", "--version"]), redirect_stdout(output):
                cli.main()
        self.assertEqual(exit_result.exception.code, 0)
        self.assertIn("edvi-worker 0.1.0", output.getvalue())

    @patch.object(cli, "collect_system_info", return_value=SystemInfo(
        "macOS", "test", "arm64", "arm", 16.0, 8, "test-host", "Apple Silicon GPU", "MLX"))
    def test_info_formats(self, collect):
        self.assertIn("Runtime hint:  MLX", self.invoke("info"))
        result = json.loads(self.invoke("info", "--json"))
        self.assertEqual(result["hostname"], "test-host")
        self.assertEqual(result["logical_cpus"], 8)

    @patch.object(cli, "run_worker")
    def test_worker_configuration(self, run):
        with patch.dict(os.environ, {"EDVI_WORKER_TOKEN": "test-token"}):
            self.invoke("worker", "start", "--coordinator-url", "http://localhost:8000",
                        "--state-dir", "local-state", "--heartbeat-only")
        run.assert_called_once_with("http://localhost:8000", "test-token", Path("local-state"))

    @patch.object(cli, "run_worker")
    @patch.object(cli.platform, "system", return_value="Darwin")
    @patch.object(cli.platform, "machine", return_value="arm64")
    def test_apple_worker_automatically_sets_up_default_model(self, machine, system, run):
        with patch.dict(os.environ, {"EDVI_WORKER_TOKEN": "test-token"}):
            self.invoke("worker", "start", "--coordinator-url", "http://localhost:8000", "--state-dir", "state")
        run.assert_called_once_with("http://localhost:8000", "test-token", Path("state"), "local-llm", None, None)

    @patch.object(cli, "run_worker")
    def test_explicit_local_model_skips_download_configuration(self, run):
        with patch.dict(os.environ, {"EDVI_WORKER_TOKEN": "test-token"}):
            self.invoke("worker", "start", "--coordinator-url", "http://localhost:8000", "--state-dir", "state",
                        "--model-path", "model", "--mlx-model", "my-model")
        run.assert_called_once_with("http://localhost:8000", "test-token", Path("state"), "my-model", Path("model"), None)
