import time
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from edvi_worker.worker import execute_job, InferenceProcess, RetryRequest
from edvi_worker.inference import infer, ModelSession


class InferenceTests(unittest.TestCase):
    def test_timeout_kills_child_and_sends_heartbeat(self):
        process = MagicMock()
        process.poll.return_value = None
        client = MagicMock()
        job = {"timeout_seconds": 1, "lease_until": time.time() + 40}
        with patch("edvi_worker.worker.subprocess.Popen", return_value=process), \
             patch("edvi_worker.worker.time.sleep"), \
             patch("edvi_worker.worker.time.monotonic", side_effect=[0, 1, 1, 1, 3]):
            job["timeout_seconds"] = 2
            self.assertIn("timed out", execute_job(client, job, "/model", 1)["error"])
        process.kill.assert_called_once()
        client.heartbeat.assert_called_once()

    def test_successful_jobs_reuse_process(self):
        process = MagicMock()
        process.poll.return_value = None
        runtime = InferenceProcess("/model")
        client = MagicMock()
        job = {"timeout_seconds": 10, "lease_until": time.time() + 40}

        def finish(_):
            directory = Path(runtime.directory.name)
            (directory / "request.json").unlink()
            (directory / "result.json").write_text(json.dumps({"text": "answer"}))

        try:
            with patch("edvi_worker.worker.subprocess.Popen", return_value=process) as spawn, \
                 patch("edvi_worker.worker.time.sleep", side_effect=finish):
                for _ in range(2):
                    self.assertEqual(runtime.execute(client, job, 1), {"text": "answer"})
                spawn.assert_called_once()
                process.kill.assert_not_called()
        finally:
            runtime.close()
        process.kill.assert_called_once()

    def test_network_interruption_discards_child(self):
        process = MagicMock()
        process.poll.return_value = None
        client = MagicMock()
        client.heartbeat.side_effect = RetryRequest
        with patch("edvi_worker.worker.subprocess.Popen", return_value=process), \
             patch("edvi_worker.worker.time.monotonic", side_effect=[0, 1]):
            with self.assertRaises(RetryRequest):
                execute_job(client, {"timeout_seconds": 10, "lease_until": time.time() + 40}, "/model", 1)
        process.kill.assert_called_once()

    def test_session_loads_model_once(self):
        mlx_lm = MagicMock()
        mlx_lm.load.return_value = (object(), MagicMock())
        session = ModelSession("/model")
        with patch.dict("sys.modules", {"mlx_lm": mlx_lm}), \
             patch("edvi_worker.inference.generate_text", return_value="answer"):
            self.assertEqual(session.infer({"prompt": "first"}), "answer")
            self.assertEqual(session.infer({"prompt": "second"}), "answer")
        mlx_lm.load.assert_called_once_with("/model")

    def test_crashed_process_is_restarted_for_next_job(self):
        dead = MagicMock()
        dead.poll.return_value = 1
        healthy = MagicMock()
        healthy.poll.return_value = None
        runtime = InferenceProcess("/model")
        job = {"timeout_seconds": 10, "lease_until": time.time() + 40}

        def finish(_):
            directory = Path(runtime.directory.name)
            (directory / "request.json").unlink()
            (directory / "result.json").write_text(json.dumps({"text": "recovered"}))

        try:
            with patch("edvi_worker.worker.subprocess.Popen", side_effect=[dead, healthy]) as spawn, \
                 patch("edvi_worker.worker.time.sleep", side_effect=finish):
                self.assertIn("error", runtime.execute(MagicMock(), job, 1))
                self.assertEqual(runtime.execute(MagicMock(), job, 1), {"text": "recovered"})
                self.assertEqual(spawn.call_count, 2)
                dead.wait.assert_called_once()
        finally:
            runtime.close()

    def test_mlx_api_parameters_and_chat_template(self):
        model = object()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        mlx_lm = MagicMock()
        mlx_lm.load.return_value = (model, tokenizer)
        mlx_lm.generate.return_value = "answer"
        sampling = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mlx_lm, "mlx_lm.sample_utils": sampling}):
            self.assertEqual(infer("/model", {"prompt": "hello", "max_tokens": 8, "temperature": 0}), "answer")
        sampling.make_sampler.assert_called_once_with(temp=0)
        self.assertEqual(mlx_lm.generate.call_args.kwargs["max_tokens"], 8)
