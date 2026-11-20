import io
import json
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from edvi_worker.coordinator import CoordinatorServer
from edvi_worker.worker import CoordinatorClient, run_worker


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "coordinator.sqlite3"
        self.start_server()
        self.worker_id = str(uuid.uuid4())
        self.client = CoordinatorClient(self.url, "test-token", self.worker_id)

    def start_server(self):
        self.server = CoordinatorServer(("127.0.0.1", 0), self.database, "test-token", 1, 3)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def tearDown(self):
        self.stop_server()
        self.directory.cleanup()

    def request(self, path="/v1/workers", method="GET", body=None, token="test-token"):
        request = Request(self.url + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_registration_idempotence_and_restart_persistence(self):
        self.assertEqual(self.client.register(), 1)
        self.client.register()
        self.client.heartbeat()
        status, result = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(len(result["workers"]), 1)
        self.assertEqual(result["workers"][0]["status"], "idle")
        self.stop_server()
        self.start_server()
        self.assertEqual(self.request()[1]["workers"][0]["worker_id"], self.worker_id)

    def test_dashboard_assets_are_public_but_api_requires_token(self):
        with urlopen(self.url + "/", timeout=3) as response:
            html = response.read()
            self.assertIn(b"New LLM job", html)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            self.assertNotIn(b"test-token", html)
        with urlopen(self.url + "/dashboard.js", timeout=3) as response:
            self.assertIn(b"/v1/jobs", response.read())
        self.assertEqual(self.request(token="wrong")[0], 401)

    def test_offline_expiry_and_stopping(self):
        self.client.register()
        with patch("edvi_worker.coordinator.time.time", return_value=9999999999):
            self.assertEqual(self.request()[1]["workers"][0]["status"], "offline")
        self.client.heartbeat()
        self.assertEqual(self.request()[1]["workers"][0]["status"], "idle")
        self.client.heartbeat("stopping")
        self.assertEqual(self.request()[1]["workers"][0]["status"], "offline")

    def test_authentication_validation_and_missing_worker(self):
        self.assertEqual(self.request(token="wrong")[0], 401)
        route = f"/v1/workers/{self.worker_id}"
        self.assertEqual(self.request(route, "PUT", {})[0], 400)
        self.assertEqual(self.request(route + "/heartbeat", "POST", {"status": "idle"})[0], 404)
        self.assertEqual(self.request(route + "/heartbeat", "POST", {"status": "busy"})[0], 400)
        self.assertEqual(self.request("/v1/workers/bad-id", "PUT", {})[0], 400)

    def test_worker_loop_against_real_coordinator(self):
        with patch("edvi_worker.worker.time.sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            run_worker(self.url, "test-token", Path(self.directory.name) / "worker-state")
        workers = self.request()[1]["workers"]
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]["status"], "offline")
        self.assertIn("logical_cpus", workers[0]["capabilities"])

    def submit_job(self, model="test-model"):
        status, response = self.request("/v1/jobs", "POST", {"model": model,
            "prompt": "Hello", "max_tokens": 16, "temperature": 0, "timeout_seconds": 10})
        self.assertEqual(status, 201)
        return response["job_id"]

    def claim_job(self):
        return self.request("/v1/jobs/claim", "POST", {"worker_id": self.worker_id})[1]["job"]

    def test_jobs_model_matching_completion_and_idempotence(self):
        self.client.models = ["test-model"]
        self.client.register()
        other = self.submit_job("other-model")
        job_id = self.submit_job()
        job = self.claim_job()
        self.assertEqual(job["job_id"], job_id)
        self.assertEqual(self.claim_job()["lease_id"], job["lease_id"])
        result = {"worker_id": self.worker_id, "lease_id": job["lease_id"], "result": {"text": "Hello back"}}
        for _ in range(2):
            self.assertEqual(self.request(f"/v1/jobs/{job_id}/result", "POST", result)[0], 200)
        response = self.request(f"/v1/jobs/{job_id}")[1]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"]["text"], "Hello back")
        self.assertIsNone(self.claim_job())
        self.assertEqual(self.request(f"/v1/jobs/{other}")[1]["status"], "queued")

    def test_expired_lease_requeues_and_rejects_old_result(self):
        self.client.models = ["test-model"]
        self.client.register()
        job_id = self.submit_job()
        job = self.claim_job()
        with patch("edvi_worker.jobs.time.time", return_value=job["lease_until"] + 1):
            self.assertEqual(self.request(f"/v1/jobs/{job_id}")[1]["status"], "queued")
        replacement = self.claim_job()
        self.assertNotEqual(job["lease_id"], replacement["lease_id"])
        result = {"worker_id": self.worker_id, "lease_id": job["lease_id"], "result": {"text": "stale"}}
        self.assertEqual(self.request(f"/v1/jobs/{job_id}/result", "POST", result)[0], 409)

    def test_invalid_jobs_and_failed_result(self):
        self.assertEqual(self.request("/v1/jobs", "POST", {})[0], 400)
        self.client.models = ["test-model"]
        self.client.register()
        job_id = self.submit_job()
        job = self.claim_job()
        result = {"worker_id": self.worker_id, "lease_id": job["lease_id"], "result": {"error": "Timed out"}}
        self.assertEqual(self.request(f"/v1/jobs/{job_id}/result", "POST", result)[0], 200)
        self.assertEqual(self.request(f"/v1/jobs/{job_id}")[1]["status"], "failed")

    def test_mlx_worker_end_to_end_with_mock_inference(self):
        job_id = self.submit_job()
        with patch("edvi_worker.inference.check_model", return_value="/local/model"), \
             patch("edvi_worker.worker.execute_job", return_value={"text": "Generated answer"}) as execute, \
             patch("edvi_worker.worker.time.sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            run_worker(self.url, "test-token", Path(self.directory.name) / "state", "test-model", "/local/model")
        execute.assert_called_once()
        response = self.request(f"/v1/jobs/{job_id}")[1]
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"]["text"], "Generated answer")
