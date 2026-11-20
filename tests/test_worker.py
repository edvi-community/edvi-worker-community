import io
import json
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from edvi_worker import worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_request(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.command, self.path, payload,
                                       self.headers.get("Authorization")))
                worker_id = self.path.split("/")[3]
                if owner.responses:
                    status, body, headers = owner.responses.pop(0)
                elif self.command == "PUT":
                    status, body, headers = 201, {
                        "worker_id": worker_id, "heartbeat_interval_seconds": 2,
                        "offline_after_seconds": 6}, {}
                else:
                    status, body, headers = 200, {
                        "worker_id": worker_id, "received_at": "2026-10-01T12:00:00Z"}, {}
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            do_PUT = handle_request
            do_POST = handle_request

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def run_loop(self, sleeps):
        with patch.object(worker.time, "sleep", side_effect=sleeps) as sleep, \
             redirect_stdout(io.StringIO()):
            worker.run_worker(self.url, "test-token", self.path)
        return sleep

    def test_registration_heartbeat_shutdown_and_persistent_id(self):
        sleep = self.run_loop([KeyboardInterrupt()])
        sleep.assert_called_once_with(2)
        self.assertEqual([r[0] for r in self.requests], ["PUT", "POST", "POST"])
        self.assertEqual(self.requests[-1][2], {"status": "stopping"})
        self.assertEqual(self.requests[1][2], {"status": "idle"})
        self.assertEqual(self.requests[0][3], "Bearer test-token")
        self.assertIn("logical_cpus", self.requests[0][2]["capabilities"])
        worker_id = worker.load_worker_id(self.path)
        self.assertIn(worker_id, self.requests[0][1])
        self.assertEqual(worker.load_worker_id(self.path), worker_id)

    def test_missing_registration_registers_again(self):
        worker_id = worker.load_worker_id(self.path)
        self.responses = [(200, {"worker_id": worker_id,
            "heartbeat_interval_seconds": 2, "offline_after_seconds": 6}, {}),
            (404, {}, {})]
        self.run_loop([KeyboardInterrupt()])
        self.assertEqual([r[0] for r in self.requests], ["PUT", "POST", "PUT", "POST", "POST"])

    def test_server_failures_backoff_and_shutdown_during_retry(self):
        self.responses = [(503, {}, {}), (503, {}, {})]
        with patch.object(worker.random, "uniform", return_value=1):
            sleep = self.run_loop([None, KeyboardInterrupt()])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(self.requests[-1][2], {"status": "stopping"})

    def test_rate_limit_and_backoff_reset_after_success(self):
        worker_id = worker.load_worker_id(self.path)
        self.responses = [(429, {}, {"Retry-After": "7"}),
            (200, {"worker_id": worker_id, "heartbeat_interval_seconds": 2,
                   "offline_after_seconds": 6}, {}), (503, {}, {})]
        with patch.object(worker.random, "uniform", return_value=1):
            sleep = self.run_loop([None, KeyboardInterrupt()])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [7, 1])

    def test_authentication_failure_does_not_expose_token(self):
        self.responses = [(401, {"secret": "test-token"}, {})]
        with self.assertRaises(worker.WorkerError) as failure, redirect_stdout(io.StringIO()):
            worker.run_worker(self.url, "test-token", self.path)
        self.assertNotIn("test-token", str(failure.exception))
        self.assertEqual(len(self.requests), 1)

    def test_protocol_errors_and_redirects(self):
        client = worker.CoordinatorClient(self.url, "test-token", str(uuid.uuid4()))
        for status, body, headers in (
            (200, [], {}),
            (200, {"worker_id": client.worker_id, "heartbeat_interval_seconds": True,
                   "offline_after_seconds": 90}, {}),
            (302, {}, {"Location": self.url + "/other"}),
        ):
            with self.subTest(status=status, body=body):
                self.responses = [(status, body, headers)]
                with self.assertRaises(worker.WorkerError):
                    client.register()
        self.assertEqual(len(self.requests), 3)

    def test_connection_failures_are_retryable(self):
        client = worker.CoordinatorClient(self.url, "test-token", str(uuid.uuid4()))
        with patch.object(client.opener, "open", side_effect=TimeoutError):
            with self.assertRaises(worker.RetryRequest):
                client.register()

    def test_malformed_json_and_invalid_heartbeat(self):
        client = worker.CoordinatorClient(self.url, "test-token", str(uuid.uuid4()))
        with patch.object(client.opener, "open") as open_request:
            response = open_request.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b"not-json"
            with self.assertRaises(worker.WorkerError):
                client.register()
        self.responses = [(200, {"worker_id": client.worker_id, "received_at": "bad"}, {})]
        with self.assertRaises(worker.WorkerError):
            client.heartbeat()

    def test_url_and_state_validation(self):
        for url in ("http://example.com", "https://user:secret@example.com", "https://example.com?token=secret"):
            with self.assertRaises(worker.WorkerError):
                worker.validate_url(url)
        self.assertEqual(worker.validate_url("https://example.com/"), "https://example.com")
        (self.path / "worker_id").write_text("bad-id")
        with self.assertRaises(worker.WorkerError):
            worker.load_worker_id(self.path)
