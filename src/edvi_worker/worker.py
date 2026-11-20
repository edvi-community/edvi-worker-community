"""Coordinator registration and heartbeat client (standard library only)."""
from __future__ import annotations

import ipaddress
import json
import random
import time
import uuid
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import __version__
from .system_info import collect_system_info


class WorkerError(Exception):
    """An actionable configuration or protocol failure."""


class RetryRequest(Exception):
    def __init__(self, delay=None):
        self.delay = delay


class RegistrationMissing(Exception):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        parts.port  # Validate malformed ports before making a request.
        local = parts.hostname == "localhost"
        try:
            local = local or ipaddress.ip_address(parts.hostname or "").is_loopback
        except ValueError:
            pass
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment
                or (parts.scheme == "http" and not local)):
            raise ValueError
    except ValueError:
        raise WorkerError("Coordinator URL must use HTTPS (HTTP is allowed on loopback only), without credentials, query, or fragment.") from None
    return url.rstrip("/")


def load_worker_id(state_dir: Path) -> str:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "worker_id"
        # Never overwrite an existing worker identity.
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(str(uuid.uuid4()))
        except FileExistsError:
            pass
        return str(uuid.UUID(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        raise WorkerError("Cannot read or create worker_id in the state directory; check permissions and UUID contents.") from None


class CoordinatorClient:
    def __init__(self, url: str, token: str, worker_id: str):
        self.url = validate_url(url)
        if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
            raise WorkerError("EDVI_WORKER_TOKEN must contain a nonempty ASCII token without whitespace.")
        self.token = token
        self.worker_id = worker_id
        self.opener = build_opener(NoRedirects())
        self.models = []

    def request(self, method, suffix, payload, timeout=10, route=None):
        request = Request(self.url + (route or f"/v1/workers/{self.worker_id}{suffix}"),
                          data=json.dumps(payload).encode("utf-8"), method=method,
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {self.token}"})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                expected = {200, 201} if method == "PUT" else {200}
                if response.status not in expected:
                    raise WorkerError("Unexpected coordinator success status.")
                raw = response.read(65537)
                if len(raw) > 65536:
                    raise WorkerError("Coordinator response is too large.")
                result = json.loads(raw)
        except HTTPError as error:
            error.close()
            if error.code == 404 and suffix:
                raise RegistrationMissing from None
            if error.code == 429:
                value = error.headers.get("Retry-After", "")
                raise RetryRequest(int(value) if value.isascii() and value.isdigit() else None) from None
            if 500 <= error.code <= 599:
                raise RetryRequest from None
            if error.code in {401, 403}:
                raise WorkerError("Coordinator authentication failed; check EDVI_WORKER_TOKEN.") from None
            raise WorkerError(f"Coordinator rejected request (HTTP {error.code}).") from None
        except (URLError, TimeoutError, OSError, HTTPException):
            raise RetryRequest from None
        except (ValueError, UnicodeError):
            raise WorkerError("Coordinator returned malformed JSON.") from None
        if not isinstance(result, dict) or result.get("worker_id") != self.worker_id:
            raise WorkerError("Coordinator returned an invalid worker ID.")
        return result

    def register(self):
        capabilities = asdict(collect_system_info())
        if self.models:
            capabilities["mlx_models"] = self.models
        result = self.request("PUT", "", {"worker_version": __version__,
                             "capabilities": capabilities})
        interval = result.get("heartbeat_interval_seconds")
        offline = result.get("offline_after_seconds")
        if (type(interval) is not int or interval <= 0 or type(offline) is not int
                or offline < 3 * interval):
            raise WorkerError("Coordinator returned invalid heartbeat timing.")
        return interval

    def heartbeat(self, status="idle", timeout=10):
        result = self.request("POST", "/heartbeat", {"status": status}, timeout)
        timestamp = result.get("received_at")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            raise WorkerError("Coordinator returned an invalid receipt timestamp.") from None


class InferenceProcess:
    """Own a reusable model process; discard it after interrupted execution."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.process = None
        self.directory = None

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.directory = tempfile.TemporaryDirectory()
        self.process = subprocess.Popen([
            sys.executable, "-m", "edvi_worker.inference", "--serve",
            self.model_path, self.directory.name], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def execute(self, client, job, interval):
        started = time.monotonic()
        budget = min(job["timeout_seconds"], max(0, job["lease_until"] - time.time() - 10))
        next_heartbeat = started + interval
        complete = False
        try:
            self.start()
            directory = Path(self.directory.name)
            result_path = directory / "result.json"
            result_path.unlink(missing_ok=True)
            temporary = directory / "request.tmp"
            temporary.write_text(json.dumps(job), encoding="utf-8")
            temporary.replace(directory / "request.json")
            while True:
                now = time.monotonic()
                if now - started >= budget:
                    return {"error": "MLX inference timed out."}
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if (not isinstance(result, dict) or set(result) not in ({"text"}, {"error"})
                            or not isinstance(next(iter(result.values())), str)):
                        return {"error": "MLX inference process returned an invalid result."}
                    complete = "text" in result
                    return result
                if self.process.poll() is not None:
                    return {"error": "MLX inference process failed."}
                if now >= next_heartbeat:
                    client.heartbeat()
                    next_heartbeat = time.monotonic() + interval
                time.sleep(min(0.05, max(0, budget - (time.monotonic() - started))))
        except (OSError, ValueError):
            return {"error": "MLX inference process failed."}
        finally:
            if not complete:
                self.close()


def execute_job(client, job, model_path, interval, runtime=None):
    if runtime is not None:
        return runtime.execute(client, job, interval)
    runtime = InferenceProcess(model_path)
    try:
        return runtime.execute(client, job, interval)
    finally:
        runtime.close()


def run_worker(url: str, token: str, state_dir: Path, mlx_model=None, model_path=None, model_repo=None) -> None:
    validate_url(url)
    client = CoordinatorClient(url, token, load_worker_id(state_dir))
    if mlx_model:
        if model_path is None:
            from .models import prepare_model, ModelDownloadError, DEFAULT_MODEL_REPO
            try:
                model_path = prepare_model(state_dir, model_repo or DEFAULT_MODEL_REPO)
            except ModelDownloadError as error:
                raise WorkerError(str(error)) from None
        from .inference import check_model
        try:
            model_path = check_model(model_path)
        except (ImportError, ValueError, OSError):
            raise WorkerError("MLX requires Apple Silicon, installed mlx-lm, and a local model directory with weights.") from None
        client.models = [mlx_model]
    interval = None
    backoff = 1
    pending = None
    runtime = InferenceProcess(model_path) if mlx_model else None
    print(f"Starting worker {client.worker_id}", flush=True)
    try:
        while True:
            try:
                if interval is None:
                    interval = client.register()
                    backoff = 1
                    print("Registered with coordinator.", flush=True)
                client.heartbeat()
                if backoff > 1:
                    print("Connected to coordinator.", flush=True)
                if mlx_model:
                    if pending is None:
                        claimed = client.request("POST", "", {"worker_id": client.worker_id}, route="/v1/jobs/claim")
                        job = claimed.get("job")
                        if job:
                            from .jobs import valid_job
                            if (not valid_job(job) or job.get("model") != mlx_model
                                    or not isinstance(job.get("lease_id"), str)
                                    or not isinstance(job.get("job_id"), str)
                                    or type(job.get("lease_until")) not in (int, float)):
                                raise WorkerError("Coordinator returned an invalid MLX job.")
                            try:
                                uuid.UUID(job["job_id"])
                            except ValueError:
                                raise WorkerError("Coordinator returned an invalid job ID.") from None
                            result = execute_job(client, job, model_path, interval, runtime)
                            pending = (job, result)
                    if pending:
                        job, result = pending
                        try:
                            client.request("POST", "", {"worker_id": client.worker_id,
                                "lease_id": job["lease_id"], "result": result},
                                route=f"/v1/jobs/{job['job_id']}/result")
                        except WorkerError:
                            if time.time() <= job["lease_until"]:
                                raise
                        pending = None
                backoff = 1
                time.sleep(interval)
            except RegistrationMissing:
                interval = None
            except RetryRequest as error:
                delay = error.delay if error.delay is not None else backoff * random.uniform(1, 1.2)
                print("Coordinator unavailable; retrying.", flush=True)
                time.sleep(delay)
                backoff = min(backoff * 2, 60)
    except KeyboardInterrupt:
        try:
            client.heartbeat("stopping", timeout=3)
        except (WorkerError, RetryRequest, RegistrationMissing, KeyboardInterrupt):
            pass
        print("Worker stopped.", flush=True)

    finally:
        if runtime is not None:
            runtime.close()
