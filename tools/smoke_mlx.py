"""Run a real local coordinator, worker, and model; keep no job database."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from edvi_worker.coordinator import CoordinatorServer


def main():
    parser = argparse.ArgumentParser()
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path", type=Path)
    model.add_argument("--auto-download", action="store_true")
    parser.add_argument("--state-dir", type=Path, help="Keep automatic downloads here between smoke tests")
    args = parser.parse_args()
    token = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as directory:
        with CoordinatorServer(("127.0.0.1", 0), Path(directory) / "jobs.sqlite3", token, 1, 3) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"

            def request(path, body=None):
                req = Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                with urlopen(req, timeout=5) as response:
                    return json.load(response)

            process = None
            try:
                command = [sys.executable, "-m", "edvi_worker.cli", "worker", "start",
                    "--coordinator-url", url, "--state-dir", str(args.state_dir or Path(directory) / "worker"),
                    "--mlx-model", "smoke-model"]
                if args.model_path:
                    command.extend(["--model-path", str(args.model_path.resolve())])
                process = subprocess.Popen(command,
                    env=dict(os.environ, EDVI_WORKER_TOKEN=token))
                for prompt in ("Say hello in one sentence.", "What is two plus two? Answer briefly."):
                    started = time.monotonic()
                    job = request("/v1/jobs", {"model": "smoke-model", "prompt": prompt,
                        "max_tokens": 32, "temperature": 0, "timeout_seconds": 60})
                    deadline = started + 90
                    while time.monotonic() < deadline:
                        result = request(f"/v1/jobs/{job['job_id']}")
                        if result["status"] in {"completed", "failed"}:
                            print(json.dumps(result, indent=2), flush=True)
                            print(f"Submission to result: {time.monotonic() - started:.2f}s", flush=True)
                            if result["status"] != "completed":
                                raise RuntimeError("Inference job failed")
                            break
                        if process.poll() is not None:
                            raise RuntimeError("Worker exited before completing the job")
                        time.sleep(0.5)
                    else:
                        raise RuntimeError("Smoke test timed out")
            finally:
                if process and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    main()
