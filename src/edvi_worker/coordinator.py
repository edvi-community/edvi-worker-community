"""Local, single-account coordinator with persistent SQLite storage."""
from __future__ import annotations

import hmac
import ipaddress
import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .jobs import SCHEMA, handle_jobs
from .web import allow_browser_request, serve_asset


class CoordinatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, database: Path, token: str, interval=30, offline_after=90):
        if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
            raise ValueError("Set EDVI_COORDINATOR_TOKEN to a nonempty ASCII token without whitespace.")
        if interval <= 0 or offline_after < interval * 3:
            raise ValueError("Offline threshold must be at least three positive heartbeat intervals.")
        address_ip = ipaddress.ip_address(address[0])
        if address_ip.version != 4 or not address_ip.is_loopback:
            raise ValueError("The local coordinator must bind to an IPv4 loopback address.")
        self.database = str(database)
        self.token = token
        self.interval = interval
        self.offline_after = offline_after
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY, worker_version TEXT NOT NULL,
                    capabilities TEXT NOT NULL, last_seen REAL NOT NULL,
                    status TEXT NOT NULL)""")
                connection.execute(SCHEMA)
        finally:
            connection.close()
        super().__init__(address, CoordinatorHandler)

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def valid_registration(body):
    if not isinstance(body, dict):
        return False
    version = body.get("worker_version")
    capabilities = body.get("capabilities")
    if not isinstance(version, str) or not version.strip() or not isinstance(capabilities, dict):
        return False
    for field in ("os", "os_version", "architecture", "processor", "hostname", "accelerator", "runtime_hint"):
        if not isinstance(capabilities.get(field), str) or not capabilities[field].strip():
            return False
    memory = capabilities.get("memory_gb")
    cpus = capabilities.get("logical_cpus")
    return (type(memory) in (int, float) and math.isfinite(memory) and memory >= 0
            and type(cpus) is int and cpus > 0)


class CoordinatorHandler(BaseHTTPRequestHandler):
    def read_body(self):
        try:
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                raise ValueError
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError
            body = json.loads(self.rfile.read(length))
            if body is None:
                raise ValueError
            return body
        except (ValueError, UnicodeError):
            self.error(400, "invalid_request", "Expected a JSON body of at most 65536 bytes")
            return None

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        # Request paths and headers are deliberately not logged.
        pass

    def reply(self, status, body):
        raw = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def error(self, status, code, message):
        self.reply(status, {"error": {"code": code, "message": message}})

    def dispatch(self):
        access = getattr(self.server, "authorize_request", None)
        if access is not None:
            if not access(self):
                return
        elif not allow_browser_request(self):
            self.error(403, "forbidden", "Use this service from its local address")
            return
        if serve_asset(self, "dashboard"):
            return
        if self.path == "/v1/config" and self.command == "GET":
            self.reply(200, {"hosted": access is not None})
            return
        if access is None and not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                   f"Bearer {self.server.token}".encode()):
            self.error(401, "unauthorized", "Invalid worker token")
            return
        parts = self.path.split("/")
        if handle_jobs(self):
            return
        listing = self.path == "/v1/workers" and self.command == "GET"
        registration = len(parts) == 4 and parts[:3] == ["", "v1", "workers"] and self.command == "PUT"
        heartbeat = (len(parts) == 5 and parts[:3] == ["", "v1", "workers"]
                     and parts[4] == "heartbeat" and self.command == "POST")
        if not (listing or registration or heartbeat):
            self.error(404, "not_found", "Unknown endpoint")
            return
        if not listing:
            try:
                worker_id = str(uuid.UUID(parts[3]))
                if worker_id != parts[3]:
                    raise ValueError
            except ValueError:
                self.error(400, "invalid_request", "worker_id must be a canonical UUID")
                return
            body = self.read_body()
            if body is None:
                return
            if registration and not valid_registration(body):
                self.error(400, "invalid_request", "Invalid worker version or capabilities")
                return
            if registration and "mlx_models" in body["capabilities"]:
                models = body["capabilities"]["mlx_models"]
                if not isinstance(models, list) or len(models) > 16 or any(not isinstance(m, str) or not 0 < len(m) <= 200 for m in models):
                    self.error(400, "invalid_request", "Invalid MLX model list")
                    return
            if heartbeat and (not isinstance(body, dict) or body.get("status") not in ("idle", "stopping")):
                self.error(400, "invalid_request", "status must be idle or stopping")
                return
        now = time.time()
        # Explicit close is needed: SQLite's context manager only commits/rolls back.
        connection = self.server.connect()
        try:
            with connection:
                if listing:
                    workers = []
                    for row in connection.execute("SELECT * FROM workers ORDER BY worker_id"):
                        online = row["status"] == "idle" and now - row["last_seen"] <= self.server.offline_after
                        workers.append({"worker_id": row["worker_id"], "worker_version": row["worker_version"],
                            "capabilities": json.loads(row["capabilities"]), "last_seen": timestamp(row["last_seen"]),
                            "status": "idle" if online else "offline"})
                    result, status = {"workers": workers}, 200
                elif registration:
                    connection.execute("BEGIN IMMEDIATE")
                    exists = connection.execute("SELECT 1 FROM workers WHERE worker_id = ?", (worker_id,)).fetchone()
                    connection.execute("""INSERT INTO workers VALUES (?, ?, ?, ?, 'idle')
                        ON CONFLICT(worker_id) DO UPDATE SET worker_version=excluded.worker_version,
                        capabilities=excluded.capabilities, last_seen=excluded.last_seen, status='idle'""",
                        (worker_id, body["worker_version"], json.dumps(body["capabilities"]), now))
                    result = {"worker_id": worker_id, "heartbeat_interval_seconds": self.server.interval,
                              "offline_after_seconds": self.server.offline_after}
                    status = 200 if exists else 201
                else:
                    updated = connection.execute("UPDATE workers SET last_seen=?, status=? WHERE worker_id=?",
                                                 (now, body["status"], worker_id))
                    if not updated.rowcount:
                        self.error(404, "not_registered", "Register the worker first")
                        return
                    result, status = {"worker_id": worker_id, "received_at": timestamp(now)}, 200
            self.reply(status, result)
        finally:
            connection.close()

    def handle_request(self):
        try:
            self.dispatch()
        except sqlite3.Error:
            self.error(503, "storage_unavailable", "Coordinator storage unavailable")
        except (TimeoutError, ConnectionError):
            self.close_connection = True

    do_GET = handle_request
    do_PUT = handle_request
    do_POST = handle_request


def run_coordinator(host, port, database, token, interval, offline_after):
    with CoordinatorServer((host, port), database, token, interval, offline_after) as server:
        print(f"Coordinator listening on http://{host}:{server.server_port}", flush=True)
        print(f"Browser dashboard: http://{host}:{server.server_port}/", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("Coordinator stopped.", flush=True)
