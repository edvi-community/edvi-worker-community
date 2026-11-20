"""SQLite-backed LLM jobs with atomic claims and expiring execution leases."""
import json
import time
import uuid


SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
    worker_id TEXT, lease_id TEXT, lease_until REAL, result TEXT, created_at REAL NOT NULL)"""


def valid_job(body):
    return (isinstance(body, dict) and isinstance(body.get("model"), str)
            and 0 < len(body["model"]) <= 200
            and isinstance(body.get("prompt"), str) and 0 < len(body["prompt"].encode()) <= 16000
            and type(body.get("max_tokens")) is int and 1 <= body["max_tokens"] <= 1024
            and type(body.get("temperature")) in (int, float) and 0 <= body["temperature"] <= 2
            and type(body.get("timeout_seconds")) is int and 1 <= body["timeout_seconds"] <= 600)


def handle_jobs(handler):
    parts = handler.path.split("/")
    if parts[:3] != ["", "v1", "jobs"]:
        return False
    submit = len(parts) == 3 and handler.command == "POST"
    listing = len(parts) == 3 and handler.command == "GET"
    claim = parts == ["", "v1", "jobs", "claim"] and handler.command == "POST"
    finish = len(parts) == 5 and parts[4] == "result" and handler.command == "POST"
    detail = len(parts) == 4 and handler.command == "GET"
    if not (submit or listing or claim or finish or detail):
        handler.error(404, "not_found", "Unknown job endpoint")
        return True
    body = None
    if handler.command == "POST":
        body = handler.read_body()
        if body is None:
            return True
        if not isinstance(body, dict):
            handler.error(400, "invalid_request", "Expected JSON object")
            return True
    now = time.time()
    connection = handler.server.connect()
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""UPDATE jobs SET status='queued', worker_id=NULL, lease_id=NULL,
                lease_until=NULL WHERE status='running' AND lease_until < ?""", (now,))
            if submit:
                if not valid_job(body):
                    handler.error(400, "invalid_request", "Invalid model, prompt, generation limits, or timeout")
                    return True
                job_id = str(uuid.uuid4())
                connection.execute("INSERT INTO jobs (job_id,payload,status,created_at) VALUES (?,?,'queued',?)",
                                   (job_id, json.dumps(body), now))
                status, response = 201, {"job_id": job_id, "status": "queued"}
            elif claim:
                worker_id = body.get("worker_id")
                if not isinstance(worker_id, str):
                    handler.error(400, "invalid_request", "worker_id is required")
                    return True
                worker = connection.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
                if worker is None:
                    handler.error(404, "not_registered", "Register the worker first")
                    return True
                models = json.loads(worker["capabilities"]).get("mlx_models", [])
                row = connection.execute("SELECT * FROM jobs WHERE status='running' AND worker_id=?", (worker_id,)).fetchone()
                if row is None and worker["status"] == "idle" and now - worker["last_seen"] <= handler.server.offline_after:
                    row = next((r for r in connection.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at")
                                if json.loads(r["payload"])["model"] in models), None)
                    if row:
                        lease = str(uuid.uuid4())
                        until = now + json.loads(row["payload"])["timeout_seconds"] + 30
                        connection.execute("UPDATE jobs SET status='running',worker_id=?,lease_id=?,lease_until=? WHERE job_id=?",
                                           (worker_id, lease, until, row["job_id"]))
                        row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
                response = {"worker_id": worker_id, "job": None}
                if row:
                    response["job"] = dict(json.loads(row["payload"]), job_id=row["job_id"],
                                           lease_id=row["lease_id"], lease_until=row["lease_until"])
                status = 200
            elif finish:
                row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (parts[3],)).fetchone()
                if row is None:
                    handler.error(404, "not_found", "Unknown job")
                    return True
                if row["lease_id"] != body.get("lease_id") or row["worker_id"] != body.get("worker_id"):
                    handler.error(409, "lease_lost", "Job lease no longer belongs to this worker")
                    return True
                result = body.get("result")
                if (not isinstance(result, dict) or set(result) not in ({"text"}, {"error"})
                        or not isinstance(next(iter(result.values())), str)
                        or len(next(iter(result.values())).encode()) > 24000):
                    handler.error(400, "invalid_request", "Expected bounded text or error result")
                    return True
                if row["status"] == "running":
                    connection.execute("UPDATE jobs SET status=?,result=? WHERE job_id=?",
                                       ("failed" if "error" in result else "completed", json.dumps(result), parts[3]))
                elif row["result"] != json.dumps(result):
                    handler.error(409, "result_conflict", "Result has already been recorded")
                    return True
                status, response = 200, {"worker_id": body["worker_id"], "job_id": parts[3]}
            else:
                rows = connection.execute("SELECT * FROM jobs ORDER BY created_at").fetchall() if listing else connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (parts[3],)).fetchall()
                if detail and not rows:
                    handler.error(404, "not_found", "Unknown job")
                    return True
                records = [dict(job_id=r["job_id"], status=r["status"], request=json.loads(r["payload"]),
                                worker_id=r["worker_id"], result=json.loads(r["result"]) if r["result"] else None) for r in rows]
                status, response = 200, {"jobs": records} if listing else records[0]
        handler.reply(status, response)
    finally:
        connection.close()
    return True
