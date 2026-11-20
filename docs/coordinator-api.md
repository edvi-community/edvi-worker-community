# Coordinator API contract (v1)

The `edvi worker start` client and local `edvi coordinator start` server implement
this contract. The local server uses one shared token for all workers and dashboard users.

## Transport and authentication

Use JSON over HTTPS, with `Content-Type: application/json` and
`Authorization: Bearer <worker-token>` on every request. Plain HTTP is permitted
only for local development. Provision tokens outside source control; never log
them. The shared token grants access to the local coordinator API.

All timestamps use UTC RFC 3339 strings. API routes start with `/v1`.
The worker generates a UUID once and persists it locally across restarts.
Hostname is descriptive metadata, not a unique identifier.

## Register a worker

`PUT /v1/workers/{worker_id}`

This operation is idempotent: repeating it for the same ID updates capabilities
without creating another worker.

Request:

```json
{
  "worker_version": "0.1.0",
  "capabilities": {
    "os": "macOS",
    "os_version": "25.0.0",
    "architecture": "arm64",
    "processor": "arm",
    "memory_gb": 16.0,
    "logical_cpus": 8,
    "hostname": "example-worker",
    "accelerator": "Apple Silicon GPU",
    "runtime_hint": "MLX"
  }
}
```

Return `201` for creation or `200` for an update, with:

```json
{
  "worker_id": "fa26bec2-59b0-4c38-9957-a1681482ee29",
  "heartbeat_interval_seconds": 30,
  "offline_after_seconds": 90
}
```

The capability fields match `edvi info --json`. `memory_gb` is a nonnegative
number expressed in GiB despite its existing name; zero means unknown.
`logical_cpus` is a positive integer. Other capability fields and
`worker_version` are nonempty strings. Runtime hints describe potential support;
they do not prove that an execution runtime is installed.
The heartbeat interval is a positive integer; the offline threshold must be at
least three heartbeat intervals. Reject invalid requests with `400`.

## Send a heartbeat

`POST /v1/workers/{worker_id}/heartbeat`

Request:

```json
{
  "status": "idle"
}
```

Heartbeat status is either `idle` or `stopping`. Here `idle` means available to
the coordinator; active jobs are tracked separately in the jobs table. Return `200` with:

```json
{
  "worker_id": "fa26bec2-59b0-4c38-9957-a1681482ee29",
  "received_at": "2026-10-01T12:00:00Z"
}
```

The coordinator records its own receipt time, so worker clock drift cannot
affect liveness. Repeated heartbeats are safe. A worker becomes offline when
its last registration or heartbeat exceeds `offline_after_seconds`.
A `stopping` heartbeat marks it offline immediately. `404` means registration
is missing and the worker should register again.

## List workers

`GET /v1/workers` requires the same bearer token and returns `200` with a
`workers` array. Each entry contains `worker_id`, `worker_version`,
`capabilities`, `last_seen` (UTC RFC 3339), and `status` (`idle` or `offline`).
Offline status is calculated at read time from the stored receipt timestamp
and configured threshold; no background cleanup is needed.

## Errors and retries

Errors use this shape, with a stable machine-readable code and a message that
does not contain credentials:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "logical_cpus must be a positive integer"
  }
}
```

| Result | Worker behavior |
| --- | --- |
| `400` invalid request | Stop and report the configuration or protocol error. |
| `401` invalid token / `403` forbidden | Stop and report the authentication error. |
| `404` on heartbeat | Register again before sending another heartbeat. |
| `429` rate limited | Honor a valid integer-seconds `Retry-After`; otherwise use backoff. |
| `5xx`, timeout, connection failure | Retry with backoff. |
| Other unexpected responses or malformed success JSON | Stop and report a protocol error. |

Use a 10-second request timeout. Start retries at 1 second, double up to 60
seconds, and add random jitter of up to 20 percent. Reset backoff after success.
Send the first heartbeat immediately after registration, then follow the
returned interval. Run one request at a time; do not accumulate queued
heartbeats while disconnected. Ctrl+C interrupts waiting and sends one
best-effort stopping heartbeat with a 3-second timeout before exiting.

## Acceptance checks for the next implementation

- Registration includes current capabilities and reuses the persistent ID.
- Repeated registration creates one coordinator record.
- Heartbeats respect the returned interval and coordinator liveness rules.
- Missing registration triggers registration again.
- Authentication errors stop the worker without exposing tokens.
- Network failures, rate limits, and server failures retry without overlapping requests.
- Ctrl+C exits promptly during normal operation and retry waiting.

Use a local coordinator for these integration checks.

## MLX job endpoints

All endpoints require the coordinator bearer token. Registration capabilities
may include `mlx_models`, a list of local model aliases (at most 16).

- `POST /v1/jobs`: submit `model`, `prompt`, `max_tokens`, `temperature`, and
  `timeout_seconds`; returns `201` with `job_id` and `status`.
- `GET /v1/jobs`: returns a `jobs` array containing status, request, worker ID,
  and result. `GET /v1/jobs/{job_id}` returns one entry or `404`.
- `POST /v1/jobs/claim`: send `worker_id`; returns `worker_id` and `job` (null
  if none eligible). A claimed job includes its request fields, `job_id`,
  `lease_id`, and `lease_until` (Unix seconds). Claims are atomic; retrying
  returns the same active lease for that worker.
- `POST /v1/jobs/{job_id}/result`: send `worker_id`, `lease_id`, and `result`
  containing either `text` or `error`. Returns `200` with worker and job IDs.
  Repeating the same result is safe; stale leases or conflicting results return `409`.

Jobs have fixed leases lasting their timeout plus 30 seconds; heartbeats do not
extend leases. Expired jobs return to the queue on the next job API request.
Limits and local model setup are documented in the README. This initial queue
uses at-least-once execution and no automatic retry limit for disconnected workers.
