# Edvi Worker Community

Edvi connects workers to a local job queue. This experimental developer preview
runs independent MLX inference jobs on Apple Silicon Macs. Windows and Linux
workers can report hardware and send heartbeats. Each job runs on one worker.

## Quick start

Create a Python 3.10+ virtual environment and install the project:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. On
Apple Silicon, install `python -m pip install '.[mlx]'` for inference.

Start the coordinator on loopback. Use a fresh shared token and enter it at
the prompt:

```sh
read -rs EDVI_COORDINATOR_TOKEN
export EDVI_COORDINATOR_TOKEN
edvi coordinator start
```

In a second terminal with the same virtual environment active, enter the same
token and start a worker:

```sh
read -rs EDVI_WORKER_TOKEN
export EDVI_WORKER_TOKEN
edvi worker start --coordinator-url http://127.0.0.1:8000
```

Use `--heartbeat-only` to try registration without installing MLX or fetching
model weights. For inference, open `http://127.0.0.1:8000/`, enter the token, and
submit a prompt using the default `local-llm` model. The worker downloads the
default weights on first use. The coordinator stores prompts and results locally.

To connect a second Mac, run `ssh -N -L 8000:127.0.0.1:8000 user@coordinator-mac`
on that Mac if SSH access is already configured, then start its worker with the
same local URL and token. Keep the coordinator bound to loopback.

See the [API contract](docs/coordinator-api.md) and [contribution guide](CONTRIBUTING.md).
Licensed under [Apache 2.0](LICENSE).
