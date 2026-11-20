from __future__ import annotations

import argparse
import os
import sqlite3
import platform
from pathlib import Path

from . import __version__
from .system_info import collect_system_info, info_as_json
from .worker import WorkerError, run_worker
from .coordinator import run_coordinator
from .models import DEFAULT_MODEL_ALIAS


def show_info(as_json: bool = False) -> None:
    info = collect_system_info()
    if as_json:
        print(info_as_json(info))
        return

    print("Edvi Worker")
    print("-----------")
    print(f"OS:            {info.os} {info.os_version}")
    print(f"Architecture:  {info.architecture}")
    print(f"Processor:     {info.processor}")
    print(f"Memory:        {info.memory_gb} GB")
    print(f"Logical CPUs:  {info.logical_cpus}")
    print(f"Accelerator:   {info.accelerator}")
    print(f"Runtime hint:  {info.runtime_hint}")
    print(f"Hostname:      {info.hostname}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="edvi", description="Edvi cross-platform compute worker")
    parser.add_argument("--version", action="version", version=f"edvi-worker {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("info", help="Show this computer's compute capabilities")
    info_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    worker_parser = subparsers.add_parser("worker", help="Connect to an Edvi coordinator")
    worker_commands = worker_parser.add_subparsers(dest="worker_command", required=True)
    start_parser = worker_commands.add_parser("start", help="Register and send heartbeats")
    start_parser.add_argument("--coordinator-url", default=os.environ.get("EDVI_COORDINATOR_URL"),
                              help="Coordinator base URL (or EDVI_COORDINATOR_URL)")
    start_parser.add_argument("--mlx-model", help="Model name advertised for MLX jobs")
    start_parser.add_argument("--model-path", type=Path, help="Local MLX model directory")
    start_parser.add_argument("--model-repo", help="Hugging Face MLX repository to download instead of the small default model")
    start_parser.add_argument("--heartbeat-only", action="store_true", help="Disable MLX jobs and automatic model setup")
    start_parser.add_argument("--state-dir", type=Path,
                              default=Path(os.environ.get("EDVI_STATE_DIR", str(Path.home() / ".edvi"))),
                              help="Persistent worker ID directory (or EDVI_STATE_DIR)")

    coordinator_parser = subparsers.add_parser("coordinator", help="Run a local coordinator")
    coordinator_commands = coordinator_parser.add_subparsers(dest="coordinator_command", required=True)
    server_parser = coordinator_commands.add_parser("start", help="Serve the coordinator API on loopback")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--database", type=Path, default=Path(".edvi/coordinator.sqlite3"))
    server_parser.add_argument("--heartbeat-interval", type=int, default=30)
    server_parser.add_argument("--offline-after", type=int, default=90)
    args = parser.parse_args()

    if args.command == "info":
        show_info(args.json)
    elif args.command == "coordinator":
        try:
            run_coordinator(args.host, args.port, args.database,
                            os.environ.get("EDVI_COORDINATOR_TOKEN", ""),
                            args.heartbeat_interval, args.offline_after)
        except (ValueError, OSError, sqlite3.Error):
            parser.exit(1, "edvi: Cannot start coordinator; check token, loopback host, port, database permissions, and heartbeat timing.\n")
    elif args.command == "worker":
        if not args.coordinator_url:
            parser.error("Set --coordinator-url or EDVI_COORDINATOR_URL.")
        try:
            if args.model_path and args.model_repo:
                parser.error("Choose --model-path or --model-repo.")
            if args.heartbeat_only and (args.mlx_model or args.model_path or args.model_repo):
                parser.error("--heartbeat-only cannot be combined with model options.")
            apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
            alias = args.mlx_model or DEFAULT_MODEL_ALIAS
            if not args.heartbeat_only and (apple_silicon or args.mlx_model or args.model_path or args.model_repo):
                run_worker(args.coordinator_url, os.environ.get("EDVI_WORKER_TOKEN", ""), args.state_dir,
                           alias, args.model_path, args.model_repo)
            else:
                run_worker(args.coordinator_url, os.environ.get("EDVI_WORKER_TOKEN", ""), args.state_dir)
        except WorkerError as error:
            parser.exit(1, f"edvi: {error}\n")
        except KeyboardInterrupt:
            parser.exit(130, "Worker setup interrupted. Restart to resume the model download.\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
