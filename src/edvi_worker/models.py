"""Prepare an MLX model before advertising it to the coordinator."""
import json
import re
from pathlib import Path

from .inference import check_runtime


DEFAULT_MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_MODEL_ALIAS = "local-llm"
MODEL_FILES = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"]


class ModelDownloadError(Exception):
    pass


def complete_model(path):
    directory = Path(path)
    try:
        config = json.loads((directory / "config.json").read_text())
        if not isinstance(config, dict) or not config.get("model_type"):
            return False
        weights = list(directory.glob("*.safetensors"))
        if not weights or any(weight.stat().st_size == 0 for weight in weights):
            return False
        index = directory / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                return False
            for name in mapping.values():
                if not isinstance(name, str) or Path(name).name != name or not (directory / name).is_file():
                    return False
        return ((directory / "tokenizer_config.json").is_file()
                and any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_model(state_dir, repo_id=DEFAULT_MODEL_REPO):
    # Reject incompatible hardware/missing runtime before downloading anything.
    try:
        check_runtime()
    except (ImportError, ValueError, OSError):
        raise ModelDownloadError("Automatic MLX setup requires Apple Silicon and mlx-lm. Install with python -m pip install -e '.[mlx]'.") from None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ModelDownloadError("Model repository must be a Hugging Face owner/model ID.")
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        cache = Path(state_dir) / "models"
        cache.mkdir(parents=True, exist_ok=True)
        arguments = dict(repo_id=repo_id, cache_dir=str(cache), allow_patterns=MODEL_FILES)
        try:
            path = snapshot_download(**arguments, local_files_only=True)
        except LocalEntryNotFoundError:
            path = None
        if path and complete_model(path):
            print(f"Using cached MLX model: {repo_id}", flush=True)
            return str(path)
        print(f"Downloading MLX model: {repo_id} (first startup needs internet)", flush=True)
        path = snapshot_download(**arguments)
        if not complete_model(path):
            raise ModelDownloadError("Downloaded model is incomplete or lacks MLX model/tokenizer files. Retry startup or choose --model-repo.")
        print(f"MLX model ready: {path}", flush=True)
        return str(path)
    except ModelDownloadError:
        raise
    except Exception:
        # Do not echo upstream exceptions, which can contain URLs or credentials.
        raise ModelDownloadError("Model download failed. Check internet access, disk space, repository access, and HF_TOKEN for gated models, then restart the worker.") from None
