"""Isolated MLX inference; imports are optional until execution."""
import json
import platform
import sys
import time
from pathlib import Path


def check_runtime():
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise ValueError("MLX jobs require an Apple Silicon Mac and an ARM Python interpreter.")
    import mlx.core  # noqa: F401
    import mlx_lm  # noqa: F401


def check_model(path):
    check_runtime()
    directory = Path(path).resolve()
    if not (directory / "config.json").is_file() or not any(directory.glob("*.safetensors")):
        raise ValueError("Model directory must contain config.json and safetensors weights.")
    return str(directory)


class ModelSession:
    """Load once, then generate independent prompts using the same weights."""
    def __init__(self, path):
        self.path = path
        self.model = None
        self.tokenizer = None

    def infer(self, job):
        from mlx_lm import load
        if self.model is None:
            self.model, self.tokenizer = load(self.path)
        return generate_text(self.model, self.tokenizer, job)


def generate_text(model, tokenizer, job):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    prompt = job["prompt"]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=job["max_tokens"],
                    sampler=make_sampler(temp=job["temperature"]), verbose=False)
    if len(text.encode("utf-8")) > 24000:
        raise ValueError("Output exceeds the initial result size limit.")
    return text


def infer(path, job):
    return ModelSession(path).infer(job)


def serve(path, directory):
    session = ModelSession(path)
    directory = Path(directory)
    request = directory / "request.json"
    while True:
        if not request.exists():
            time.sleep(0.05)
            continue
        failed = False
        try:
            job = json.loads(request.read_text(encoding="utf-8"))
            request.unlink()
            result = {"text": session.infer(job)}
        except Exception:
            result = {"error": "MLX inference failed; check local model compatibility and available memory."}
            failed = True
        temporary = directory / "result.tmp"
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(directory / "result.json")
        if failed:
            return  # Discard potentially unhealthy model state.


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        serve(sys.argv[2], sys.argv[3])
        sys.exit(0)
    try:
        job = json.load(sys.stdin)
        result = {"text": infer(sys.argv[1], job)}
    except Exception:
        result = {"error": "MLX inference failed; check local model compatibility and available memory."}
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
