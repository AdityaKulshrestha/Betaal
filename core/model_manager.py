"""Model bootstrap helpers.

Downloads required assets into the local .models/ directory when missing.
"""

from __future__ import annotations

import argparse
import os
from urllib.request import urlopen
from typing import Dict

from core.model_registry import DEFAULT_MODEL_DISPLAY, resolve_model_id

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE_DIR, ".models")
ASR_DIR = os.path.join(MODELS_DIR, "asr")
VAD_DIR = os.path.join(MODELS_DIR, "vad")
VAD_ONNX_PATH = os.path.join(VAD_DIR, "silero_vad.onnx")
VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "master/src/silero_vad/data/silero_vad.onnx"
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def ensure_asr_model(display_name: str) -> str:
    """Ensure ASR model is present locally and return its folder path."""
    repo_id = resolve_model_id(display_name)
    model_dir = os.path.join(ASR_DIR, _safe_repo_name(repo_id))
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return model_dir

    _ensure_dir(ASR_DIR)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download ASR models."
        ) from exc

    print(f"[Betaal][models] Downloading ASR model: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    return model_dir


def ensure_vad_model() -> str:
    """Ensure ONNX Silero VAD model exists and return file path."""
    if os.path.isfile(VAD_ONNX_PATH):
        return VAD_ONNX_PATH

    _ensure_dir(VAD_DIR)
    try:
        with urlopen(VAD_ONNX_URL) as response, open(VAD_ONNX_PATH, "wb") as handle:
            handle.write(response.read())
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not download Silero VAD ONNX model: {exc}") from exc

    print("[Betaal][models] Downloaded Silero VAD ONNX model from GitHub")
    return VAD_ONNX_PATH


def ensure_required_models(display_name: str | None = None) -> Dict[str, str]:
    """Ensure all runtime models exist locally."""
    selected = display_name or DEFAULT_MODEL_DISPLAY
    return {
        "asr_model_dir": ensure_asr_model(selected),
        "vad_model_path": ensure_vad_model(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Betaal runtime models")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DISPLAY,
        help="ASR display model name (as shown in GUI)",
    )
    args = parser.parse_args()
    paths = ensure_required_models(args.model)
    print("[Betaal][models] Ready:")
    for key, value in paths.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
