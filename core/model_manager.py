"""Model bootstrap helpers.

Downloads required assets into the local .models/ directory when missing.
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.request import urlopen
from typing import Dict

from core.model_registry import DEFAULT_MODEL_DISPLAY, resolve_model_id

_META_JSON = "ov_cohere_transcribe_kvcache.json"

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


def _is_complete_ov_model(model_dir: str) -> bool:
    """Return True if model_dir holds a complete OV IR model (meta + xml + bin)."""
    meta_path = os.path.join(model_dir, _META_JSON)
    if not os.path.isfile(meta_path):
        return False
    try:
        meta = json.loads(open(meta_path, encoding="utf-8").read())
    except (OSError, ValueError):
        return False

    for key in ("encoder_ir", "decoder_ir", "decoder_with_past_ir"):
        xml_name = meta.get(key)
        if not xml_name:
            return False
        xml_path = os.path.join(model_dir, xml_name)
        bin_path = os.path.splitext(xml_path)[0] + ".bin"
        if not os.path.isfile(xml_path) or not os.path.isfile(bin_path):
            return False
    return True


def _find_local_asr_model(repo_id: str) -> str | None:
    """Locate an already-present complete OV model without downloading."""
    candidates = [
        os.path.join(ASR_DIR, _safe_repo_name(repo_id)),
        os.path.join(ASR_DIR, _safe_repo_name(repo_id), "models"),
        os.path.join(MODELS_DIR, "models"),
    ]
    for candidate in candidates:
        if _is_complete_ov_model(candidate):
            return candidate
    return None


def ensure_asr_model(display_name: str) -> str:
    """Ensure ASR model is present locally and return its folder path."""
    repo_id = resolve_model_id(display_name)

    local = _find_local_asr_model(repo_id)
    if local is not None:
        return local

    model_dir = os.path.join(ASR_DIR, _safe_repo_name(repo_id))
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
