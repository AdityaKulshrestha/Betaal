"""Model display names, registry IDs, and backend kinds used by the app."""

from __future__ import annotations

# Each selectable model maps to its Hugging Face repo id and the backend that
# knows how to run it:
#   - "cohere_ov"     : Cohere Transcribe OpenVINO IR (manual KV-cache decode)
#   - "whisper_genai" : Whisper via OpenVINO GenAI WhisperPipeline
MODELS = {
    "Cohere-transcribe": {
        "id": "Aditya02/cohere-transcribe-03-2026-ov-fp16",
        "backend": "cohere_ov",
    },
    "Whisper Large": {
        "id": "OpenVINO/whisper-large-v3-int4-ov",
        "backend": "whisper_genai",
    },
}

DEFAULT_MODEL_DISPLAY = "Whisper Large"

# Backward-compatible name -> id mapping.
MODEL_DISPLAY_TO_ID = {name: meta["id"] for name, meta in MODELS.items()}


def _entry(display_name: str) -> dict:
    return MODELS.get(display_name, MODELS[DEFAULT_MODEL_DISPLAY])


def resolve_model_id(display_name: str) -> str:
    """Map a GUI display name to the actual model registry ID."""
    return _entry(display_name)["id"]


def resolve_backend(display_name: str) -> str:
    """Return the backend kind ("cohere_ov" or "whisper_genai") for a model."""
    return _entry(display_name)["backend"]


def list_model_names() -> list[str]:
    """Return all selectable model names for the GUI."""
    return list(MODELS.keys())
