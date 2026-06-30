"""Model display names and backend IDs used by the app."""

from __future__ import annotations

MODEL_DISPLAY_TO_ID = {
    "Cohere-transcribe": "Aditya02/cohere-transcribe-03-2026-ov-fp16",
}

DEFAULT_MODEL_DISPLAY = "Cohere-transcribe"


def resolve_model_id(display_name: str) -> str:
    """Map a GUI display name to the actual model registry ID."""
    return MODEL_DISPLAY_TO_ID.get(display_name, MODEL_DISPLAY_TO_ID[DEFAULT_MODEL_DISPLAY])


def list_model_names() -> list[str]:
    """Return all selectable model names for the GUI."""
    return list(MODEL_DISPLAY_TO_ID.keys())
