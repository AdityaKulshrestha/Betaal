"""Shared configuration load/save for Betaal.

A single source of truth for reading and writing ``config.json`` so the engine
entry point and the desktop UI stay in sync.
"""

from __future__ import annotations

import json
import os

from core import paths
from core.model_registry import DEFAULT_LLM_DISPLAY, DEFAULT_MODEL_DISPLAY

CONFIG_PATH = str(paths.config_path())

DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+space",
    "vad_threshold": 0.35,
    "asr_model": DEFAULT_MODEL_DISPLAY,
    "asr_device": "GPU",
    "log_transcript": False,
    "user_name": "there",
    "min_silence_ms": 300,
    "max_segment_seconds": 15,
    "reformat_hotkey": "ctrl+alt+r",
    "llm_model": DEFAULT_LLM_DISPLAY,
    "llm_device": "CPU",
}


def load_config() -> dict:
    """Read config.json, creating it with defaults if missing or invalid."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                return {**DEFAULT_CONFIG, **json.load(fh)}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Betaal] Bad config, using defaults: {exc}")
    save_config(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    """Persist the given config dict to config.json."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    except OSError as exc:
        print(f"[Betaal] Could not write config: {exc}")
