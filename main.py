"""Betaal entry point: load config, start hotkey listener, launch GUI."""

import json
import os

from core.hotkey_listener import HotkeyListener
from core.model_registry import DEFAULT_MODEL_DISPLAY
from database import db_manager
from gui.settings_window import SettingsWindow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+space",
    "vad_threshold": 0.5,
    "asr_model": DEFAULT_MODEL_DISPLAY,
    "log_transcript": True,
}


def load_config():
    """Read config.json, creating it with defaults if missing or invalid."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            return {**DEFAULT_CONFIG, **cfg}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Betaal] Bad config, using defaults: {exc}")
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=2)
    except OSError as exc:
        print(f"[Betaal] Could not write config: {exc}")
    return dict(DEFAULT_CONFIG)


def main():
    config = load_config()
    db_manager.init_db()

    listener = HotkeyListener(
        hotkey=config["hotkey"],
        model_name=config["asr_model"],
        vad_threshold=config["vad_threshold"],
        log_transcript=config["log_transcript"],
    )
    listener.start()

    app = SettingsWindow(config)
    app.mainloop()


if __name__ == "__main__":
    main()
