"""Betaal entry point.

Default: launch the native PySide6 desktop app (a standalone Windows window)
which also runs the background dictation engine (global hotkey + ASR).

    python main.py            # native desktop app (default)
    python main.py headless   # engine only, no window (global hotkey)
"""

import sys
import threading

from core.config_store import load_config
from core.hotkey_listener import HotkeyListener
from database import db_manager


def main():
    """Launch the native desktop application."""
    from desktop import run

    run()


def headless():
    """Run the engine with no window: global hotkey + background dictation."""
    config = load_config()
    db_manager.init_db()

    listener = HotkeyListener(
        hotkey=config["hotkey"],
        model_name=config["asr_model"],
        vad_threshold=config["vad_threshold"],
        log_transcript=config["log_transcript"],
        min_silence_ms=config["min_silence_ms"],
        max_segment_seconds=config["max_segment_seconds"],
        reformat_hotkey=config["reformat_hotkey"],
        llm_model=config["llm_model"],
        llm_device=config["llm_device"],
        reformat_prompt=config["reformat_prompt"],
    )
    listener.start()

    print(
        f"[Betaal] Running headless. Hotkey: {config['hotkey']}. "
        f"Reformat: {config['reformat_hotkey']}. Press Ctrl+C to quit."
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("[Betaal] Shutting down.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "headless":
        headless()
    else:
        main()
