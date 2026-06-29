"""Background global hotkey listener for Betaal.

Registers a system-wide hotkey via the ``keyboard`` library. On trigger it runs
the simulated text pipeline and types the result into whichever window holds
focus, then logs the session to the analytics database.
"""

import threading

import keyboard

from core.processor import TextPipeline
from database import db_manager

# Default combo; overridden by config.json at startup.
DEFAULT_HOTKEY = "ctrl+shift+space"


class HotkeyListener:
    """Runs the global hotkey hook on a dedicated daemon thread."""

    def __init__(self, hotkey=DEFAULT_HOTKEY, type_delay=0.01):
        self._hotkey = hotkey or DEFAULT_HOTKEY
        self._type_delay = type_delay
        self._pipeline = TextPipeline()
        self._thread = None
        self._busy = threading.Lock()

    def _on_trigger(self):
        # Avoid overlapping injections if the hotkey is pressed repeatedly.
        if not self._busy.acquire(blocking=False):
            return
        try:
            text, duration = self._pipeline.process()
            keyboard.write(text, delay=self._type_delay)
            words = len(text.split())
            db_manager.log_entry(words, duration)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            print(f"[Betaal][hotkey] Injection failed: {exc}")
        finally:
            self._busy.release()

    def _run(self):
        try:
            keyboard.add_hotkey(self._hotkey, self._on_trigger)
            print(f"[Betaal] Listening for hotkey: {self._hotkey}")
            keyboard.wait()  # Blocks this thread, keeping the hook alive.
        except Exception as exc:
            print(f"[Betaal][hotkey] Could not register hook '{self._hotkey}': {exc}")
            print("[Betaal] Hint: run as Administrator if hooks are blocked.")

    def start(self):
        """Launch the listener on a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._thread
