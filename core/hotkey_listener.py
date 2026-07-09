"""Background global hotkey listener for Betaal.

Registers a system-wide hotkey via the ``keyboard`` library. On trigger it runs
the simulated text pipeline and types the result into whichever window holds
focus, then logs the session to the analytics database.
"""

import threading
import queue
import time

import keyboard
import pyperclip

from core.processor import TextPipeline
from database import db_manager

# Default combo; overridden by config.json at startup.
DEFAULT_HOTKEY = "ctrl+shift+space"

# Paste shortcut used for text injection. Ctrl+Shift+V works in terminals;
# most other apps also accept it (or fall back to Ctrl+V).
PASTE_HOTKEY = "ctrl+shift+v"


class HotkeyListener:
    """Runs the global hotkey hook on a dedicated daemon thread."""

    def __init__(
        self,
        hotkey=DEFAULT_HOTKEY,
        type_delay=0.01,
        model_name="Cohere-transcribe",
        vad_threshold=0.5,
        log_transcript=False,
    ):
        self._hotkey = hotkey or DEFAULT_HOTKEY
        self._type_delay = type_delay
        self._pipeline = TextPipeline(
            model_display_name=model_name,
            vad_threshold=vad_threshold,
            log_transcript=log_transcript,
        )
        self._thread = None
        self._key_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._key_thread = None
        self._dictation_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

    def _key_worker(self):
        while True:
            text = self._key_queue.get()
            if not text:
                continue
            try:
                previous = pyperclip.paste()
                pyperclip.copy(text + " ")
                keyboard.send(PASTE_HOTKEY)
                time.sleep(0.05)
                pyperclip.copy(previous)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[Betaal][hotkey] Clipboard paste failed: {exc}")

    def _dictation_loop(self):
        while not self._stop_event.is_set():
            try:
                text, duration = self._pipeline.process(
                    capture_seconds=3.0,
                    stop_event=self._stop_event,
                )
                if not text:
                    continue
                self._key_queue.put(text)
                words = len(text.split())
                db_manager.log_entry(words, duration)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                print(f"[Betaal][hotkey] Dictation loop error: {exc}")

    def _on_trigger(self):
        with self._state_lock:
            active = self._dictation_thread is not None and self._dictation_thread.is_alive()
            if active:
                self._stop_event.set()
                print("[Betaal] Dictation stopped")
                return

            self._stop_event.clear()
            self._dictation_thread = threading.Thread(target=self._dictation_loop, daemon=True)
            self._dictation_thread.start()
            print("[Betaal] Dictation started")

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
        self._key_thread = threading.Thread(target=self._key_worker, daemon=True)
        self._key_thread.start()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._thread
