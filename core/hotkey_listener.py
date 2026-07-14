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
from core.reformatter import DEFAULT_PROMPT, Reformatter
from database import db_manager

# Default combo; overridden by config.json at startup.
DEFAULT_HOTKEY = "ctrl+shift+space"
DEFAULT_REFORMAT_HOTKEY = "ctrl+shift+f"

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
        device="GPU",
        min_silence_ms=300.0,
        max_segment_seconds=None,
        reformat_hotkey=DEFAULT_REFORMAT_HOTKEY,
        llm_model="LFM2.5 350M",
        llm_device="CPU",
        reformat_prompt=None,
        on_note=None,
        on_state=None,
        on_llm_state=None,
    ):
        self._hotkey = hotkey or DEFAULT_HOTKEY
        self._type_delay = type_delay
        self._pipeline = TextPipeline(
            model_display_name=model_name,
            vad_threshold=vad_threshold,
            log_transcript=log_transcript,
            device=device,
            min_silence_ms=min_silence_ms,
            max_segment_seconds=max_segment_seconds,
        )
        self._reformat_hotkey = reformat_hotkey or DEFAULT_REFORMAT_HOTKEY
        self._reformatter = Reformatter(
            model_display_name=llm_model,
            device=llm_device,
            prompt=reformat_prompt or DEFAULT_PROMPT,
        )
        self._on_note = on_note
        self._on_state = on_state
        self._on_llm_state = on_llm_state
        self._thread = None
        self._key_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._key_thread = None
        self._dictation_thread = None
        self._llm_thread = None
        self._reformat_lock = threading.Lock()
        self._reformat_busy = False
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
        # One "session" spans a full hotkey press -> press cycle. We inject each
        # recognized chunk live (so text appears as you speak) but accumulate the
        # whole session and log a SINGLE entry when dictation stops.
        session_parts: list[str] = []
        session_start = time.time()
        # Open one continuous mic stream for the whole session so audio keeps
        # buffering during ASR (no dropped packets between windows).
        self._pipeline.start_capture()
        try:
            while not self._stop_event.is_set():
                try:
                    text, _duration = self._pipeline.process(
                        stop_event=self._stop_event,
                    )
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"[Betaal][hotkey] Dictation loop error: {exc}")
                    continue
                if text:
                    self._key_queue.put(text)  # live-inject into the focused app
                    session_parts.append(text)
            # Flush the final in-progress utterance (no trailing silence yet).
            try:
                text, _duration = self._pipeline.flush()
                if text:
                    self._key_queue.put(text)
                    session_parts.append(text)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[Betaal][hotkey] Dictation flush error: {exc}")
        finally:
            self._pipeline.stop_capture()

        self._finalize_session(session_parts, time.time() - session_start)

    def _finalize_session(self, parts, duration):
        """Log the whole session as one entry once dictation has stopped."""
        full_text = " ".join(p for p in parts if p).strip()
        if not full_text:
            return
        words = len(full_text.split())
        db_manager.log_entry(words, duration, text=full_text)
        if self._on_note is not None:
            try:
                self._on_note(full_text, words, duration)
            except Exception as exc:  # pragma: no cover - callback guard
                print(f"[Betaal][hotkey] on_note callback failed: {exc}")

    def _emit_state(self, recording):
        if self._on_state is not None:
            try:
                self._on_state(recording)
            except Exception as exc:  # pragma: no cover - callback guard
                print(f"[Betaal][hotkey] on_state callback failed: {exc}")

    def _emit_llm_state(self, status):
        if self._on_llm_state is not None:
            try:
                self._on_llm_state(status)
            except Exception as exc:  # pragma: no cover - callback guard
                print(f"[Betaal][hotkey] on_llm_state callback failed: {exc}")

    def _load_reformatter(self):
        """Warm up the reformatter LLM in the background and report status."""
        self._emit_llm_state("loading")
        try:
            self._reformatter.load()
            self._emit_llm_state("ready")
            print("[Betaal] Reformatter LLM ready")
        except Exception as exc:  # pragma: no cover - runtime/model dependent
            print(f"[Betaal][reformat] LLM load failed: {exc}")
            self._emit_llm_state("error")

    def _on_reformat(self):
        """Hotkey handler: reformat the current selection / clipboard text."""
        with self._reformat_lock:
            if self._reformat_busy:
                return
            self._reformat_busy = True
        threading.Thread(target=self._reformat_worker, daemon=True).start()

    def _reformat_worker(self):
        original = None
        try:
            if self._reformatter.status != "ready":
                self._emit_llm_state("loading")
                self._reformatter.load()
                self._emit_llm_state("ready")

            # Grab the current selection: clear the clipboard, copy, then read.
            original = pyperclip.paste()
            pyperclip.copy("")
            keyboard.send("ctrl+c")
            time.sleep(0.15)
            content = pyperclip.paste()
            if not content.strip():
                print("[Betaal][reformat] No text selected; nothing to reformat")
                pyperclip.copy(original)
                self._emit_llm_state("ready")
                return

            self._emit_llm_state("reformatting")
            result = self._reformatter.reformat(content)
            if result:
                pyperclip.copy(result)
                keyboard.send(PASTE_HOTKEY)
                time.sleep(0.05)
                pyperclip.copy(original)  # restore the user's clipboard
                print("[Betaal] Reformatted selection")
            else:
                pyperclip.copy(original)
            self._emit_llm_state("ready")
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[Betaal][reformat] Reformat failed: {exc}")
            if original is not None:
                try:
                    pyperclip.copy(original)
                except Exception:
                    pass
            self._emit_llm_state("error")
        finally:
            with self._reformat_lock:
                self._reformat_busy = False

    def _on_trigger(self):
        with self._state_lock:
            active = self._dictation_thread is not None and self._dictation_thread.is_alive()
            if active:
                self._stop_event.set()
                print("[Betaal] Dictation stopped")
                self._emit_state(False)
                return

            self._stop_event.clear()
            self._dictation_thread = threading.Thread(target=self._dictation_loop, daemon=True)
            self._dictation_thread.start()
            print("[Betaal] Dictation started")
            self._emit_state(True)

    def toggle(self):
        """Programmatically start/stop dictation (same as pressing the hotkey)."""
        self._on_trigger()

    def is_recording(self):
        """Return True when the dictation loop is currently active."""
        with self._state_lock:
            return self._dictation_thread is not None and self._dictation_thread.is_alive()

    def _run(self):
        try:
            keyboard.add_hotkey(self._hotkey, self._on_trigger)
            print(f"[Betaal] Listening for hotkey: {self._hotkey}")
            if self._reformat_hotkey:
                keyboard.add_hotkey(self._reformat_hotkey, self._on_reformat)
                print(f"[Betaal] Listening for reformat hotkey: {self._reformat_hotkey}")
            keyboard.wait()  # Blocks this thread, keeping the hook alive.
        except Exception as exc:
            print(f"[Betaal][hotkey] Could not register hook '{self._hotkey}': {exc}")
            print("[Betaal] Hint: run as Administrator if hooks are blocked.")

    def start(self):
        """Launch the listener on a background daemon thread."""
        self._key_thread = threading.Thread(target=self._key_worker, daemon=True)
        self._key_thread.start()

        # Warm up the reformatter LLM without blocking dictation startup.
        self._llm_thread = threading.Thread(target=self._load_reformatter, daemon=True)
        self._llm_thread.start()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._thread
