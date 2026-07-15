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

from core.active_window import active_app_name
from core.processor import TextPipeline
from core.reformatter import Reformatter
from database import db_manager

# Default combo; overridden by config.json at startup.
DEFAULT_HOTKEY = "ctrl+shift+space"
DEFAULT_REFORMAT_HOTKEY = "ctrl+alt+r"

# Paste shortcut used for text injection. Plain Ctrl+V is accepted by virtually
# every app; Ctrl+Shift+V is not a paste shortcut in some apps (e.g. Outlook),
# which reject it with a warning beep and no paste.
PASTE_HOTKEY = "ctrl+v"

# How long to wait after Ctrl+V before restoring the previous clipboard. Slow
# apps (Outlook) read the clipboard asynchronously; restoring too early makes
# them paste the STALE previous content instead of the injected text. This runs
# on the background key thread *after* the paste, so it never delays how fast
# text appears to the user.
PASTE_SETTLE_SECONDS = 0.25


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
                # Inject the chunk via clipboard paste. We deliberately do NOT
                # restore the clipboard here: doing so per-chunk both races the
                # target app's async paste (slow apps like Outlook then paste the
                # stale previous clipboard) and serialises the key thread, making
                # dictation laggy. The original clipboard is saved once at session
                # start and restored once when dictation stops.
                pyperclip.copy(text + " ")
                keyboard.send(PASTE_HOTKEY)
                time.sleep(0.02)  # brief spacing so consecutive pastes land in order
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[Betaal][hotkey] Clipboard paste failed: {exc}")

    def _restore_clipboard(self, saved: str):
        """Restore the pre-session clipboard, once all chunks have been pasted.

        Waits for the injection queue to drain and the final paste to settle so
        we never overwrite the clipboard before the target app has read it.
        """
        deadline = time.time() + 3.0
        while not self._key_queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(PASTE_SETTLE_SECONDS)  # let the last paste be consumed
        try:
            pyperclip.copy(saved)
        except Exception as exc:  # pragma: no cover - clipboard backend dependent
            print(f"[Betaal][hotkey] Clipboard restore failed: {exc}")

    def _dictation_loop(self):
        # One "session" spans a full hotkey press -> press cycle. We inject each
        # recognized chunk live (so text appears as you speak) but accumulate the
        # whole session and log a SINGLE entry when dictation stops.
        session_parts: list[str] = []
        session_start = time.time()
        # Snapshot the user's clipboard once; restored after the session ends so
        # live injection never races a per-chunk restore.
        try:
            saved_clipboard = pyperclip.paste()
        except Exception:  # pragma: no cover - clipboard backend dependent
            saved_clipboard = ""
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
            self._restore_clipboard(saved_clipboard)

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

    @staticmethod
    def _wait_modifiers_released(timeout=1.5):
        """Wait for the user to let go of the hotkey modifier keys.

        The reformat hotkey (e.g. ctrl+alt+r) means its modifiers are still held
        when the handler fires. Sending synthetic Ctrl+A / Ctrl+C / Ctrl+V while
        they are down (a) merges into the wrong combo so nothing is copied
        ("No text selected"), and (b) desyncs the OS modifier state, which stops
        later global hotkeys (dictation) from firing. Waiting for release fixes
        both.
        """
        mods = ("ctrl", "shift", "alt", "left windows", "right windows")
        end = time.time() + timeout
        while time.time() < end:
            try:
                if not any(keyboard.is_pressed(m) for m in mods):
                    return
            except Exception:
                return
            time.sleep(0.02)

    def _reformat_worker(self):
        original = None
        try:
            if self._reformatter.status != "ready":
                self._emit_llm_state("loading")
                self._reformatter.load()
                self._emit_llm_state("ready")

            # Capture which app is focused BEFORE we touch the clipboard, so the
            # LLM can align its output to that app (mail, editor, etc.).
            app_name = active_app_name()

            # Wait for the hotkey modifiers to be released so our synthetic
            # Ctrl+A/C/V are clean and don't leave modifiers stuck (which would
            # break the dictation hotkey afterwards).
            self._wait_modifiers_released()

            # Capture the entire text on screen: select all, then copy.
            original = pyperclip.paste()
            pyperclip.copy("")
            keyboard.send("ctrl+a")
            time.sleep(0.05)
            keyboard.send("ctrl+c")
            time.sleep(0.15)
            content = pyperclip.paste()
            if not content.strip():
                print("[Betaal][reformat] No text captured; nothing to reformat")
                pyperclip.copy(original)
                self._emit_llm_state("ready")
                return

            self._emit_llm_state("reformatting")
            result = self._reformatter.reformat(content, app_name=app_name)
            if result:
                pyperclip.copy(result)
                keyboard.send("ctrl+a")  # reselect all so paste replaces it
                time.sleep(0.05)
                keyboard.send(PASTE_HOTKEY)
                time.sleep(PASTE_SETTLE_SECONDS)
                pyperclip.copy(original)  # restore the user's clipboard
                print(f"[Betaal] Reformatted screen text (app: {app_name or 'unknown'})")
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
