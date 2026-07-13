# Betaal

Betaal is an offline, Windows background dictation app that produces real-time
transcriptions (in the spirit of Wispr Flow).

- **Standalone native Windows app** — PySide6 (Qt) desktop window, dark theme.
  No browser, no WebView, no web toolchain.
- **System tray + collapsible settings sidebar**.
- Global hotkey **`Ctrl+Shift+Space`** to start/stop continuous dictation.
- Microphone capture at 16 kHz, mono, `float32`.
- Voice-activity chunking with **Silero VAD** (ONNX).
- **Model-agnostic** speech-to-text. First backend: Cohere Transcribe (OpenVINO IR),
  running fully in-process via OpenVINO (no HTTP).
- Injects recognized text into **whatever app currently has focus** and logs it
  as an activity entry.
- Usage analytics (words dictated, time saved, avg daily words) in SQLite.
- Fully offline after a one-time model download on first run.

> **Native, not browser-based.** The UI is a PySide6 / Qt window that runs the
> dictation engine in the *same* process — no Node, Rust, or WebView2 required,
> and it packages to a single `.exe`.

## Architecture

```
        ┌────────────────────────────────────────┐
        │         Native UI (PySide6 / Qt)        │   activity logger:
        │   dashboard · activity log · sidebar    │   dashboard + live log;
        │         system tray · dark theme        │   never a paste target
        └────────────────────┬───────────────────┘
                     in-process calls + Qt signals
        ┌────────────────────┴───────────────────┐
        │         Dictation engine (Python)       │──▶ Global hotkey (keyboard)
        │   sounddevice · Silero VAD · OpenVINO   │    keystroke/clipboard inject
        │         ASR · SQLite analytics          │    into the FOCUSED app
        └─────────────────────────────────────────┘
```

Everything runs in **one Python process**: the Qt window and the background
hotkey engine share memory directly — no IPC, no sockets, no sidecar. Engine
callbacks (which fire on background threads) are marshalled onto the UI thread
with Qt signals.

### How it works

- Betaal runs **in the background** (system tray). You work in any app —
  Gmail, Chrome, Word, a text field — and press the **global hotkey**.
- Recognized text is injected **into whatever app currently has focus**
  (via clipboard paste, restoring your previous clipboard afterward). It is
  **never** shown or pasted onto Betaal's own window.
- The desktop window is a **logger**: it displays what happened — usage metrics
  and a live activity log of every dictation — plus the settings sidebar. It
  is read-only and does not capture your dictation focus.

Pipeline (dictation):

```
mic ─▶ AudioCapture ─▶ Silero VAD ─▶ [queue] ─▶ ASR (OpenVINO) ─▶ inject into focused app + log entry
       16k mono f32     speech segs    bounded     text
```

## Repository layout

```
Betaal/
  main.py                   Entry point: native app (default) · headless
  config.json               Persisted user configuration
  pyproject.toml            Python package + dependencies
  app_metrics.db            SQLite usage stats + notes (auto-created)
  assets/                   App icon (betaal.png / betaal.ico)
  .models/                  Auto-downloaded ASR + VAD assets (first run)
  desktop/                  Native PySide6 UI (the app)
    main_window.py          Window: sidebar · dashboard · activity log · tray
    widgets.py              Metric cards, collapsible sections, note cards
    theme.py                Dark Qt stylesheet + palette
  core/
    config_store.py         Shared config load/save
    hotkey_listener.py      Background global hotkey + dictation loop
    processor.py            AudioCapture · Silero VAD · OpenVINO ASR pipeline
    model_manager.py        First-run model download/verify
    model_registry.py       Display-name → registry-ID mapping
    speaker.py              Speaker-enrollment WAV recorder
  database/
    db_manager.py           SQLite logging (words, duration, time saved, notes)
```

## Setup

The whole app is Python. You only need **Python 3.10+** and
[`uv`](https://docs.astral.sh/uv/) — no Node, Rust, MSVC, or WebView2.

From the `Betaal/` folder:

```powershell
uv venv
.venv\Scripts\activate            # Windows
uv sync
```

Launch the **native desktop app** (this is the default):

```powershell
uv run betaal
```

That opens the standalone Betaal window — dashboard (words, time saved, avg
daily words), the live activity log, and a collapsible sidebar to pick the ASR
model, tune VAD sensitivity, and enroll speaker samples. Closing the window
hides it to the system tray; the global hotkey keeps working in the background.

Other modes:

```powershell
python main.py headless   # engine only, no window (global hotkey)
```

> The `keyboard` library needs Administrator rights for global hooks in some
> apps. Run the terminal as Administrator if the hotkey is blocked.

## Usage

Put focus in any app (Gmail, Chrome, Word, a text box), then press the global
hotkey (default `ctrl+shift+space`) once to start continuous dictation and
again to stop. Recognized text is typed **into that focused app** at your
cursor and logged in Betaal's activity view. Settings persist to `config.json`;
analytics and the activity log persist in `app_metrics.db`.

## Models

`Cohere-transcribe` maps to the registry ID
`Aditya02/cohere-transcribe-03-2026-ov-fp16`. On first run, Betaal
auto-downloads missing assets into `.models/`:

- ASR OpenVINO model snapshot under `.models/asr/`
- ONNX Silero VAD file at `.models/vad/silero_vad.onnx`

## Packaging

Bundle the whole app (window + engine) into a single windowed executable with
PyInstaller:

```powershell
pyinstaller --noconsole --name Betaal --icon assets\betaal.ico `
  --add-data "assets;assets" main.py
```

The result is `dist/Betaal/Betaal.exe`. Ship the `.models/` folder alongside it
(or let it download on first run). Optionally wrap it with Inno Setup and add a
`Shell:Startup` shortcut so Betaal launches at login.

## License

Internal / TBD.
