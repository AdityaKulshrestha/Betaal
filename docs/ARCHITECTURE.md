# Betaal — Architecture

Betaal is a **native, single-process** Windows app: the PySide6 (Qt) window and
the background dictation engine share memory directly — no browser, WebView,
Node, Rust, IPC, or sidecar.

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
hotkey engine share memory directly. Engine callbacks (which fire on background
threads) are marshalled onto the UI thread with Qt signals.

## How it works

- Betaal runs **in the background** (system tray). You work in any app —
  Gmail, Chrome, Word, a text field — and press the **global hotkey**.
- Recognized text is injected **into whatever app currently has focus**
  (via clipboard paste, restoring your previous clipboard afterward). It is
  **never** shown or pasted onto Betaal's own window.
- The desktop window is a **logger**: it displays what happened — usage metrics
  and a live activity log of every dictation — plus the settings sidebar. It
  is read-only and does not capture your dictation focus.

## Pipeline (dictation)

```
mic ─▶ AudioCapture ─▶ Silero VAD ─▶ [queue] ─▶ ASR (OpenVINO) ─▶ inject into focused app + log entry
       16k mono f32     speech segs    bounded     text
```

## Repository layout

```
Betaal/
  main.py                   Entry point: native app (default) · headless · setup
  config.json               Persisted user configuration (source runs)
  pyproject.toml            Python package + dependencies
  betaal.spec               PyInstaller bundle definition
  scripts/build.ps1         Build the app bundle + optional installer
  installer/betaal.iss      Inno Setup installer script
  assets/                   App icon (betaal.png / betaal.ico)
  desktop/                  Native PySide6 UI (the app)
    main_window.py          Window: sidebar · dashboard · activity log · tray
    widgets.py              Metric cards, collapsible sections, note cards
    theme.py                Dark Qt stylesheet + palette
  core/
    paths.py                Centralized data/asset path resolution
    config_store.py         Shared config load/save
    hotkey_listener.py      Background global hotkey + dictation loop
    processor.py            AudioCapture · Silero VAD · OpenVINO ASR pipeline
    reformatter.py          OpenVINO GenAI LLM clipboard reformatter
    model_manager.py        First-run model download/verify
    model_registry.py       Display-name → registry-ID mapping
    speaker.py              Speaker-enrollment WAV recorder
  database/
    db_manager.py           SQLite logging (words, duration, time saved, notes)
```

## Runtime data location

All writable runtime data lives under `%USERPROFILE%\.cache\betaal` (never in a
synced folder): downloaded models, the OpenVINO compile cache, `config.json`,
`app_metrics.db`, and speaker samples. When running from a source checkout,
pre-existing in-repo `.models/`, `config.json`, and `app_metrics.db` are reused
if present.
