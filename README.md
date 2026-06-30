# Betaal

Betaal is an offline, Windows floating-widget / background dictation app that produces
**formatted** real-time transcriptions (in the spirit of Wispr Flow).

- Floating always-on-top overlay + system tray + settings GUI (Tauri / React).
- Global hotkey **`Ctrl+Shift+R`** to start/stop recording.
- Microphone capture at 16 kHz, mono, `float32` by default (configurable encoding).
- Voice-activity chunking with **Silero VAD**.
- **Model-agnostic** speech-to-text. First backend: Cohere Transcribe (OpenVINO IR),
  running fully in-process via OpenVINO (no HTTP).
- **Model-agnostic** text formatting via an OpenVINO GenAI LLM.
- Auto-types the formatted text into the focused application (email/doc) and shows it in the overlay.
- Two modes: **Dictation** (live) and **Meeting** (coming soon).
- Fully offline after a one-time model download on first run.

## Architecture

```
┌──────────────────────────────┐        JSON-RPC over stdio        ┌───────────────────────────────┐
│  Tauri frontend (Rust + TS)  │  <───────────────────────────────> │  Python backend (sidecar)     │
│  overlay · tray · settings   │     (no HTTP, no sockets)          │  audio · VAD · ASR · LLM · IPC │
│  global hotkey Ctrl+Shift+R  │                                    │  keystroke injection          │
└──────────────────────────────┘                                    └───────────────────────────────┘
```

Pipeline (Dictation mode):

```
mic ─▶ AudioCapture ─▶ VADSegmenter (Silero) ─▶ [queue] ─▶ ASRBackend ─▶ [queue] ─▶ LLMFormatter ─▶ inject + overlay
       16k mono f32        speech segments        bounded       text          bounded     formatted text
```

## Repository layout

```
Betaal/
  VERSION
  backend/            Python sidecar (PyInstaller target)
    betaal/
      asr/            model-agnostic ASR (ASRBackend ABC + CohereOVASR)
      llm/            model-agnostic formatting (LLMFormatter ABC + OVGenAIFormatter)
      audio/          capture + encoding
      vad/            Silero VAD segmenter
      pipeline/       queues + orchestration engine
      models/         first-run model download/verify + manifest
      inject/         Windows keystroke / clipboard injection
      ipc/            JSON-RPC over stdio
      modes/          dictation (live) + meeting (coming soon)
  frontend/           Tauri app (overlay, settings, tray, hotkey, sidecar)
  installer/          PyInstaller spec + Inno Setup script
```

## Development

### Backend

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -e .[dev]
python -m betaal --help
```

Run the backend as a stdio JSON-RPC server (this is how the frontend launches it):

```bash
python -m betaal serve
```

### Frontend

```bash
cd frontend
npm install
npm run tauri dev
```

## Packaging

See [installer/README.md](installer/README.md). In short:

1. `pyinstaller installer/betaal-backend.spec` → one-folder Python sidecar.
2. `npm run tauri build` → Tauri app bundling the sidecar.
3. Compile `installer/betaal.iss` with Inno Setup → `Betaal-Setup-x.y.z.exe`.

---

## Lightweight Python prototype (buffer mode)

A standalone, dependency-light implementation lives in this folder for early
testing. It runs a global hotkey, simulates the ASR pipeline, and types a
hardcoded buffer into the focused window. Real audio/VAD/ASR are mocked behind
`core/processor.py` and wired in later.

### Layout

```
Betaal/
  config.json               Persisted user configuration
  app_metrics.db            SQLite usage statistics (auto-created)
  main.py                   Entry point: config + listener + GUI
  core/
    hotkey_listener.py      Background global hotkey listener
    processor.py            Simulated ASR buffer pipeline
  database/
    db_manager.py           SQLite logging (words, duration, time saved)
  gui/
    settings_window.py      CustomTkinter dark settings + analytics
```

### Quick start (Windows, copy-paste)

Open **Command Prompt** in this folder and run, one block at a time:

```bat
uv venv
.venv\Scripts\activate
uv sync
uv run betaal
```

That's it — the settings window opens. Press `ctrl+shift+space` once to start
continuous dictation, and press it again to stop. Recognized text is pushed to
the key-injection queue and typed at your cursor (Notepad, Chrome, Word, etc.).

Next time, just run:

```bat
.venv\Scripts\activate
uv run betaal
```

### Local setup

```bash
uv venv
.venv\Scripts\activate            # Windows
uv sync
uv run betaal
```

If you prefer an explicit editable install flow instead of `uv sync`:

```bash
uv pip install -e .
uv run betaal
```

Press the hotkey (default `ctrl+shift+space`) once to start continuous
dictation and again to stop. Settings save to `config.json`; analytics persist
in `app_metrics.db`.

ASR model selection now includes `Cohere-transcribe` in the GUI. Internally it
maps to the registry ID `Aditya02/cohere-transcribe-03-2026-ov-fp16`.
On first run, Betaal auto-downloads missing model assets into `models/`:

- ASR OpenVINO model snapshot under `models/asr/`
- ONNX Silero VAD file at `models/vad/silero_vad.onnx`

> Note: the `keyboard` library requires Administrator rights for global hooks in
> some apps. Run the terminal/exe as Administrator if the hotkey is blocked.

### Build a single .exe (no console)

```bash
pyinstaller --noconsole --onefile --name="Betaal" main.py
```

The result is `dist/Betaal.exe`.

### Inno Setup + boot with Windows

Wrap `Betaal.exe` in an Inno Setup wizard and add a `Shell:Startup` shortcut so
it launches at login:

```iss
[Setup]
AppName=Betaal
AppVersion=1.0
DefaultDirName={autopf}\Betaal
OutputBaseFilename=Betaal-Setup

[Files]
Source: "dist\Betaal.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Startup shortcut so Betaal boots with Windows
Name: "{userstartup}\Betaal"; Filename: "{app}\Betaal.exe"

[Registry]
; Alternative: register under the Run key
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
  ValueType: string; ValueName: "Betaal"; ValueData: "{app}\Betaal.exe"; \
  Flags: uninsdeletevalue
```

Compile the `.iss` with Inno Setup to produce the installer.

## License

Internal / TBD.
