# Betaal

Betaal is an offline, Windows background dictation app that produces real-time
transcriptions. It runs as a native PySide6 desktop
app that lives in the system tray and types recognized speech into whatever app
currently has focus.

> Native, not browser-based — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
> for the internal design.

## Features

- **Standalone native Windows app** — PySide6 (Qt) desktop window, dark theme.
  No browser, no WebView, no web toolchain.
- **System tray + collapsible settings sidebar**; runs in the background.
- Global hotkey **`Ctrl+Shift+Space`** to start/stop continuous dictation.
- Microphone capture at 16 kHz, mono, `float32`.
- Voice-activity chunking with **Silero VAD** (ONNX).
- **Model-agnostic** speech-to-text running fully in-process via **OpenVINO**
  (no HTTP). See [Supported models](#supported-models).
- Injects recognized text into **whatever app currently has focus** and logs it.
- Optional **clipboard reformatter** — a small local LLM cleans up selected text
  on a second hotkey (`Ctrl+Alt+R`).
- Usage analytics (words dictated, time saved, avg daily words) in SQLite.
- **Fully offline** after a one-time model download on first run.
- **Reset all data** button in the sidebar to wipe usage metrics and the
  activity log.

## How to use

1. Install Betaal (see [Installation](#installation)) and let the first-run
   setup download + optimize the models — see [Supported models](#supported-models)
   below to choose which ones.
2. Betaal starts minimized to the **system tray**. Put focus in any app (Gmail,
   Chrome, Word, a text box).
3. Press the global hotkey (default `Ctrl+Shift+Space`) once to **start**
   continuous dictation and again to **stop**. Recognized text is typed into
   that focused app at your cursor.
4. Select text anywhere and press `Ctrl+Alt+R` to **reformat** it with the local
   LLM. Open the Betaal window to see analytics, the live activity log, and the
   settings sidebar (pick models, tune VAD sensitivity, enroll a speaker).

Settings persist to `config.json`; analytics persist in `app_metrics.db` — both
under `%USERPROFILE%\.cache\betaal`.

> The `keyboard` library may need Administrator rights for global hooks in some
> apps. Run Betaal as Administrator if the hotkey is blocked in an elevated app.

## Demo

<!-- Replace the link below with the actual demo video URL. -->
[![Betaal demo](assets/betaal.png)](https://example.com/betaal-demo)

## Supported models

All models are pre-optimized **OpenVINO IR** and download on first run (or via
`Betaal.exe setup`) into `%USERPROFILE%\.cache\betaal`.

### Speech-to-text (ASR)

| Display name        | Registry ID                                    | Precision |
| ------------------- | ---------------------------------------------- | --------- |
| `Cohere-transcribe` | `Aditya02/cohere-transcribe-03-2026-ov-fp16`   | FP16      |
| `Whisper Large`     | `OpenVINO/whisper-large-v3-int4-ov`            | INT4      |

### Clipboard reformatter (LLM)

| Display name             | Registry ID                                 | Precision |
| ------------------------ | ------------------------------------------- | --------- |
| `LFM2.5 350M`            | `OpenVINO/LFM2.5-350M-int8-ov`              | INT8      |
| `Qwen2.5-1.5B Instruct`  | `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`    | INT4      |
| `TinyLlama 1.1B Chat`    | `OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov` | INT4      |
| `Phi-3 Mini Instruct`    | `OpenVINO/Phi-3-mini-4k-instruct-int4-ov`   | INT4      |

### Voice activity detection

- **Silero VAD** (ONNX) — auto-downloaded to `.cache/betaal/vad/silero_vad.onnx`.

## Installation

### Option 1 — Download the installer (recommended)

1. Download [Betaal-Setup.exe](https://github.com/AdityaKulshrestha/Betaal/releases/download/v0.1/Betaal-Setup-0.1.0.exe) from the releases page.
2. Run it. The installer is **per-user (no admin)**, adds a Start Menu entry,
   optionally a login **Startup** shortcut (background tray), and — when the
   *first-run setup* task is selected — downloads and optimizes the models.
3. Launch Betaal from the Start Menu (or it auto-starts at login if enabled).

### Option 2 — Build from source

**Prerequisites**

- **Python 3.10+**
- [`uv`](https://docs.astral.sh/uv/) (package / venv manager)
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) — only needed to build the
  installer (2b)

**Clean setup**

```powershell
git clone <repo-url>
cd Betaal
uv venv
uv sync
```

**Run directly from source** (no packaging):

```powershell
uv run betaal                   # native desktop app (default)
uv run python main.py headless  # engine only, no window
uv run python main.py setup     # download + optimize models, then exit
```

**2a. Build the standalone app bundle**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Produces `dist\Betaal\Betaal.exe` — a portable one-directory bundle you can zip
and copy to another machine.

**2b. Build the installer**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Installer
```

Produces `packaging\installer\Output\Betaal-Setup-<version>.exe`. The script
auto-detects `ISCC.exe` from a machine-wide or per-user Inno Setup install.

> Models are **not** bundled. They download and are optimized (device-specific
> OpenVINO compile) on the target machine into `%USERPROFILE%\.cache\betaal`.

## Supported hardware (Windows on Intel AI PC)

Betaal runs entirely on **OpenVINO**, so it targets Intel AI PCs end to end:

- **OS:** Windows 10 / 11, x86-64.
- **CPU:** any modern Intel Core (used by default for the reformatter LLM, and as
  the automatic fallback for ASR).
- **GPU:** Intel Arc / Iris Xe integrated or discrete GPU (default device for
  ASR; falls back to CPU when no GPU is present).
- **NPU:** Intel Core Ultra (Meteor Lake / Lunar Lake / Arrow Lake) AI PCs —
  selectable as an OpenVINO device where supported.

Device selection is configurable per model (`asr_device`, `llm_device`) with
`AUTO` and CPU fallback, so Betaal works across the full Intel AI PC lineup.

## Logs

Betaal writes a rotating debug log to
`%USERPROFILE%\.cache\betaal\logs\betaal.log` (kept for both source and
installed runs). It captures startup, model load/compile, and any errors —
check it first if the engine fails to start or the UI behaves unexpectedly.

## License
MIT
