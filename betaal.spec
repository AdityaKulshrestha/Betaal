# PyInstaller spec for Betaal (native PySide6 app + dictation engine).
#
# Build (from the Betaal/ folder, inside the project venv):
#     pyinstaller betaal.spec --noconfirm
#
# Produces a one-directory bundle at dist/Betaal/ with Betaal.exe.
# Models are NOT bundled: they download + compile into ~/.cache/betaal on first
# run (or via "Betaal.exe setup"), which the installer triggers post-install.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Heavy native packages whose data files, DLLs and hidden imports PyInstaller's
# static analysis alone does not fully capture.
_datas = []
_binaries = []
_hiddenimports = []

for _pkg in (
    "openvino",
    "openvino_genai",
    "onnxruntime",
    "librosa",
    "soundfile",
    "sounddevice",
    "tokenizers",
    "numba",
    "llvmlite",
):
    try:
        d, b, h = collect_all(_pkg)
        _datas += d
        _binaries += b
        _hiddenimports += h
    except Exception:
        pass

# huggingface_hub powers snapshot_download() for first-run model fetching.
for _pkg in ("huggingface_hub",):
    try:
        d, b, h = collect_all(
            _pkg, include_py_files=False, exclude_datas=["**/*.pt", "**/*.h5"]
        )
        _datas += d
        _binaries += b
        _hiddenimports += h
    except Exception:
        pass

_hiddenimports += [
    "core.paths",
    "core.logging_setup",
    "core.processor",
    "core.reformatter",
    "core.hotkey_listener",
    "core.model_manager",
    "core.model_registry",
    "core.config_store",
    "core.active_window",
    "core.speaker",
    "database.db_manager",
    "desktop.main_window",
    "desktop.theme",
    "desktop.widgets",
    "keyboard",
    "pyperclip",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=_binaries,
    datas=_datas + [("assets", "assets")],
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "transformers", "tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Betaal",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app (no console)
    disable_windowed_traceback=False,
    icon="assets/betaal.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Betaal",
)
