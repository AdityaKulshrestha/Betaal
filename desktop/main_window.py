"""Betaal native desktop window (PySide6).

A standalone Windows application (no browser / WebView) that runs the dictation
engine in-process and acts as a live activity logger. Text is injected into
whichever external app has focus; this window only shows what happened.
"""

from __future__ import annotations

import os
import sys
import threading
import logging
from datetime import datetime

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from core.config_store import load_config, save_config
from core.hotkey_listener import HotkeyListener
from core.model_registry import (
    DEFAULT_LLM_DISPLAY,
    DEFAULT_MODEL_DISPLAY,
    list_llm_model_names,
    list_model_names,
)
from core.processor import available_devices
from core import paths
from core import speaker
from database import db_manager
from desktop import theme
from desktop.widgets import (
    CollapsibleSection,
    MetricCard,
    NoteCard,
    SpeakerRow,
    kbd,
)

_ICON_PNG = str(paths.asset_path("betaal.png"))
_ICON_ICO = str(paths.asset_path("betaal.ico"))

_PIPELINE_KEYS = {"hotkey", "vad_threshold", "asr_model", "asr_device", "log_transcript", "min_silence_ms", "max_segment_seconds", "reformat_hotkey", "llm_model", "llm_device"}

_log = logging.getLogger("betaal.gui")


def _initials(name: str) -> str:
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "B"
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()


def _greeting() -> str:
    h = datetime.now().hour
    if h < 12:
        return "Good morning"
    if h < 18:
        return "Good afternoon"
    return "Good evening"


def _fmt_saved(minutes: float) -> tuple[str, str]:
    if minutes >= 60:
        return f"{minutes / 60:.1f}", "hrs"
    return f"{round(minutes)}", "min"


def _time_ago(iso: str) -> str:
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    diff = (datetime.now() - then).total_seconds()
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff // 60)}m ago"
    if diff < 86400:
        return f"{int(diff // 3600)}h ago"
    return then.strftime("%b %d")


class EngineBridge(QObject):
    """Marshals engine callbacks (background threads) onto the Qt main thread."""

    note = Signal(str, int, float)
    state = Signal(bool)
    ready = Signal(bool)
    busy = Signal(str)
    error = Signal(str)
    speakers_changed = Signal()
    llm_state = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._bridge = EngineBridge()
        self._listener: HotkeyListener | None = None
        self._listener_lock = threading.Lock()
        self._collapsed = False
        self._note_cards: list[NoteCard] = []

        self.setObjectName("Root")
        self.setWindowTitle("Betaal")
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)
        if os.path.isfile(_ICON_PNG):
            self.setWindowIcon(QIcon(_ICON_PNG))

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = self._build_sidebar()
        root.addWidget(self._sidebar)
        root.addWidget(self._build_content(), 1)

        self._bridge.note.connect(self._on_note)
        self._bridge.state.connect(self._on_state)
        self._bridge.ready.connect(self._on_ready)
        self._bridge.busy.connect(self._on_busy)
        self._bridge.error.connect(self._on_error)
        self._bridge.speakers_changed.connect(self._reload_speakers)
        self._bridge.llm_state.connect(self._on_llm_state)

        self._build_tray()
        self._refresh_stats()
        self._load_notes()
        self._reload_speakers()
        self._start_engine()

    # ---- sidebar ----------------------------------------------------------

    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(280)
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(12, 16, 12, 12)
        outer.setSpacing(10)

        # brand row
        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = QLabel("B")
        mark.setObjectName("BrandMark")
        mark.setFixedSize(34, 34)
        mark.setAlignment(Qt.AlignCenter)
        brand.addWidget(mark)
        name_col = QVBoxLayout()
        name_col.setSpacing(0)
        self._brand_name = QLabel("Betaal")
        self._brand_name.setObjectName("BrandName")
        self._brand_sub = QLabel("Offline dictation")
        self._brand_sub.setObjectName("BrandSub")
        name_col.addWidget(self._brand_name)
        name_col.addWidget(self._brand_sub)
        brand.addLayout(name_col)
        brand.addStretch(1)
        self._collapse_btn = QPushButton("\u276e")  # ❮
        self._collapse_btn.setObjectName("CollapseBtn")
        self._collapse_btn.setFixedSize(30, 30)
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.clicked.connect(self._toggle_sidebar)
        brand.addWidget(self._collapse_btn)
        outer.addLayout(brand)

        self._nav_label = QLabel("CONFIGURATION")
        self._nav_label.setObjectName("NavLabel")
        outer.addWidget(self._nav_label)

        # sections
        self._sidebar_sections = []
        section_specs = [
            ("\u25a0", "ASR Model", True, self._build_model_section),
            ("\u2261", "VAD Settings", False, self._build_vad_section),
            ("\u25c9", "Speaker ID", False, self._build_speaker_section),
            ("\u270e", "Reformatter (LLM)", False, self._build_reformat_section),
        ]
        for glyph, title, expanded, builder in section_specs:
            section = CollapsibleSection(glyph, title, expanded=expanded)
            try:
                builder(section)
            except Exception:
                # A broken section (e.g. device enumeration failing on some
                # machines) must not take the rest of the sidebar down with it.
                _log.exception("Sidebar section %r failed to build", title)
                err = QLabel("Unavailable \u2014 see logs")
                err.setProperty("class", "FieldLabel")
                section.add_widget(err)
            outer.addWidget(section)
            self._sidebar_sections.append(section)

        outer.addStretch(1)

        # footer: hotkey
        self._foot = QWidget()
        foot_col = QVBoxLayout(self._foot)
        foot_col.setContentsMargins(2, 8, 2, 2)
        foot_col.setSpacing(6)
        hk_label = QLabel("GLOBAL HOTKEY")
        hk_label.setObjectName("NavLabel")
        foot_col.addWidget(hk_label)
        self._hotkey_edit = QLineEdit(self._config.get("hotkey", "ctrl+shift+space"))
        self._hotkey_edit.setPlaceholderText("ctrl+shift+space")
        self._hotkey_edit.returnPressed.connect(self._apply_hotkey)
        foot_col.addWidget(self._hotkey_edit)
        apply_btn = QPushButton("Apply hotkey")
        apply_btn.setProperty("class", "Ghost")
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._apply_hotkey)
        foot_col.addWidget(apply_btn)
        outer.addWidget(self._foot)

        return bar

    def _build_model_section(self, section: CollapsibleSection) -> None:
        lbl = QLabel("Active model")
        lbl.setProperty("class", "FieldLabel")
        section.add_widget(lbl)

        self._model_combo = QComboBox()
        models = list_model_names()
        current = self._config.get("asr_model", DEFAULT_MODEL_DISPLAY)
        if current not in models:
            models = [current, *models]
        self._model_combo.addItems(models)
        self._model_combo.setCurrentText(current)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        section.add_widget(self._model_combo)

        dev_lbl = QLabel("Compute device")
        dev_lbl.setProperty("class", "FieldLabel")
        section.add_widget(dev_lbl)

        self._device_combo = QComboBox()
        devices = available_devices()
        current_dev = self._config.get("asr_device", "GPU")
        if current_dev not in devices:
            devices = [current_dev, *devices]
        self._device_combo.addItems(devices)
        self._device_combo.setCurrentText(current_dev)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        section.add_widget(self._device_combo)

        self._log_check = QCheckBox("Log transcript to console")
        self._log_check.setChecked(bool(self._config.get("log_transcript", False)))
        self._log_check.toggled.connect(self._on_log_toggled)
        section.add_widget(self._log_check)

        hint = QLabel("Runs fully offline via OpenVINO. Switching reloads the engine.")
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        section.add_widget(hint)

    def _build_vad_section(self, section: CollapsibleSection) -> None:
        row = QHBoxLayout()
        lbl = QLabel("Sensitivity threshold")
        lbl.setProperty("class", "FieldLabel")
        self._vad_value = QLabel(f"{float(self._config.get('vad_threshold', 0.5)):.2f}")
        self._vad_value.setProperty("class", "FieldValue")
        self._vad_value.setAlignment(Qt.AlignRight)
        row.addWidget(lbl)
        row.addWidget(self._vad_value)
        wrap = QWidget()
        wrap.setLayout(row)
        section.add_widget(wrap)

        self._vad_slider = QSlider(Qt.Horizontal)
        self._vad_slider.setMinimum(5)
        self._vad_slider.setMaximum(95)
        self._vad_slider.setValue(int(float(self._config.get("vad_threshold", 0.5)) * 100))
        self._vad_slider.valueChanged.connect(self._on_vad_preview)
        self._vad_slider.sliderReleased.connect(self._on_vad_commit)
        section.add_widget(self._vad_slider)

        hint = QLabel("Higher = clearer speech required before a segment is captured.")
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        section.add_widget(hint)

    def _build_speaker_section(self, section: CollapsibleSection) -> None:
        self._speaker_name = QLineEdit()
        self._speaker_name.setPlaceholderText("Speaker name")
        section.add_widget(self._speaker_name)

        self._speaker_btn = QPushButton("\u25cf  Enroll voice sample")
        self._speaker_btn.setProperty("class", "Primary")
        self._speaker_btn.setCursor(Qt.PointingHandCursor)
        self._speaker_btn.clicked.connect(self._record_speaker)
        section.add_widget(self._speaker_btn)

        self._speaker_list = QVBoxLayout()
        self._speaker_list.setSpacing(6)
        holder = QWidget()
        holder.setLayout(self._speaker_list)
        section.add_widget(holder)

        hint = QLabel("Record a short sample per person for speaker identification.")
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        section.add_widget(hint)

    def _build_reformat_section(self, section: CollapsibleSection) -> None:
        # LLM status
        status_row = QHBoxLayout()
        status_lbl = QLabel("Model status")
        status_lbl.setProperty("class", "FieldLabel")
        self._llm_status = QLabel("Not loaded")
        self._llm_status.setProperty("class", "FieldValue")
        self._llm_status.setAlignment(Qt.AlignRight)
        status_row.addWidget(status_lbl)
        status_row.addWidget(self._llm_status)
        status_wrap = QWidget()
        status_wrap.setLayout(status_row)
        section.add_widget(status_wrap)

        # LLM model
        model_lbl = QLabel("LLM model")
        model_lbl.setProperty("class", "FieldLabel")
        section.add_widget(model_lbl)

        self._llm_combo = QComboBox()
        llm_models = list_llm_model_names()
        current_llm = self._config.get("llm_model", DEFAULT_LLM_DISPLAY)
        if current_llm not in llm_models:
            llm_models = [current_llm, *llm_models]
        self._llm_combo.addItems(llm_models)
        self._llm_combo.setCurrentText(current_llm)
        self._llm_combo.currentTextChanged.connect(self._on_llm_model_changed)
        section.add_widget(self._llm_combo)

        # LLM device
        dev_lbl = QLabel("Compute device")
        dev_lbl.setProperty("class", "FieldLabel")
        section.add_widget(dev_lbl)

        self._llm_device_combo = QComboBox()
        devices = available_devices()
        current_dev = self._config.get("llm_device", "CPU")
        if current_dev not in devices:
            devices = [current_dev, *devices]
        self._llm_device_combo.addItems(devices)
        self._llm_device_combo.setCurrentText(current_dev)
        self._llm_device_combo.currentTextChanged.connect(self._on_llm_device_changed)
        section.add_widget(self._llm_device_combo)

        # Reformat hotkey
        hk_lbl = QLabel("Reformat hotkey")
        hk_lbl.setProperty("class", "FieldLabel")
        section.add_widget(hk_lbl)

        self._reformat_hotkey_edit = QLineEdit(
            self._config.get("reformat_hotkey", "ctrl+alt+r")
        )
        self._reformat_hotkey_edit.setPlaceholderText("ctrl+alt+r")
        self._reformat_hotkey_edit.returnPressed.connect(self._apply_reformat_hotkey)
        section.add_widget(self._reformat_hotkey_edit)

        apply_btn = QPushButton("Apply reformat hotkey")
        apply_btn.setProperty("class", "Ghost")
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._apply_reformat_hotkey)
        section.add_widget(apply_btn)

        hint = QLabel(
            "Press the reformat hotkey to rewrite all text in the focused app "
            "using the selected LLM."
        )
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        section.add_widget(hint)

    # ---- content ----------------------------------------------------------

    def _build_content(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()
        page.setObjectName("Root")
        col = QVBoxLayout(page)
        col.setContentsMargins(34, 28, 34, 34)
        col.setSpacing(22)

        col.addLayout(self._build_header())
        col.addWidget(self._build_metrics())
        col.addWidget(self._build_activity(), 1)

        scroll.setWidget(page)
        return scroll

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        avatar = QLabel(_initials(self._config.get("user_name", "there")))
        avatar.setObjectName("Avatar")
        avatar.setFixedSize(52, 52)
        avatar.setAlignment(Qt.AlignCenter)
        row.addWidget(avatar)

        greet_col = QVBoxLayout()
        greet_col.setSpacing(2)
        greeting = QLabel(f"{_greeting()}, {self._config.get('user_name', 'there')}")
        greeting.setObjectName("Greeting")
        date_line = QLabel(datetime.now().strftime("%A, %B %d, %Y"))
        date_line.setObjectName("DateLine")
        greet_col.addWidget(greeting)
        greet_col.addWidget(date_line)
        row.addLayout(greet_col)
        row.addStretch(1)

        # status chip
        chip = QFrame()
        chip.setObjectName("StatusChip")
        chip_l = QHBoxLayout(chip)
        chip_l.setContentsMargins(14, 8, 16, 8)
        chip_l.setSpacing(8)
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(9, 9)
        self._set_dot("#7a8199")
        self._status_text = QLabel("Starting engine\u2026")
        self._status_text.setObjectName("StatusText")
        chip_l.addWidget(self._status_dot)
        chip_l.addWidget(self._status_text)
        # Indeterminate "busy" bar shown while the engine loads / compiles.
        self._loading_bar = QProgressBar()
        self._loading_bar.setObjectName("LoadingBar")
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(96, 8)
        self._loading_bar.setStyleSheet(
            "QProgressBar{border:none;background:#2a2f45;border-radius:4px;}"
            f"QProgressBar::chunk{{background:{theme.COLORS['accent']};border-radius:4px;}}"
        )
        chip_l.addWidget(self._loading_bar)
        row.addWidget(chip)

        # hotkey hint
        hint = QFrame()
        hint.setObjectName("HotkeyHint")
        hint_l = QHBoxLayout(hint)
        hint_l.setContentsMargins(14, 8, 14, 8)
        hint_l.setSpacing(8)
        htext = QLabel("Dictate anywhere")
        htext.setObjectName("HintText")
        hint_l.addWidget(htext)
        for i, key in enumerate(self._hotkey_keys()):
            if i:
                plus = QLabel("+")
                plus.setProperty("class", "Hint")
                hint_l.addWidget(plus)
            hint_l.addWidget(kbd(key))
        self._hotkey_hint = hint
        row.addWidget(hint)

        quit_btn = QPushButton("\u23fb")  # power symbol
        quit_btn.setObjectName("QuitBtn")
        quit_btn.setFixedSize(38, 38)
        quit_btn.setCursor(Qt.PointingHandCursor)
        quit_btn.setToolTip("Quit Betaal")
        quit_btn.clicked.connect(self._quit)
        row.addWidget(quit_btn)
        return row

    def _build_metrics(self) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        title = QLabel("OVERVIEW")
        title.setProperty("class", "SectionTitle")
        col.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(14)
        self._card_words = MetricCard("\u25a4", "Words dictated", theme.COLORS["accent"])
        self._card_saved = MetricCard("\u25f7", "Time saved vs typing", theme.COLORS["success"])
        self._card_avg = MetricCard("\u2197", "Avg words / day", theme.COLORS["accent2"])
        self._card_sessions = MetricCard("\u25c8", "Sessions", theme.COLORS["amber"])
        for i, card in enumerate(
            (self._card_words, self._card_saved, self._card_avg, self._card_sessions)
        ):
            grid.addWidget(card, 0, i)
        col.addLayout(grid)
        return wrap

    def _build_activity(self) -> QWidget:
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        head = QHBoxLayout()
        title = QLabel("ACTIVITY LOG")
        title.setProperty("class", "SectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        self._count_pill = QLabel("0 entries")
        self._count_pill.setObjectName("CountPill")
        head.addWidget(self._count_pill)
        col.addLayout(head)

        self._notes_container = QVBoxLayout()
        self._notes_container.setSpacing(12)
        holder = QWidget()
        holder.setLayout(self._notes_container)
        col.addWidget(holder)

        self._empty = self._build_empty_state()
        self._notes_container.addWidget(self._empty)
        self._notes_container.addStretch(1)
        return wrap

    def _build_empty_state(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("EmptyState")
        v = QVBoxLayout(frame)
        v.setContentsMargins(20, 44, 20, 44)
        v.setSpacing(8)
        v.setAlignment(Qt.AlignCenter)
        title = QLabel("No activity yet")
        title.setObjectName("EmptyTitle")
        title.setAlignment(Qt.AlignCenter)
        body = QLabel(
            "Focus any app (Gmail, Chrome, Word\u2026), press your global hotkey, "
            "and dictated text is typed there \u2014 every entry is logged here."
        )
        body.setObjectName("EmptyBody")
        body.setAlignment(Qt.AlignCenter)
        body.setWordWrap(True)
        v.addWidget(title)
        v.addWidget(body)
        return frame

    # ---- tray + titlebar --------------------------------------------------

    def _build_tray(self) -> None:
        icon = QIcon(_ICON_PNG) if os.path.isfile(_ICON_PNG) else self.windowIcon()
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("Betaal \u2014 offline dictation")
        menu = QMenu()
        show_action = QAction("Show Betaal", self)
        show_action.triggered.connect(self._show_from_tray)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def apply_dark_titlebar(self) -> None:
        """Ask Windows DWM for a dark title bar (Windows 10 2004+ / 11)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            value = ctypes.c_int(1)
            for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
                )
        except Exception:
            pass

    # ---- engine wiring ----------------------------------------------------

    def _start_engine(self) -> None:
        threading.Thread(target=self._build_listener, daemon=True).start()

    def _build_listener(self) -> None:
        with self._listener_lock:
            model = self._config.get("asr_model", DEFAULT_MODEL_DISPLAY)
            device = self._config.get("asr_device", "GPU")
            _log.info("Building engine: model=%s device=%s", model, device)
            self._bridge.busy.emit(f"Loading {model} on {device}\u2026")
            try:
                import keyboard

                keyboard.clear_all_hotkeys()
            except Exception:
                pass
            try:
                listener = HotkeyListener(
                    hotkey=self._config["hotkey"],
                    model_name=self._config["asr_model"],
                    vad_threshold=self._config["vad_threshold"],
                    log_transcript=self._config["log_transcript"],
                    device=device,
                    min_silence_ms=self._config["min_silence_ms"],
                    max_segment_seconds=self._config["max_segment_seconds"],
                    reformat_hotkey=self._config["reformat_hotkey"],
                    llm_model=self._config["llm_model"],
                    llm_device=self._config["llm_device"],
                    on_note=lambda t, w, d: self._bridge.note.emit(t, w, d),
                    on_state=lambda r: self._bridge.state.emit(r),
                    on_llm_state=lambda s: self._bridge.llm_state.emit(s),
                )
                listener.start()
                self._listener = listener
                _log.info("Engine ready: model=%s device=%s", model, device)
                self._bridge.ready.emit(True)
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                _log.exception("Engine failed to start: model=%s device=%s", model, device)
                self._bridge.error.emit(str(exc))

    def _rebuild_engine(self) -> None:
        threading.Thread(target=self._build_listener, daemon=True).start()

    # ---- slots ------------------------------------------------------------

    @Slot(str, int, float)
    def _on_note(self, text: str, words: int, duration: float) -> None:
        # A whole dictation session was just written as one entry; reflect it.
        self._refresh_stats()
        self._load_notes(mark_fresh=True)

    @Slot(bool)
    def _on_state(self, recording: bool) -> None:
        if recording:
            self._set_dot(theme.COLORS["danger"])
            self._status_text.setText("Listening")
        else:
            self._set_dot(theme.COLORS["success"])
            self._status_text.setText("Idle \u00b7 running in background")

    @Slot(bool)
    def _on_ready(self, _ready: bool) -> None:
        self._set_dot(theme.COLORS["success"])
        self._status_text.setText("Idle \u00b7 running in background")
        self._loading_bar.setVisible(False)
        self._set_engine_controls_enabled(True)

    @Slot(str)
    def _on_busy(self, message: str) -> None:
        self._set_dot(theme.COLORS["muted"])
        self._status_text.setText(message)
        self._loading_bar.setVisible(True)
        self._set_engine_controls_enabled(False)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._set_dot(theme.COLORS["danger"])
        self._status_text.setText("Engine error")
        self._loading_bar.setVisible(False)
        self._set_engine_controls_enabled(True)
        _log.error("Engine error reported to UI: %s", message)

    def _set_engine_controls_enabled(self, enabled: bool) -> None:
        """Disable model/device switching while the engine is (re)loading."""
        for widget in (getattr(self, "_model_combo", None), getattr(self, "_device_combo", None)):
            if widget is not None:
                widget.setEnabled(enabled)

    @Slot(str)
    def _on_llm_state(self, status: str) -> None:
        labels = {
            "loading": "Loading\u2026",
            "ready": "Ready",
            "reformatting": "Reformatting\u2026",
            "error": "Error",
            "unloaded": "Not loaded",
        }
        colors = {
            "loading": theme.COLORS["muted"],
            "ready": theme.COLORS["success"],
            "reformatting": theme.COLORS["accent"],
            "error": theme.COLORS["danger"],
            "unloaded": "#7a8199",
        }
        if getattr(self, "_llm_status", None) is not None:
            self._llm_status.setText(labels.get(status, status))
            self._llm_status.setStyleSheet(f"color: {colors.get(status, '#7a8199')};")

    # ---- config handlers --------------------------------------------------

    def _on_model_changed(self, value: str) -> None:
        self._update_config({"asr_model": value})

    def _on_device_changed(self, value: str) -> None:
        self._update_config({"asr_device": value})

    def _on_log_toggled(self, checked: bool) -> None:
        self._update_config({"log_transcript": bool(checked)})

    def _on_llm_model_changed(self, value: str) -> None:
        self._update_config({"llm_model": value})

    def _on_llm_device_changed(self, value: str) -> None:
        self._update_config({"llm_device": value})

    def _apply_reformat_hotkey(self) -> None:
        text = self._reformat_hotkey_edit.text().strip() or "ctrl+alt+r"
        self._update_config({"reformat_hotkey": text})

    def _on_vad_preview(self, value: int) -> None:
        self._vad_value.setText(f"{value / 100:.2f}")

    def _on_vad_commit(self) -> None:
        self._update_config({"vad_threshold": round(self._vad_slider.value() / 100, 2)})

    def _apply_hotkey(self) -> None:
        text = self._hotkey_edit.text().strip() or "ctrl+shift+space"
        self._update_config({"hotkey": text})

    def _update_config(self, patch: dict) -> None:
        rebuild = any(k in _PIPELINE_KEYS for k in patch)
        self._config.update(patch)
        save_config(self._config)
        if rebuild:
            self._rebuild_engine()

    # ---- speakers ---------------------------------------------------------

    def _record_speaker(self) -> None:
        name = self._speaker_name.text().strip() or "Speaker"
        self._speaker_btn.setEnabled(False)
        self._speaker_btn.setText("\u25cf  Recording 5s\u2026")

        def worker() -> None:
            try:
                speaker.record_speaker(name, seconds=5.0)
            except Exception as exc:  # pragma: no cover - device dependent
                print(f"[Betaal][gui] speaker record failed: {exc}")
            self._bridge.speakers_changed.emit()

        threading.Thread(target=worker, daemon=True).start()

    @Slot()
    def _reload_speakers(self) -> None:
        self._speaker_btn.setEnabled(True)
        self._speaker_btn.setText("\u25cf  Enroll voice sample")
        self._speaker_name.clear()
        while self._speaker_list.count():
            item = self._speaker_list.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for sp in speaker.list_speakers():
            self._speaker_list.addWidget(SpeakerRow(sp["name"], sp["seconds"]))

    # ---- data -------------------------------------------------------------

    def _refresh_stats(self) -> None:
        stats = db_manager.get_stats()
        self._card_words.set_value(f"{stats['total_words']:,}")
        val, unit = _fmt_saved(stats["total_minutes_saved"])
        self._card_saved.set_value(val, unit)
        self._card_avg.set_value(f"{stats['avg_daily_words']:,}")
        self._card_sessions.set_value(str(stats["total_sessions"]))
        self._card_sessions.set_label(f"Sessions \u00b7 {stats['active_days']} active days")

    def _load_notes(self, mark_fresh: bool = False) -> None:
        notes = db_manager.get_notes(limit=200)
        for card in self._note_cards:
            card.setParent(None)
            card.deleteLater()
        self._note_cards.clear()
        self._empty.setVisible(not notes)
        newest_id = notes[0]["id"] if notes else None
        for note in reversed(notes):  # oldest first, so newest ends on top
            self._prepend_note(
                note_id=note["id"],
                text=note["text"],
                meta=(
                    f"{_time_ago(note['timestamp'])}  \u00b7  "
                    f"{note['word_count']} words  \u00b7  {note['duration_seconds']}s"
                ),
                fresh=(mark_fresh and note["id"] == newest_id),
            )
        self._count_pill.setText(f"{len(notes)} entries")

    def _prepend_note(self, note_id: int, text: str, meta: str, fresh: bool) -> None:
        self._empty.setVisible(False)
        card = NoteCard(note_id, text, meta, fresh=fresh)
        card.deleted.connect(self._delete_note)
        # index 0 is the empty state; insert notes right after it (top of the list)
        self._notes_container.insertWidget(1, card)
        self._note_cards.insert(0, card)

    @Slot(int)
    def _delete_note(self, note_id: int) -> None:
        if note_id >= 0:
            db_manager.delete_note(note_id)
        self._load_notes()

    # ---- sidebar collapse -------------------------------------------------

    def _toggle_sidebar(self) -> None:
        self._collapsed = not self._collapsed
        self._sidebar.setFixedWidth(64 if self._collapsed else 280)
        self._brand_name.setVisible(not self._collapsed)
        self._brand_sub.setVisible(not self._collapsed)
        self._nav_label.setVisible(not self._collapsed)
        self._foot.setVisible(not self._collapsed)
        self._collapse_btn.setText("\u276f" if self._collapsed else "\u276e")
        for section in self._sidebar_sections:
            section.set_mini(self._collapsed)

    # ---- helpers ----------------------------------------------------------

    def _hotkey_keys(self) -> list[str]:
        return [
            k.strip().capitalize()
            for k in self._config.get("hotkey", "ctrl+shift+space").split("+")
            if k.strip()
        ]

    def _set_dot(self, color: str) -> None:
        self._status_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")

    # ---- tray / lifecycle -------------------------------------------------

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit(self) -> None:
        self._tray.hide()
        QApplication.quit()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Keep running in the background instead of exiting.
        event.ignore()
        self.hide()
        self._tray.showMessage(
            "Betaal",
            "Still running in the background. Use your hotkey any time.",
            QSystemTrayIcon.Information,
            2500,
        )


def run() -> None:
    """Launch the native desktop application."""
    config = load_config()
    db_manager.init_db()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Betaal")
    if os.path.isfile(_ICON_PNG):
        app.setWindowIcon(QIcon(_ICON_PNG))
    app.setStyleSheet(theme.stylesheet())
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow(config)
    window.show()
    window.apply_dark_titlebar()
    sys.exit(app.exec())
