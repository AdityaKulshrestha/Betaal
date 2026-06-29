"""Betaal settings & analytics window built with CustomTkinter."""

import json
import os

import customtkinter as ctk

from database import db_manager

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_BASE_DIR, "config.json")

MODEL_OPTIONS = ["Cloud API", "Local Base", "Local Large"]


class SettingsWindow(ctk.CTk):
    """Dark-themed dashboard with Analytics and Settings tabs."""

    def __init__(self, config):
        super().__init__()
        self._config = config

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Betaal")
        self.geometry("520x420")
        self.minsize(480, 380)

        ctk.CTkLabel(
            self, text="Betaal", font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(16, 4))

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=12)
        self.tabs.add("Analytics")
        self.tabs.add("Settings")

        self._build_analytics(self.tabs.tab("Analytics"))
        self._build_settings(self.tabs.tab("Settings"))

    def _build_analytics(self, parent):
        stats = db_manager.get_stats()
        kpis = [
            ("Total Words Typed", stats["total_words"]),
            ("Sessions Logged", stats["total_sessions"]),
            ("Total Minutes Saved", stats["total_minutes_saved"]),
        ]
        for title, value in kpis:
            card = ctk.CTkFrame(parent)
            card.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(
                card, text=str(value), font=ctk.CTkFont(size=28, weight="bold")
            ).pack(pady=(10, 0))
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13)).pack(
                pady=(0, 10)
            )

    def _build_settings(self, parent):
        ctk.CTkLabel(parent, text="Hotkey Combination").pack(
            anchor="w", padx=12, pady=(12, 0)
        )
        self.hotkey_entry = ctk.CTkEntry(parent)
        self.hotkey_entry.insert(0, self._config.get("hotkey", "ctrl+shift+space"))
        self.hotkey_entry.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(parent, text="VAD Sensitivity Threshold").pack(
            anchor="w", padx=12
        )
        self.vad_slider = ctk.CTkSlider(parent, from_=0.0, to=1.0)
        self.vad_slider.set(float(self._config.get("vad_threshold", 0.5)))
        self.vad_slider.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(parent, text="ASR Model").pack(anchor="w", padx=12)
        self.model_dropdown = ctk.CTkOptionMenu(parent, values=MODEL_OPTIONS)
        self.model_dropdown.set(self._config.get("asr_model", "Local Base"))
        self.model_dropdown.pack(fill="x", padx=12, pady=(0, 12))

        self.status = ctk.CTkLabel(parent, text="")
        self.status.pack(pady=(4, 0))

        ctk.CTkButton(parent, text="Save Settings", command=self._save).pack(
            pady=8
        )

    def _save(self):
        self._config["hotkey"] = self.hotkey_entry.get().strip()
        self._config["vad_threshold"] = round(self.vad_slider.get(), 2)
        self._config["asr_model"] = self.model_dropdown.get()
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._config, fh, indent=2)
            self.status.configure(text="Saved. Restart to apply hotkey changes.")
        except OSError as exc:
            self.status.configure(text=f"Save failed: {exc}")
