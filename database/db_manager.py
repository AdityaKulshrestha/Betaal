"""SQLite logging layer for Betaal usage analytics.

Tracks words typed, session durations, and estimated time saved. The database
file is created automatically on first use and lives alongside the application.
"""

import os
import sqlite3
import threading
from datetime import datetime

# Resolve the database path relative to the project root so the app works
# regardless of the current working directory it is launched from.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE_DIR, "app_metrics.db")

# A single connection is shared across threads (the hotkey listener writes,
# the GUI reads). check_same_thread=False is required; a lock serializes access.
_lock = threading.Lock()


def _connect():
    """Open the SQLite connection, allowing cross-thread use."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """Create the analytics table if it does not already exist."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                words_entered       INTEGER NOT NULL,
                duration_seconds    REAL    NOT NULL,
                time_saved_seconds  REAL    NOT NULL
            )
            """
        )
        conn.commit()


def _calc_time_saved(words_entered, duration_seconds):
    """Estimate time saved vs. manual typing at ~40 WPM.

    Time Saved = (Words Entered / 40) - (Duration Seconds / 60)  [in minutes]
    Returned in seconds for storage.
    """
    minutes_saved = (words_entered / 40.0) - (duration_seconds / 60.0)
    return minutes_saved * 60.0


def log_entry(words_entered, duration_seconds):
    """Insert a new analytics record and return the time saved (seconds)."""
    time_saved = _calc_time_saved(words_entered, duration_seconds)
    try:
        with _lock, _connect() as conn:
            conn.execute(
                """
                INSERT INTO analytics
                    (timestamp, words_entered, duration_seconds, time_saved_seconds)
                VALUES (?, ?, ?, ?)
                """,
                (datetime.now().isoformat(timespec="seconds"),
                 int(words_entered), float(duration_seconds), float(time_saved)),
            )
            conn.commit()
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to log entry: {exc}")
    return time_saved


def get_stats():
    """Return aggregated KPIs: total words, sessions, minutes used, minutes saved."""
    defaults = {
        "total_words": 0,
        "total_sessions": 0,
        "total_minutes_used": 0.0,
        "total_minutes_saved": 0.0,
    }
    try:
        with _lock, _connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(words_entered), 0),
                       COUNT(*),
                       COALESCE(SUM(duration_seconds), 0),
                       COALESCE(SUM(time_saved_seconds), 0)
                FROM analytics
                """
            ).fetchone()
        return {
            "total_words": int(row[0]),
            "total_sessions": int(row[1]),
            "total_minutes_used": round(row[2] / 60.0, 1),
            "total_minutes_saved": round(row[3] / 60.0, 1),
        }
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to read stats: {exc}")
        return defaults
