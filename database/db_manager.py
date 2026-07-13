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
    """Create the analytics and notes tables if they do not already exist."""
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                text              TEXT    NOT NULL,
                word_count        INTEGER NOT NULL,
                duration_seconds  REAL    NOT NULL
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


def log_entry(words_entered, duration_seconds, text=None):
    """Insert a new analytics record (and optional note) and return time saved."""
    time_saved = _calc_time_saved(words_entered, duration_seconds)
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        with _lock, _connect() as conn:
            conn.execute(
                """
                INSERT INTO analytics
                    (timestamp, words_entered, duration_seconds, time_saved_seconds)
                VALUES (?, ?, ?, ?)
                """,
                (stamp, int(words_entered), float(duration_seconds),
                 float(time_saved)),
            )
            if text and str(text).strip():
                conn.execute(
                    """
                    INSERT INTO notes
                        (timestamp, text, word_count, duration_seconds)
                    VALUES (?, ?, ?, ?)
                    """,
                    (stamp, str(text).strip(), int(words_entered),
                     float(duration_seconds)),
                )
            conn.commit()
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to log entry: {exc}")
    return time_saved


def get_notes(limit=200):
    """Return recent notes (newest first) as a list of dicts."""
    try:
        with _lock, _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, text, word_count, duration_seconds
                FROM notes
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "timestamp": row[1],
                "text": row[2],
                "word_count": int(row[3]),
                "duration_seconds": round(float(row[4]), 1),
            }
            for row in rows
        ]
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to read notes: {exc}")
        return []


def delete_note(note_id):
    """Delete a single note by id. Returns True on success."""
    try:
        with _lock, _connect() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
            conn.commit()
        return True
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to delete note {note_id}: {exc}")
        return False


def get_stats():
    """Return aggregated KPIs for the dashboard.

    Includes total words, sessions, minutes used/saved, and the average number
    of words captured per active day.
    """
    defaults = {
        "total_words": 0,
        "total_sessions": 0,
        "total_minutes_used": 0.0,
        "total_minutes_saved": 0.0,
        "avg_daily_words": 0,
        "active_days": 0,
    }
    try:
        with _lock, _connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(words_entered), 0),
                       COUNT(*),
                       COALESCE(SUM(duration_seconds), 0),
                       COALESCE(SUM(time_saved_seconds), 0),
                       COUNT(DISTINCT substr(timestamp, 1, 10))
                FROM analytics
                """
            ).fetchone()
        total_words = int(row[0])
        active_days = int(row[4]) or 0
        avg_daily = round(total_words / active_days) if active_days else 0
        return {
            "total_words": total_words,
            "total_sessions": int(row[1]),
            "total_minutes_used": round(row[2] / 60.0, 1),
            "total_minutes_saved": round(row[3] / 60.0, 1),
            "avg_daily_words": int(avg_daily),
            "active_days": active_days,
        }
    except sqlite3.Error as exc:
        print(f"[Betaal][db] Failed to read stats: {exc}")
        return defaults
