"""
Lightweight conversation memory for RAG.

Design goals:
- No new dependencies (stdlib sqlite3).
- Privacy-friendly defaults (session-only unless enabled).
- Useful without embeddings (simple keyword search + recent context).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class MemoryFact:
    key: str
    value: str
    updated_at: float


class MemoryStore:
    """SQLite-backed store (supports ':memory:' for session-only)."""

    def __init__(self, db_path: Path | str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        if self.enabled:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_message(self, role: str, content: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            conn.execute(
                "INSERT INTO messages (ts, role, content) VALUES (?, ?, ?)",
                (time.time(), role, content),
            )
            conn.commit()

    def set_fact(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            conn.execute(
                "INSERT INTO facts (key, value, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (key, value, now),
            )
            conn.commit()

    def get_fact(self, key: str) -> Optional[MemoryFact]:
        if not self.enabled:
            return None
        with self._lock:
            conn = self._conn
            if conn is None:
                return None
            cur = conn.execute("SELECT key, value, ts FROM facts WHERE key=?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        return MemoryFact(key=row[0], value=row[1], updated_at=float(row[2]))

    def recent_messages(self, limit: int = 10) -> list[tuple[str, str]]:
        if not self.enabled:
            return []
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            cur = conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [(str(r), str(c)) for (r, c) in rows]

    def search_facts(self, query: str, limit: int = 8) -> list[MemoryFact]:
        if not self.enabled:
            return []
        tokens = [t.strip().lower() for t in query.split() if t.strip()]
        if not tokens:
            return []

        # Simple token matching on key/value (LIKE). Good enough to be useful without embeddings.
        where_clauses: list[str] = []
        params: list[str] = []
        for t in tokens[:6]:
            where_clauses.append("(lower(key) LIKE ? OR lower(value) LIKE ?)")
            like = f"%{t}%"
            params.extend([like, like])

        sql = (
            "SELECT key, value, ts FROM facts WHERE "
            + " AND ".join(where_clauses)
            + " ORDER BY ts DESC LIMIT ?"
        )
        params.append(str(limit))
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [
            MemoryFact(key=str(k), value=str(v), updated_at=float(ts))
            for (k, v, ts) in rows
        ]

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM facts")
            conn.commit()

    @staticmethod
    def build_for_mode(mode: str, cache_dir: Path, db_name: str) -> "MemoryStore":
        mode = (mode or "off").lower()
        if mode == "off":
            return MemoryStore(":memory:", enabled=False)
        if mode == "session":
            return MemoryStore(":memory:", enabled=True)
        # persist
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / db_name
        return MemoryStore(db_path, enabled=True)


def format_memory_context(
    facts: Iterable[MemoryFact], messages: Iterable[tuple[str, str]]
) -> str:
    fact_lines = [f"- {f.key}: {f.value}" for f in facts]
    msg_lines = [f"{role}: {content}" for (role, content) in messages]

    parts: list[str] = []
    if fact_lines:
        parts.append("Remembered facts:\n" + "\n".join(fact_lines))
    if msg_lines:
        parts.append("Recent conversation:\n" + "\n".join(msg_lines))
    return "\n\n".join(parts).strip()
