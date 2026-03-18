"""
Lightweight conversation memory for RAG.

Design goals:
- No new dependencies (stdlib sqlite3).
- Privacy-friendly defaults (session-only unless enabled).
- Useful without embeddings (simple keyword search + recent context).
"""

from __future__ import annotations

import sqlite3
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
        if self.enabled:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
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
        if not self.enabled or self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO messages (ts, role, content) VALUES (?, ?, ?)",
            (time.time(), role, content),
        )
        self._conn.commit()

    def set_fact(self, key: str, value: str) -> None:
        if not self.enabled or self._conn is None:
            return
        now = time.time()
        self._conn.execute(
            "INSERT INTO facts (key, value, ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (key, value, now),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> Optional[MemoryFact]:
        if not self.enabled or self._conn is None:
            return None
        cur = self._conn.execute("SELECT key, value, ts FROM facts WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        return MemoryFact(key=row[0], value=row[1], updated_at=float(row[2]))

    def recent_messages(self, limit: int = 10) -> list[tuple[str, str]]:
        if not self.enabled or self._conn is None:
            return []
        cur = self._conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
        rows.reverse()
        return [(str(r), str(c)) for (r, c) in rows]

    def search_facts(self, query: str, limit: int = 8) -> list[MemoryFact]:
        if not self.enabled or self._conn is None:
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
        cur = self._conn.execute(sql, tuple(params))
        return [
            MemoryFact(key=str(k), value=str(v), updated_at=float(ts))
            for (k, v, ts) in cur.fetchall()
        ]

    def clear(self) -> None:
        if not self.enabled or self._conn is None:
            return
        self._conn.execute("DELETE FROM messages")
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()

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
