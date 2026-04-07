"""Unit tests for memory.py."""

import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from src.rag.memory import MemoryStore


def test_memory_session_mode_stores_messages_and_facts(tmp_path: Path):
    store = MemoryStore.build_for_mode("session", tmp_path, "ignored.sqlite3")
    store.set_fact("user_name", "Niladri")
    store.add_message("user", "hello")
    store.add_message("assistant", "hi")

    fact = store.get_fact("user_name")
    assert fact is not None
    assert fact.value == "Niladri"

    msgs = store.recent_messages(limit=10)
    assert msgs == [("user", "hello"), ("assistant", "hi")]


def test_memory_persist_mode_writes_to_disk(tmp_path: Path):
    store1 = MemoryStore.build_for_mode("persist", tmp_path, "memory.sqlite3")
    store1.set_fact("pref", "dark-mode")
    store1.close()

    store2 = MemoryStore.build_for_mode("persist", tmp_path, "memory.sqlite3")
    fact = store2.get_fact("pref")
    assert fact is not None
    assert fact.value == "dark-mode"


def test_memory_session_mode_allows_cross_thread_access(tmp_path: Path):
    store = MemoryStore.build_for_mode("session", tmp_path, "ignored.sqlite3")
    errors: list[Exception] = []

    def worker() -> None:
        try:
            store.set_fact("threaded", "ok")
            store.add_message("assistant", "done")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors == []
    fact = store.get_fact("threaded")
    assert fact is not None
    assert fact.value == "ok"
