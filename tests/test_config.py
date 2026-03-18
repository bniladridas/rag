"""
Unit tests for config.py
"""

import pytest

pytestmark = pytest.mark.unit

from src.rag.config import Config  # noqa: E402


def test_config_defaults():
    """Test Config class with default values"""
    config = Config()

    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.GENERATOR_MODEL == "google/flan-t5-small"
    assert config.DATASET_DIR.name == "datasets"
    assert config.KNOWLEDGE_BASE_FILE.name == "knowledge_base.json"
    assert config.SKILL_FILE.name == "skill.asc"
    assert config.MAX_WORKERS == 5
    assert config.TOP_K_RETRIEVAL == 3
    assert config.MAX_ITERATIONS == 3
    assert config.MAX_LENGTH == 150


def test_config_env_vars(monkeypatch):
    """Test Config loading from environment variables"""
    monkeypatch.setenv("TMDB_API_KEY", "dummy_tmdb")
    monkeypatch.setenv("NASA_API_KEY", "dummy_nasa")
    config = Config()
    assert config.TMDB_API_KEY == "dummy_tmdb"
    assert config.NASA_API_KEY == "dummy_nasa"


def test_config_dedupes_third_party_loggers():
    """Config clears third-party handlers to prevent duplicate log lines."""
    import logging

    third_party = logging.getLogger("huggingface_hub")
    original_handlers = list(third_party.handlers)
    original_propagate = third_party.propagate

    try:
        third_party.handlers[:] = [logging.StreamHandler()]
        third_party.propagate = True

        Config()

        assert third_party.handlers == []
        assert third_party.propagate is True
    finally:
        third_party.handlers[:] = original_handlers
        third_party.propagate = original_propagate
