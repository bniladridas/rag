"""
Unit tests for config.py
"""

import pytest

pytestmark = pytest.mark.unit

from src.rag.config import Config


def test_config_defaults():
    """Test Config class with default values"""
    config = Config()

    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.GENERATOR_MODEL == "google/flan-t5-small"
    assert config.DATASET_DIR.name == "datasets"
    assert config.KNOWLEDGE_BASE_FILE.name == "knowledge_base.json"
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
