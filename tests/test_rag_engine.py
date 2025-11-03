"""
Unit tests for rag_engine.py
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.rag.rag_engine import RAGEngine


@pytest.fixture
def mock_config():
    config = Mock()
    config.EMBEDDING_MODEL = "test-model"
    config.GENERATOR_MODEL = "test-gen"
    config.DATASET_DIR = "test_datasets"
    config.KNOWLEDGE_BASE_FILE = "test_kb.json"
    config.TOP_K_RETRIEVAL = 3
    config.MAX_ITERATIONS = 1
    config.MAX_LENGTH = 50
    return config


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
@patch("src.rag.rag_engine.SentenceTransformer")
@patch("src.rag.rag_engine.AutoTokenizer")
@patch("src.rag.rag_engine.AutoModelForSeq2SeqLM")
@patch("src.rag.rag_engine.faiss.IndexFlatL2")  # Patch faiss for CI
def test_rag_engine_init(
    mock_faiss,
    mock_model,
    mock_tokenizer,
    mock_embed,
    mock_tool,
    mock_config_class,
    mock_config,
):
    mock_config_class.return_value = mock_config
    mock_embedding_model = Mock()
    mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_embed.return_value = mock_embedding_model
    mock_tokenizer.from_pretrained.return_value = Mock()
    mock_model.from_pretrained.return_value = Mock()

    engine = RAGEngine()
    assert engine.config == mock_config
    mock_embed.assert_called_once_with("test-model")
    mock_tokenizer.from_pretrained.assert_called_once_with("test-gen")
    mock_model.from_pretrained.assert_called_once_with("test-gen")


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
def test_generate_response_greeting(mock_tool, mock_config_class):
    mock_config = Mock()
    mock_config.MAX_ITERATIONS = 1
    mock_config_class.return_value = mock_config

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.tool_executor = Mock()
        engine.retrieve_context = Mock(return_value=[])
        result = engine.generate_response("hello")
        assert "Hello! I'm an agentic AI assistant" in result


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
def test_generate_response_calc(mock_tool, mock_config_class):
    mock_config = Mock()
    mock_config.MAX_ITERATIONS = 1
    mock_config_class.return_value = mock_config

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.tool_executor = Mock()
        engine.tool_executor.execute_tool.return_value = "Result: 5"
        engine.retrieve_context = Mock(return_value=[])
        result = engine.generate_response("calculate 2+3")
        assert "Result: 5" in result


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
@patch("src.rag.rag_engine.SentenceTransformer")
@patch("src.rag.rag_engine.AutoTokenizer")
@patch("src.rag.rag_engine.AutoModelForSeq2SeqLM")
def test_retrieve_context(
    mock_model, mock_tokenizer, mock_embed, mock_tool, mock_config_class
):
    mock_config = Mock()
    mock_config.TOP_K_RETRIEVAL = 2
    mock_config_class.return_value = mock_config

    mock_embedding_model = Mock()
    mock_embedding_model.encode.return_value = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
    mock_embed.return_value = mock_embedding_model

    mock_index = Mock()
    mock_index.search.return_value = ([0.1, 0.2], [[0, 1]])
    with patch("src.rag.rag_engine.faiss.IndexFlatL2", return_value=mock_index):
        with patch.object(RAGEngine, "__init__", lambda self: None):
            engine = RAGEngine()
            engine.embedding_model = mock_embedding_model
            engine.knowledge_base = ["doc1", "doc2", "doc3"]
            engine.index = mock_index
            engine.query_cache = {}
            engine.config = mock_config
            result = engine.retrieve_context("test query")
            assert result == ["doc1", "doc2"]
