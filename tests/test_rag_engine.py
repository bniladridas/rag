"""
Unit tests for rag_engine.py
"""

import pytest

pytestmark = pytest.mark.unit

from unittest.mock import Mock, patch  # noqa: E402

import numpy as np  # noqa: E402

from src.rag.rag_engine import RAGEngine  # noqa: E402


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
    config.DEVICE = "cpu"
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
    mock_embed.assert_called_once_with("test-model", device="cpu")
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


@patch("src.rag.rag_engine.RAGEngine.load_knowledge_base")
@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
@patch("src.rag.rag_engine.SentenceTransformer")
@patch(
    "src.rag.rag_engine.AutoModelForSeq2SeqLM.from_pretrained",
    side_effect=Exception("Load failed"),
)
@patch(
    "src.rag.rag_engine.AutoTokenizer.from_pretrained",
    side_effect=Exception("Load failed"),
)
def test_generator_model_failure(
    mock_tokenizer, mock_model, mock_embed, mock_tool, mock_config_class, mock_load_kb
):
    """Test initialization when generator model fails to load"""
    mock_config = Mock()
    mock_config_class.return_value = mock_config

    engine = RAGEngine()
    assert engine.tokenizer is None
    assert engine.generator is None


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
def test_load_knowledge_base_file_not_found(mock_tool, mock_config_class):
    """Test loading knowledge base when file is not found"""
    mock_config = Mock()
    mock_config.DATASET_DIR = "nonexistent"
    mock_config.KNOWLEDGE_BASE_FILE = "missing.json"
    mock_config_class.return_value = mock_config

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.config = mock_config
        engine.add_documents = Mock()
        engine.load_knowledge_base()
        engine.add_documents.assert_called_with(
            [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Science fiction explores futuristic concepts and advanced technology.",
            ]
        )


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
def test_generate_response_no_models(mock_tool, mock_config_class):
    """Test generate_response when models are not loaded (CI fallback)"""
    mock_config = Mock()
    mock_config.MAX_ITERATIONS = 1
    mock_config_class.return_value = mock_config

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.tool_executor = Mock()
        engine.retrieve_context = Mock(return_value=["context doc"])
        engine.tokenizer = None
        engine.generator = None
        result = engine.generate_response("some query")
        assert "context doc" in result


@patch("src.rag.rag_engine.Config")
@patch("src.rag.rag_engine.ToolExecutor")
@patch("src.rag.rag_engine.SentenceTransformer")
@patch("src.rag.rag_engine.AutoTokenizer")
@patch("src.rag.rag_engine.AutoModelForSeq2SeqLM")
def test_generate_response_with_tools(
    mock_model, mock_tokenizer, mock_embed, mock_tool, mock_config_class
):
    """Test generate_response with tool invocation in generation loop"""
    mock_config = Mock()
    mock_config.MAX_ITERATIONS = 2
    mock_config.MAX_LENGTH = 50
    mock_config_class.return_value = mock_config

    mock_tokenizer_instance = Mock()
    mock_tokenizer.return_value.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.decode.side_effect = [
        "CALC: 2+2",
        "The result of the calculation is 4.",
    ]
    mock_generator_instance = Mock()
    mock_model.return_value.from_pretrained.return_value = mock_generator_instance
    mock_generator_instance.generate.return_value = [Mock()]

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.tool_executor = Mock()
        engine.tool_executor.execute_tool.return_value = "4"
        engine.tool_executor.get_available_tools.return_value = "Tools: CALC, WIKI"
        engine.retrieve_context = Mock(return_value=["context"])
        engine.tokenizer = mock_tokenizer_instance
        engine.generator = mock_generator_instance
        engine.config = mock_config
        result = engine.generate_response("calculate something")
        assert result == "4"
