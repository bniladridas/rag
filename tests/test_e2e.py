"""
End-to-end tests for the application
"""
import pytest
from unittest.mock import patch, Mock
from src.main import main


@patch('builtins.input', side_effect=['hello', 'exit'])
@patch('builtins.print')
@patch('src.main.RAGEngine')
def test_main_greeting_flow(mock_rag, mock_print, mock_input):
    """Test main function with greeting and exit"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Hello response"
    mock_rag.return_value = mock_engine

    main()

    mock_print.assert_any_call("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    mock_print.assert_any_call("\nResponse: Hello response")


@patch('builtins.input', side_effect=['calculate 2+3', 'exit'])
@patch('builtins.print')
@patch('src.main.RAGEngine')
def test_main_calc_flow(mock_rag, mock_print, mock_input):
    """Test main function with calculation"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Calculation result: 5"
    mock_rag.return_value = mock_engine

    main()

    mock_engine.generate_response.assert_called_with('calculate 2+3')
    mock_print.assert_any_call("\nResponse: Calculation result: 5")


@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
@patch('src.main.RAGEngine')
def test_main_help_flow(mock_rag, mock_print, mock_input):
    """Test main function with help command"""
    mock_engine = Mock()
    mock_rag.return_value = mock_engine

    main()

    help_calls = [call for call in mock_print.call_args_list if "This is an Agentic AI Assistant" in str(call)]
    assert len(help_calls) > 0