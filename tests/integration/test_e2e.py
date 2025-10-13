"""
End-to-end tests for the application
"""

from unittest.mock import Mock, patch

from src.rag.__main__ import main
from src.rag.ui.tui import run_tui


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["hello", "exit"])
@patch("builtins.print")
@patch("src.rag.__main__.RAGEngine")
def test_main_greeting_flow(mock_rag, mock_print, mock_input, mock_isatty):
    """Test main function with greeting and exit"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Hello response"
    mock_rag.return_value = mock_engine

    main()
    mock_print.assert_any_call("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    mock_print.assert_any_call("\nResponse: Hello response")


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["calculate 2+3", "exit"])
@patch("builtins.print")
@patch("src.rag.__main__.RAGEngine")
def test_main_calc_flow(mock_rag, mock_print, mock_input, mock_isatty):
    """Test main function with calculation"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Calculation result: 5"
    mock_rag.return_value = mock_engine

    main()
    mock_engine.generate_response.assert_called_with("calculate 2+3")
    mock_print.assert_any_call("\nResponse: Calculation result: 5")


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["help", "exit"])
@patch("builtins.print")
@patch("src.rag.__main__.RAGEngine")
def test_main_help_flow(mock_rag, mock_print, mock_input, mock_isatty):
    """Test main function with help command"""
    mock_engine = Mock()
    mock_rag.return_value = mock_engine

    main()
    help_calls = [
        call for call in mock_print.call_args_list if "This is an Agentic AI Assistant" in str(call)
    ]
    assert len(help_calls) > 0


@patch("sys.stdin.isatty", return_value=True)
@patch("rich.prompt.Prompt.ask", side_effect=["hello", "exit"])
@patch("rich.console.Console.print")
@patch("src.rag.ui.tui.RAGEngine")
@patch("rich.panel.Panel")  # Mock Panel for CI-safe test
def test_tui_greeting_flow(mock_panel, mock_rag, mock_print, mock_ask, mock_isatty):
    """Test TUI function with greeting and exit"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Hello response"
    mock_rag.return_value = mock_engine

    run_tui()
    assert mock_print.called
    mock_engine.generate_response.assert_called_with("hello")


@patch("sys.stdin.isatty", return_value=True)
@patch("rich.prompt.Prompt.ask", side_effect=["help", "exit"])
@patch("rich.console.Console.print")
@patch("src.rag.ui.tui.RAGEngine")
@patch("rich.panel.Panel")  # Mock Panel for CI-safe test
def test_tui_help_flow(mock_panel, mock_rag, mock_print, mock_ask, mock_isatty):
    """Test TUI function with help command"""
    mock_engine = Mock()
    mock_rag.return_value = mock_engine

    run_tui()
    assert mock_print.called  # Help text is printed


@patch("sys.stdin.isatty", return_value=True)
@patch("rich.prompt.Prompt.ask", side_effect=["calculate 2+3", "exit"])
@patch("rich.console.Console.print")
@patch("src.rag.ui.tui.RAGEngine")
@patch("rich.panel.Panel")  # Mock Panel for CI-safe test
def test_tui_calc_flow(mock_panel, mock_rag, mock_print, mock_ask, mock_isatty):
    """Test TUI function with calculation"""
    mock_engine = Mock()
    mock_engine.generate_response.return_value = "Calculation result: 5"
    mock_rag.return_value = mock_engine

    run_tui()
    mock_engine.generate_response.assert_called_with("calculate 2+3")
    assert mock_print.called  # Response is printed


@patch("sys.stdin.isatty", return_value=True)
@patch("rich.prompt.Prompt.ask", side_effect=["exit"])
@patch("rich.console.Console.print")
@patch("src.rag.ui.tui.RAGEngine")
@patch("rich.panel.Panel")  # Mock Panel for CI-safe test
def test_tui_exit_flow(mock_panel, mock_rag, mock_print, mock_ask, mock_isatty):
    """Test TUI function with immediate exit"""
    mock_engine = Mock()
    mock_rag.return_value = mock_engine

    run_tui()
    # Should not crash and should print goodbye
    goodbye_calls = [call for call in mock_print.call_args_list if "Goodbye" in str(call)]
    assert len(goodbye_calls) > 0
