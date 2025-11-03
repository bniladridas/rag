"""
End-to-end tests for the application with CLI policy compliance
"""

from io import StringIO
from unittest.mock import Mock, patch

from src.rag.__main__ import main
from src.rag.data_fetcher import main as collector_main
from src.rag.ui.tui import run_tui


class TestCLIE2E:
    """End-to-end tests for CLI functionality"""

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["hello", "exit"])
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_main_interactive_greeting_flow(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test main function interactive mode with greeting and exit"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "Hello response"
        mock_rag.return_value = mock_engine

        main([])  # Pass empty args to avoid pytest interference

        # Check welcome message
        welcome_calls = [
            call
            for call in mock_print.call_args_list
            if "Agentic RAG Transformer" in str(call)
        ]
        assert len(welcome_calls) > 0

        # Check response (could be with or without emoji depending on color settings)
        response_calls = [
            call for call in mock_print.call_args_list if "Hello response" in str(call)
        ]
        assert len(response_calls) > 0

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["help", "exit"])
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_main_interactive_help_flow(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test main function interactive mode with help command"""
        mock_rag.return_value.generate_response.return_value = "Help response"

        main([])  # Pass empty args to avoid pytest interference

        # Check help content
        help_calls = [
            call
            for call in mock_print.call_args_list
            if "RAG Transformer Help:" in str(call)
        ]
        assert len(help_calls) > 0

    @patch("src.rag.__main__.RAGEngine")
    def test_main_single_query_mode(self, mock_rag):
        """Test main function single query mode"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "Test response"
        mock_rag.return_value = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["--query", "test question"])
            output = mock_stdout.getvalue()
            assert "Test response" in output

    @patch("src.rag.__main__.RAGEngine")
    def test_main_verbose_mode(self, mock_rag):
        """Test main function with verbose flag"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "Test response"
        mock_rag.return_value = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["--query", "test", "--verbose"])
            output = mock_stdout.getvalue()
            assert "Processing query: test" in output

    def test_main_version_flag(self):
        """Test main function with version flag"""
        with patch("sys.exit") as mock_exit:
            main(["--version"])
            mock_exit.assert_called_with(0)

    def test_main_help_flag(self):
        """Test main function with help flag"""
        with patch("sys.exit") as mock_exit:
            main(["--help"])
            mock_exit.assert_called_with(0)


class TestTUIE2E:
    """End-to-end tests for TUI functionality"""

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.prompt.Prompt.ask", side_effect=["hello", "exit"])
    @patch("rich.console.Console.print")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_greeting_flow(self, mock_rag, mock_print, mock_ask, mock_isatty):
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
    def test_tui_help_flow(self, mock_rag, mock_print, mock_ask, mock_isatty):
        """Test TUI function with help command"""
        mock_engine = Mock()
        mock_rag.return_value = mock_engine

        run_tui()
        # Verify print was called multiple times for help display
        assert mock_print.call_count > 2  # Welcome + help panel + other content

    @patch("sys.stdin.isatty", return_value=False)
    @patch("builtins.print")
    def test_tui_non_interactive_detection(self, mock_print, mock_isatty):
        """Test TUI detects non-interactive environment"""
        run_tui()

        # Should print non-interactive message
        non_interactive_calls = [
            call
            for call in mock_print.call_args_list
            if "Non-interactive environment detected" in str(call)
        ]
        assert len(non_interactive_calls) > 0

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.prompt.Prompt.ask", side_effect=["clear", "exit"])
    @patch("rich.console.Console.print")
    @patch("rich.console.Console.clear")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_clear_command(
        self, mock_rag, mock_clear, mock_print, mock_ask, mock_isatty
    ):
        """Test TUI clear command"""
        mock_engine = Mock()
        mock_rag.return_value = mock_engine

        run_tui()
        mock_clear.assert_called_once()


class TestDataCollectorE2E:
    """End-to-end tests for data collection functionality"""

    @patch("src.rag.data_fetcher.DataFetcher")
    def test_collector_dry_run(self, mock_fetcher_class):
        """Test data collector dry run mode"""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = collector_main(["--dry-run"])
            output = mock_stdout.getvalue()

            assert "Dry run mode" in output
            assert "Machine Learning concepts" in output
            assert result == 0

    @patch("src.rag.data_fetcher.DataFetcher")
    def test_collector_verbose_mode(self, mock_fetcher_class):
        """Test data collector verbose mode"""
        mock_fetcher = Mock()
        mock_fetcher.fetch_all_data.return_value = ["doc1", "doc2"]
        mock_fetcher_class.return_value = mock_fetcher

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = collector_main(["--verbose"])
            output = mock_stdout.getvalue()

            assert "RAG Transformer Data Collection Tool" in output
            assert "Starting data collection" in output
            assert result == 0

    @patch("src.rag.data_fetcher.DataFetcher")
    def test_collector_error_handling(self, mock_fetcher_class):
        """Test data collector error handling"""
        mock_fetcher_class.side_effect = Exception("Test error")

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = collector_main([])
            error_output = mock_stdout.getvalue()

            assert "Error during data collection" in error_output
            assert result == 1

    def test_collector_version_flag(self):
        """Test data collector version flag"""
        with patch("sys.exit") as mock_exit:
            collector_main(["--version"])
            mock_exit.assert_called_with(0)


class TestCLIPolicyCompliance:
    """Test CLI policy compliance across all commands"""

    def test_all_commands_support_help(self):
        """Test that all CLI commands support --help"""
        commands = [
            (main, ["--help"]),
            (collector_main, ["--help"]),
        ]

        for command_func, args in commands:
            with patch("sys.exit") as mock_exit:
                command_func(args)
                mock_exit.assert_called_with(0)

    def test_all_commands_support_version(self):
        """Test that all CLI commands support --version"""
        commands = [
            (main, ["--version"]),
            (collector_main, ["--version"]),
        ]

        for command_func, args in commands:
            with patch("sys.exit") as mock_exit:
                command_func(args)
                mock_exit.assert_called_with(0)

    @patch("src.rag.__main__.RAGEngine")
    def test_error_exit_codes(self, mock_rag):
        """Test that errors produce proper exit codes"""
        mock_rag.side_effect = Exception("Test error")

        with patch("sys.stderr", new_callable=StringIO):
            with patch("sys.exit") as mock_exit:
                main(["--query", "test"])
                mock_exit.assert_called_with(1)

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=KeyboardInterrupt())
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_keyboard_interrupt_handling(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test graceful handling of Ctrl+C"""
        mock_engine = Mock()
        mock_rag.return_value = mock_engine

        main([])  # Pass empty args to avoid pytest interference

        # Should print goodbye message
        goodbye_calls = [
            call for call in mock_print.call_args_list if "Goodbye!" in str(call)
        ]
        assert len(goodbye_calls) > 0


class TestFullApplicationFlow:
    """Integration tests for complete application workflows"""

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["CALC: 2+2", "WIKI: Python", "TIME:", "exit"])
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_complete_interactive_session(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test a complete interactive session with various commands"""
        mock_engine = Mock()
        mock_engine.generate_response.side_effect = [
            "4",
            "Python is a programming language...",
            "Current time is 12:00 PM",
        ]
        mock_rag.return_value = mock_engine

        main([])  # Pass empty args to avoid pytest interference

        # Verify all queries were processed
        assert mock_engine.generate_response.call_count == 3

        # Check responses were printed (could be with or without emoji depending on color settings)
        calc_calls = [call for call in mock_print.call_args_list if "4" in str(call)]
        wiki_calls = [
            call
            for call in mock_print.call_args_list
            if "Python is a programming language" in str(call)
        ]
        time_calls = [
            call
            for call in mock_print.call_args_list
            if "Current time is 12:00 PM" in str(call)
        ]

        assert len(calc_calls) > 0
        assert len(wiki_calls) > 0
        assert len(time_calls) > 0
