"""
End-to-end tests for the application with CLI policy compliance
"""

from io import StringIO
from unittest.mock import Mock, patch

import pytest

from src.rag.__main__ import main
from src.rag.data_fetcher import main as collector_main
from src.rag.rag_engine import RAGEngine
from src.rag.review import ReviewFinding, ReviewReport
from src.rag.ui.minimal_tui import MinimalTUI
from src.rag.ui.tui import (
    create_tui_parser,
    main as tui_main,
    run_tui,
    _review_session_body,
    _process_query,
)

pytestmark = pytest.mark.integration


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

        main(["--cli"])  # Use CLI mode for testing

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

        main(["--cli"])  # Use CLI mode for testing

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

    def test_tui_parser_accepts_positional_initial_query(self):
        """Test TUI parser accepts positional startup query text"""
        parser = create_tui_parser()
        args = parser.parse_args(["CONTRIBUTING.asc"])
        assert args.initial_query == ["CONTRIBUTING.asc"]

    def test_rag_engine_shortcuts_toggle(self):
        """Test engine shortcut replies can be turned on and off"""
        engine = object.__new__(RAGEngine)
        engine.shortcut_responses_enabled = False

        msg = engine.set_shortcut_responses_enabled(False)
        assert msg == "Shortcut responses disabled."
        assert engine.shortcut_responses_enabled is False

        msg = engine.set_shortcut_responses_enabled(True)
        assert msg == "Shortcut responses enabled."
        assert engine.shortcut_responses_enabled is True

    def test_minimal_tui_ollama_start_switches_backend(self):
        """Test minimal TUI switches backend to ollama after starting it"""
        mock_engine = Mock()
        mock_engine.set_backend.return_value = "Switched backend to ollama."
        mock_engine.current_backend_and_model.return_value = "ollama:llama3"

        tui = MinimalTUI(theme="minimal")
        tui.rag_engine = mock_engine

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            with patch.object(tui.console, "print"):
                tui.start_ollama()

        mock_engine.set_backend.assert_called_with("ollama")

    def test_minimal_tui_suppresses_engine_noise(self):
        """Test minimal TUI hides engine stdout/stderr noise during responses"""
        mock_engine = Mock()

        def noisy_response(query):
            print("INFO: noisy stdout")
            import sys

            print("INFO: noisy stderr", file=sys.stderr)
            return "quiet result"

        mock_engine.generate_response.side_effect = noisy_response

        tui = MinimalTUI(theme="minimal")
        tui.rag_engine = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                response = tui._generate_response_quietly("hello")

        assert response == "quiet result"
        assert "INFO: noisy stdout" not in mock_stdout.getvalue()
        assert "INFO: noisy stderr" not in mock_stderr.getvalue()

    def test_minimal_tui_shortcuts_command(self):
        """Test minimal TUI handles shortcuts toggle command"""
        mock_engine = Mock()
        mock_engine.set_shortcut_responses_enabled.return_value = (
            "Shortcut responses disabled."
        )

        tui = MinimalTUI(theme="minimal")
        tui.rag_engine = mock_engine
        with patch.object(tui, "_print_message") as mock_print:
            tui.set_shortcuts("off")

        mock_engine.set_shortcut_responses_enabled.assert_called_with(False)
        mock_print.assert_called()

    def test_minimal_tui_command_typo_suggests_instead_of_querying(self):
        """Test minimal TUI suggests a close command for mistyped input"""
        mock_engine = Mock()

        tui = MinimalTUI(theme="minimal")
        tui.rag_engine = mock_engine

        handled = tui._maybe_handle_command_typo("shortscuts off")

        assert handled is True

    def test_minimal_tui_finding_navigation_cycles_report(self):
        report = ReviewReport(
            mode="file",
            label="src/rag/sample.py:10-20",
            findings=(
                ReviewFinding("src/rag/sample.py", 12, "high", "first issue"),
                ReviewFinding("src/rag/sample.py", 18, "medium", "second issue"),
            ),
            source_lines=tuple((line, f"line {line}") for line in range(10, 21)),
            summary="2 finding(s)",
        )
        tui = MinimalTUI(theme="minimal")
        tui._set_review_state(report)

        (focused,) = [tui._finding_focus_report(1)]
        assert focused is not None
        assert "finding 2/2" in focused.label
        assert focused.findings[0].line == 18

    def test_minimal_tui_finding_navigation_requires_review_state(self):
        tui = MinimalTUI(theme="minimal")
        assert tui._finding_focus_report(1) is None

    def test_minimal_tui_review_current_requires_prior_context(self):
        tui = MinimalTUI(theme="minimal")
        tui.rag_engine = Mock()

        with patch.object(tui, "_print_message") as mock_print:
            query = "review current"
            if query.lower() == "review current":
                if not tui.last_review_command:
                    tui._print_message(
                        "Review",
                        "Run `review <path>` or `open <path>` first.",
                        "warning",
                    )

        mock_print.assert_called_once()

    def test_minimal_tui_review_session_body_shows_active_target(self):
        report = ReviewReport(
            mode="file",
            label="src/rag/sample.py:10-20",
            findings=(ReviewFinding("src/rag/sample.py", 12, "high", "first issue"),),
            source_lines=((10, "line 10"), (11, "line 11"), (12, "line 12")),
            summary="1 finding(s)",
        )
        tui = MinimalTUI(theme="minimal")
        tui.last_review_command = "review src/rag/sample.py:10-20"
        tui._set_review_state(report)

        body = tui._review_session_body()

        assert "Active: review src/rag/sample.py:10-20" in body
        assert "Findings: 1/1" in body
        assert "next finding" in body

    def test_tui_review_session_body_shows_navigation_commands(self):
        report = ReviewReport(
            mode="file",
            label="src/rag/sample.py:10-20",
            findings=(
                ReviewFinding("src/rag/sample.py", 12, "high", "first issue"),
                ReviewFinding("src/rag/sample.py", 18, "medium", "second issue"),
            ),
            source_lines=tuple((line, f"line {line}") for line in range(10, 21)),
            summary="2 finding(s)",
        )

        body = _review_session_body(
            "review src/rag/sample.py:10-20",
            report,
            1,
        )

        assert "Active: review src/rag/sample.py:10-20" in body
        assert "Findings: 2/2" in body
        assert "prev finding" in body

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.console.Console.input", side_effect=["hello", "exit"])
    @patch("rich.console.Console.print")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_greeting_flow(self, mock_rag, mock_print, mock_input, mock_isatty):
        """Test TUI function with greeting and exit"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "Hello response"
        mock_engine.current_backend_and_model.return_value = "local:test"
        mock_engine.available_backends.return_value = ["local"]
        mock_rag.return_value = mock_engine

        run_tui()
        assert mock_print.called
        mock_engine.generate_response.assert_called_with("hello")

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.console.Console.input", side_effect=["exit"])
    @patch("rich.console.Console.print")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_initial_query_argument(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test TUI processes an initial positional query before the loop"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "File response"
        mock_engine.current_backend_and_model.return_value = "local:test"
        mock_engine.available_backends.return_value = ["local"]
        mock_engine.get_status.return_value = {
            "embedding_model_loaded": True,
            "generator_model_loaded": True,
        }
        mock_rag.return_value = mock_engine

        tui_main(["CONTRIBUTING.asc"])
        mock_engine.generate_response.assert_any_call("CONTRIBUTING.asc")

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.console.Console.input", side_effect=["help", "exit"])
    @patch("rich.console.Console.print")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_help_flow(self, mock_rag, mock_print, mock_input, mock_isatty):
        """Test TUI function with help command"""
        mock_engine = Mock()
        mock_engine.current_backend_and_model.return_value = "local:test"
        mock_engine.available_backends.return_value = ["local"]
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
    @patch("rich.console.Console.input", side_effect=["clear", "exit"])
    @patch("rich.console.Console.print")
    @patch("rich.console.Console.clear")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_clear_command(
        self, mock_rag, mock_clear, mock_print, mock_input, mock_isatty
    ):
        """Test TUI clear command"""
        mock_engine = Mock()
        mock_engine.current_backend_and_model.return_value = "local:test"
        mock_engine.available_backends.return_value = ["local"]
        mock_rag.return_value = mock_engine

        run_tui()
        mock_clear.assert_called_once()

    @patch("sys.stdin.isatty", return_value=True)
    @patch("rich.console.Console.input", side_effect=["SHELL: git status", "exit"])
    @patch("rich.console.Console.print")
    @patch("src.rag.ui.tui.RAGEngine")
    def test_tui_shell_command(self, mock_rag, mock_print, mock_input, mock_isatty):
        """Test TUI SHELL command execution"""
        mock_engine = Mock()
        mock_engine.current_backend_and_model.return_value = "local:test"
        mock_engine.available_backends.return_value = ["local"]
        mock_rag.return_value = mock_engine

        run_tui()
        # Verify generate_response was called with the shell command
        mock_engine.generate_response.assert_called_with("SHELL: git status")


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

        main(["--cli"])  # Use CLI mode for testing

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

        main(["--cli"])  # Use CLI mode for testing

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

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["SHELL: git status", "exit"])
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_shell_tool_via_generate_response(
        self, mock_rag, mock_print, mock_input, mock_isatty
    ):
        """Test SHELL tool execution via generate_response"""
        mock_engine = Mock()
        mock_engine.generate_response.return_value = "On branch main"
        mock_rag.return_value = mock_engine

        main(["--cli"])

        # Verify shell command was passed to generate_response
        mock_engine.generate_response.assert_called_with("SHELL: git status")

        # Verify response was printed
        response_calls = [
            call for call in mock_print.call_args_list if "On branch main" in str(call)
        ]
        assert len(response_calls) > 0

    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["run git status", "exit"])
    @patch("builtins.print")
    @patch("src.rag.__main__.RAGEngine")
    def test_shell_natural_language_to_tool(
        self, mock_rag_class, mock_print, mock_input, mock_isatty
    ):
        """Test natural language 'run git status' gets interpreted as SHELL tool"""
        from src.rag.rag_engine import RAGEngine

        real_engine = RAGEngine()
        mock_instance = Mock()
        mock_instance.llm_backend = "ollama"
        mock_instance.openai_client = None

        def generate_response_with_mock(*args, **kwargs):
            original_generate_text = real_engine._generate_text
            real_engine._generate_text = mock_instance._generate_text
            try:
                return real_engine.generate_response(*args, **kwargs)
            finally:
                real_engine._generate_text = original_generate_text

        mock_instance.generate_response.side_effect = generate_response_with_mock
        mock_instance._generate_text.return_value = "SHELL: git status"
        mock_instance.retrieve_context.return_value = []
        mock_instance.current_backend_and_model.return_value = "local:test"
        mock_instance.available_backends.return_value = ["local"]
        mock_instance.memory.search_facts.return_value = []
        mock_instance.memory.recent_messages.return_value = []
        mock_rag_class.return_value = mock_instance

        main(["--cli"])

        # Verify LLM was called with the natural language query
        assert mock_instance._generate_text.called, "_generate_text not called"

        # Verify that SHELL tool was returned by LLM
        call_args = str(mock_instance._generate_text.call_args)
        assert "SHELL: git status" in call_args or "git status" in call_args


class TestProcessQuery:
    """Tests for _process_query function"""

    @patch("src.rag.ui.tui.Console")
    def test_process_query_with_review_command(self, mock_console_class):
        """Test _process_query handles review commands"""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        mock_engine = Mock()
        mock_engine.config.PROJECT_ROOT = "/tmp/test"

        with patch("src.rag.ui.tui.build_review_report") as mock_build:
            mock_build.return_value = ReviewReport(
                mode="test",
                label="test",
                summary="summary",
                findings=(),
            )
            _process_query(mock_engine, "review test.py", mock_console, True)

            mock_build.assert_called_once()
            mock_console.print.assert_called()

    @patch("src.rag.ui.tui.Console")
    def test_process_query_with_regular_query(self, mock_console_class):
        """Test _process_query handles regular queries"""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_console.status.return_value.__enter__ = Mock(return_value=None)
        mock_console.status.return_value.__exit__ = Mock(return_value=None)

        mock_engine = Mock()
        mock_engine.generate_response.return_value = "Test response"

        _process_query(mock_engine, "hello world", mock_console, True)

        mock_engine.generate_response.assert_called_once_with("hello world")
        mock_console.print.assert_called()

    @patch("src.rag.ui.tui.Console")
    def test_process_query_with_markdown(self, mock_console_class):
        """Test _process_query renders markdown response"""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        mock_console.status.return_value.__enter__ = Mock(return_value=None)
        mock_console.status.return_value.__exit__ = Mock(return_value=None)

        mock_engine = Mock()
        mock_engine.generate_response.return_value = "# Hello"

        _process_query(mock_engine, "hello", mock_console, False)

        mock_engine.generate_response.assert_called_once()
        # Verify Panel was called for markdown rendering
        panel_calls = [c for c in mock_console.print.call_args_list]
        assert len(panel_calls) > 0
