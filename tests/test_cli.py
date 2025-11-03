"""
Tests for CLI functionality and policy compliance
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from rag.__main__ import main, create_parser, handle_single_query


class TestCLIParser:
    """Test CLI argument parsing"""

    def test_parser_creation(self):
        """Test that parser is created with correct configuration"""
        parser = create_parser()
        assert parser.prog == "rag"
        assert "Agentic RAG Transformer" in parser.description

    def test_version_argument(self):
        """Test --version argument"""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_argument(self):
        """Test --help argument"""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_query_argument(self):
        """Test --query argument parsing"""
        parser = create_parser()
        args = parser.parse_args(["--query", "test question"])
        assert args.query == "test question"

    def test_quiet_argument(self):
        """Test --quiet argument"""
        parser = create_parser()
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

        # Test short form
        args = parser.parse_args(["-q"])
        assert args.quiet is True

    def test_verbose_argument(self):
        """Test --verbose argument"""
        parser = create_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_no_color_argument(self):
        """Test --no-color argument"""
        parser = create_parser()
        args = parser.parse_args(["--no-color"])
        assert args.no_color is True


class TestCLIFunctionality:
    """Test CLI functionality and user interactions"""

    @patch("rag.__main__.RAGEngine")
    def test_single_query_mode(self, mock_rag_engine):
        """Test single query mode functionality"""
        mock_engine = MagicMock()
        mock_engine.generate_response.return_value = "Test response"
        mock_rag_engine.return_value = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            handle_single_query("test question")
            output = mock_stdout.getvalue()
            assert "Test response" in output

    @patch("rag.__main__.RAGEngine")
    def test_single_query_error_handling(self, mock_rag_engine):
        """Test error handling in single query mode"""
        mock_rag_engine.side_effect = Exception("Test error")

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            with pytest.raises(SystemExit) as exc_info:
                handle_single_query("test question")
            assert exc_info.value.code == 1
            assert "Error: Test error" in mock_stderr.getvalue()

    @patch("rag.__main__.interactive_mode")
    def test_main_interactive_mode(self, mock_interactive):
        """Test main function calls interactive mode by default"""
        main([])
        mock_interactive.assert_called_once()

    @patch("rag.__main__.handle_single_query")
    def test_main_single_query_mode(self, mock_single_query):
        """Test main function handles single query mode"""
        main(["--query", "test"])
        mock_single_query.assert_called_once_with("test", False, False, False)


class TestCLIPolicy:
    """Test CLI policy compliance"""

    def test_exit_codes(self):
        """Test that proper exit codes are used"""
        # Test successful execution (should not raise SystemExit)
        parser = create_parser()
        args = parser.parse_args([])
        assert args is not None

        # Test help exits with code 0
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_error_output_to_stderr(self):
        """Test that errors are written to stderr"""
        with patch("rag.__main__.RAGEngine") as mock_rag_engine:
            mock_rag_engine.side_effect = Exception("Test error")

            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    handle_single_query("test")

                error_output = mock_stderr.getvalue()
                assert "Error:" in error_output
                assert "Test error" in error_output

    def test_standard_options_available(self):
        """Test that all standard CLI options are available"""
        parser = create_parser()

        # Test that standard options exist
        help_action = None
        version_action = None
        verbose_action = None

        for action in parser._actions:
            if "--help" in action.option_strings:
                help_action = action
            elif "--version" in action.option_strings:
                version_action = action
            elif "--verbose" in action.option_strings:
                verbose_action = action

        assert help_action is not None, "Missing --help option"
        assert version_action is not None, "Missing --version option"
        assert verbose_action is not None, "Missing --verbose option"


class TestInteractiveMode:
    """Test interactive mode functionality"""

    @patch("sys.stdin.isatty")
    @patch("rag.__main__.RAGEngine")
    @patch("builtins.input")
    def test_interactive_exit_commands(self, mock_input, mock_rag_engine, mock_isatty):
        """Test that exit commands work in interactive mode"""
        mock_isatty.return_value = True
        mock_engine = MagicMock()
        mock_rag_engine.return_value = mock_engine

        # Test 'exit' command
        mock_input.side_effect = ["exit"]

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            from rag.__main__ import interactive_mode

            interactive_mode()
            output = mock_stdout.getvalue()
            assert "Goodbye!" in output

    @patch("sys.stdin.isatty")
    @patch("rag.__main__.RAGEngine")
    @patch("builtins.input")
    def test_interactive_help_command(self, mock_input, mock_rag_engine, mock_isatty):
        """Test help command in interactive mode"""
        mock_isatty.return_value = True
        mock_engine = MagicMock()
        mock_rag_engine.return_value = mock_engine

        mock_input.side_effect = ["help", "exit"]

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            from rag.__main__ import interactive_mode

            interactive_mode()
            output = mock_stdout.getvalue()
            assert "RAG Transformer Help:" in output
            assert "Machine Learning" in output

    @patch("sys.stdin.isatty")
    def test_non_interactive_detection(self, mock_isatty):
        """Test detection of non-interactive environment"""
        mock_isatty.return_value = False

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            from rag.__main__ import interactive_mode

            interactive_mode()
            output = mock_stdout.getvalue()
            assert "Non-interactive environment detected" in output


class TestCLIIntegration:
    """Integration tests for CLI functionality"""

    @patch("rag.__main__.RAGEngine")
    def test_verbose_mode_integration(self, mock_rag_engine):
        """Test verbose mode provides additional output"""
        mock_engine = MagicMock()
        mock_engine.generate_response.return_value = "Test response"
        mock_rag_engine.return_value = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            handle_single_query("test question", verbose=True)
            output = mock_stdout.getvalue()
            assert "Processing query:" in output

    @patch("rag.__main__.RAGEngine")
    def test_quiet_mode_integration(self, mock_rag_engine):
        """Test quiet mode suppresses non-essential output"""
        mock_engine = MagicMock()
        mock_engine.generate_response.return_value = "Test response"
        mock_rag_engine.return_value = mock_engine

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            handle_single_query("test question", verbose=True, quiet=True)
            output = mock_stdout.getvalue()
            # Should not contain verbose processing message when quiet is True
            assert "Processing query:" not in output
            # But should still contain the response
            assert "Test response" in output

    def test_no_color_mode_integration(self):
        """Test no-color mode removes emojis from output"""
        from rag.__main__ import format_message

        # Test format_message directly with explicit parameters
        # When should_use_color would return True (mocked scenario)
        with patch("rag.__main__.should_use_color", return_value=True):
            colored_msg = format_message("Hello", "🤖", no_color=False)
            assert "🤖" in colored_msg
            assert "Hello" in colored_msg

        # When should_use_color would return False (no_color=True overrides)
        with patch("rag.__main__.should_use_color", return_value=False):
            plain_msg = format_message("Hello", "🤖", no_color=True)
            assert "🤖" not in plain_msg
            assert "Hello" in plain_msg

    @patch.dict("os.environ", {"NO_COLOR": "1"})
    def test_no_color_environment_variable(self):
        """Test that NO_COLOR environment variable is respected"""
        from rag.__main__ import should_use_color

        # Should return False when NO_COLOR is set
        assert should_use_color(no_color=False) is False

    @patch("sys.stdout.isatty", return_value=False)
    def test_no_color_non_tty(self, mock_isatty):
        """Test that color is disabled for non-TTY output"""
        from rag.__main__ import should_use_color

        # Should return False when output is not to a terminal
        assert should_use_color(no_color=False) is False

    def test_cli_policy_documentation_exists(self):
        """Test that CLI policy documentation exists"""
        import os

        policy_file = "CLI_POLICY.md"
        assert os.path.exists(policy_file), "CLI_POLICY.md file should exist"

        with open(policy_file, "r") as f:
            content = f.read()
            assert "CLI Policy and Standards" in content
            assert "POSIX Compliance" in content
            assert "Exit Codes" in content


if __name__ == "__main__":
    pytest.main([__file__])
