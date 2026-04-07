"""
Unit tests for deterministic review helpers.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.rag.memory import MemoryStore
from src.rag.rag_engine import RAGEngine
from src.rag.review import (
    build_open_report,
    build_review_report,
    handle_thread_command,
    load_threads,
    parse_open_target,
    parse_review_target,
    review_command,
)


pytestmark = pytest.mark.unit


def test_parse_review_target_diff():
    target = parse_review_target("review diff")
    assert target.mode == "diff"
    assert target.path is None


def test_parse_review_target_staged():
    target = parse_review_target("review staged")
    assert target.mode == "staged"
    assert target.path is None


def test_parse_review_target_with_range():
    target = parse_review_target("review src/rag/example.py:12-18")
    assert target.mode == "file"
    assert target.path == Path("src/rag/example.py")
    assert target.start_line == 12
    assert target.end_line == 18


def test_review_command_finds_python_issues(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "import subprocess",
                "",
                "def risky(cmd):",
                "    try:",
                "        subprocess.run(cmd, shell=True)",
                "    except:",
                "        return None",
            ]
        ),
        encoding="utf-8",
    )

    result = review_command("review sample.py", tmp_path)

    assert "Review findings for sample.py" in result
    assert "sample.py:5 [high]" in result
    assert "shell=True" in result
    assert "sample.py:6 [medium]" in result
    assert "Bare `except:`" in result


def test_review_command_filters_to_requested_range(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "def risky(expr):",
                "    return eval(expr)",
            ]
        ),
        encoding="utf-8",
    )

    result = review_command("review sample.py:1-1", tmp_path)

    assert result == "No review findings for sample.py:1-1."


def test_build_review_report_returns_source_excerpt(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "def ok():",
                "    return 1",
                "",
                "def risky(expr):",
                "    return eval(expr)",
            ]
        ),
        encoding="utf-8",
    )

    report = build_review_report("review sample.py:4-5", tmp_path)

    assert report is not None
    assert report.label == "sample.py:4-5"
    assert report.summary == "1 finding(s)"
    assert report.source_lines == (
        (4, "def risky(expr):"),
        (5, "    return eval(expr)"),
    )
    assert report.findings[0].line == 5


def test_build_review_report_returns_none_for_diff():
    assert build_review_report("review diff", Path(".")) is None


def test_build_review_report_returns_none_for_staged():
    assert build_review_report("review staged", Path(".")) is None


def test_parse_open_target_with_line():
    target = parse_open_target("open src/rag/example.py:72")
    assert target.path == Path("src/rag/example.py")
    assert target.line == 72


def test_build_open_report_returns_excerpt(tmp_path):
    sample = tmp_path / "README.asc"
    sample.write_text(
        "\n".join(
            [
                "Title",
                "=====",
                "",
                "Line four",
                "Line five",
                "",
                "Line seven",
            ]
        ),
        encoding="utf-8",
    )

    report = build_open_report("open README.asc:4", tmp_path)

    assert report is not None
    assert report.label == "README.asc:4"
    assert report.summary == "Source excerpt"
    assert report.source_lines[0] == (1, "Title")
    assert report.source_lines[-1] == (7, "Line seven")


@patch("src.rag.review.subprocess.run")
def test_review_diff_filters_findings_to_changed_lines(mock_run, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "import subprocess",
                "",
                "def risky(cmd):",
                "    subprocess.run(cmd, shell=True)",
            ]
        ),
        encoding="utf-8",
    )
    mock_run.return_value = Mock(
        returncode=0,
        stdout="\n".join(
            [
                "diff --git a/sample.py b/sample.py",
                "--- a/sample.py",
                "+++ b/sample.py",
                "@@ -4,0 +4,1 @@",
                "+    subprocess.run(cmd, shell=True)",
            ]
        ),
    )

    result = review_command("review diff", tmp_path)

    assert "Review findings for changed lines" in result
    assert "sample.py:4 [high]" in result


@patch("src.rag.review.subprocess.run")
def test_review_staged_uses_staged_diff(mock_run, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("def risky(expr):\n    return eval(expr)\n", encoding="utf-8")
    mock_run.return_value = Mock(
        returncode=0,
        stdout="\n".join(
            [
                "diff --git a/sample.py b/sample.py",
                "--- a/sample.py",
                "+++ b/sample.py",
                "@@ -2,0 +2,1 @@",
                "+    return eval(expr)",
            ]
        ),
    )

    result = review_command("review staged", tmp_path)

    assert "Review findings for changed lines" in result
    assert "sample.py:2 [high]" in result
    assert "--staged" in " ".join(mock_run.call_args.args[0])


def test_handle_thread_command_persists_threads(tmp_path):
    result = handle_thread_command(
        "thread add src/rag/review.py:42 inspect this branch", tmp_path
    )

    assert "Saved thread" in result
    threads = load_threads(tmp_path)
    assert len(threads) == 1
    assert threads[0].path == "src/rag/review.py"
    assert threads[0].line == 42


def test_handle_thread_command_rejects_paths_outside_repo(tmp_path):
    result = handle_thread_command("thread add ../foo.py:1 note", tmp_path)
    assert result == "Thread path must stay inside the repo: ../foo.py"


def test_review_command_supports_non_python_text_files(tmp_path):
    sample = tmp_path / "sample.js"
    sample.write_text("element.innerHTML = userInput;\n", encoding="utf-8")

    result = review_command("review sample.js", tmp_path)

    assert "sample.js:1 [high]" in result
    assert "innerHTML" in result


@patch("src.rag.review.tomllib", None)
@patch("src.rag.review.tomli", None)
def test_review_command_uses_pip_tomli_fallback(tmp_path):
    sample = tmp_path / "pyproject.toml"
    sample.write_text("[tool.ruff\nline-length = 88\n", encoding="utf-8")

    result = review_command("review pyproject.toml", tmp_path)

    assert "pyproject.toml" in result
    assert "Invalid TOML:" in result


def test_review_command_detects_security_todo_in_text_file(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("TODO security: validate auth flow\n", encoding="utf-8")

    result = review_command("review sample.txt", tmp_path)

    assert "sample.txt:1 [medium]" in result
    assert "Security-sensitive TODO" in result


@patch("src.rag.rag_engine.review_command")
def test_generate_response_routes_review_queries(mock_review_command):
    mock_review_command.return_value = "Review findings for sample.py"

    with patch.object(RAGEngine, "__init__", lambda self: None):
        engine = RAGEngine()
        engine.memory = MemoryStore(":memory:", enabled=False)
        engine.openai_client = None
        engine.llm_backend = "local"
        engine.system_instructions = ""
        engine.tool_executor = Mock()
        engine.retrieve_context = Mock(return_value=[])
        engine.shortcut_responses_enabled = False
        engine.config = Mock()
        engine.config.PROJECT_ROOT = Path("/tmp/project")

        result = engine.generate_response("review sample.py")

    assert result == "Review findings for sample.py"
    mock_review_command.assert_called_once_with(
        "review sample.py", Path("/tmp/project")
    )
