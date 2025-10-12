"""
Unit tests for tools.py
"""

from unittest.mock import Mock, patch

import pytest

from src.rag.tools import ToolExecutor


@pytest.fixture
def tool_executor():
    return ToolExecutor()


def test_get_available_tools(tool_executor):
    tools = tool_executor.get_available_tools()
    assert "CALC:" in tools
    assert "WIKI:" in tools
    assert "TIME:" in tools


def test_execute_tool_unknown(tool_executor):
    result = tool_executor.execute_tool("UNKNOWN: test")
    assert result == "Unknown tool"


def test_execute_calc(tool_executor):
    result = tool_executor._execute_calc("CALC: 2 + 3")
    assert result == "Calculation result: 5"
    result = tool_executor._execute_calc("CALC: sqrt(4)")
    assert result == "Calculation result: 2.0"
    result = tool_executor._execute_calc("CALC: invalid")
    assert "Invalid calculation" in result


@patch("src.rag.tools.requests.Session.get")
def test_execute_wiki(mock_get, tool_executor):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"extract": "Test summary"}
    mock_get.return_value = mock_response

    result = tool_executor._execute_wiki("WIKI: Test Topic")
    assert "Wikipedia summary for 'Test Topic'" in result
    assert "Test summary" in result


@patch("src.rag.tools.requests.Session.get")
def test_execute_wiki_not_found(mock_get, tool_executor):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = tool_executor._execute_wiki("WIKI: Nonexistent")
    assert "No Wikipedia page found" in result


def test_execute_time(tool_executor):
    result = tool_executor._execute_time("TIME:")
    assert "Current date and time:" in result
