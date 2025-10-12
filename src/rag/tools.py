"""
Tool definitions for the RAG agent
"""

import math
from datetime import datetime

import requests

from .config import Config


class ToolExecutor:
    """Handles execution of various tools"""

    def __init__(self):
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RAG-Transformer/1.0"})

    def get_available_tools(self) -> str:
        """Get description of available tools"""
        return """Available tools:
CALC: Calculate a mathematical expression (e.g., CALC: 2 + 3 * 4)
WIKI: Search Wikipedia for information (e.g., WIKI: Machine Learning)
TIME: Get current date and time"""

    def execute_tool(self, tool_call: str) -> str:
        """Execute a tool based on the tool call string"""
        tool_call_upper = tool_call.upper()
        if tool_call_upper.startswith("CALC:"):
            return self._execute_calc(tool_call)
        elif tool_call_upper.startswith("WIKI:"):
            return self._execute_wiki(tool_call)
        elif tool_call_upper.startswith("TIME:"):
            return self._execute_time(tool_call)
        else:
            return "Unknown tool"

    def _execute_calc(self, tool_call: str) -> str:
        """Execute calculator tool safely"""
        expr = tool_call[5:].strip()
        try:
            allowed_names = {
                "__builtins__": None,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expr, allowed_names)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Invalid calculation: {e}"

    def _execute_wiki(self, tool_call: str) -> str:
        """Execute Wikipedia search tool safely"""
        topic = tool_call[5:].strip().replace(" ", "_")
        try:
            # Reduce timeout in CI/Docker for faster failure
            response = self.session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "No summary available")
                return f"Wikipedia summary for '{topic.replace('_', ' ')}': {extract}"
            return f"No Wikipedia page found for '{topic.replace('_', ' ')}'"
        except Exception as e:
            return f"Error fetching Wikipedia: {e}"

    def _execute_time(self, tool_call: str) -> str:
        """Execute time tool"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current date and time: {current_time}"
