"""
Tool definitions for the RAG agent
"""

import ast
from typing import Callable
import math
import re
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config


class ToolExecutor:
    """Handles execution of various tools"""

    def __init__(self) -> None:
        self.config = Config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RAG-Transformer/1.0"})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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
            result = _safe_eval_math(expr)
            if re.search(r"[A-Za-z]", expr):
                result_str = str(result)
            elif float(result).is_integer():
                result_str = str(int(result))
            else:
                result_str = str(result)
            return f"Calculation result: {result_str}"
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


_ALLOWED_FUNCS: dict[str, Callable[[float], float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval_math(expr: str) -> float:
    """Safely evaluate a math expression using a restricted AST."""
    expr = expr.replace("^", "**")
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        ):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return float(left**right)
            if isinstance(n.op, ast.Mod):
                return left % right
        if isinstance(n, ast.Name):
            if n.id in _ALLOWED_CONSTS:
                return float(_ALLOWED_CONSTS[n.id])
            raise ValueError(f"Unknown identifier: {n.id}")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            func = _ALLOWED_FUNCS.get(n.func.id)
            if not func:
                raise ValueError(f"Function not allowed: {n.func.id}")
            if len(n.args) != 1:
                raise ValueError("Only single-argument functions are allowed")
            return float(func(_eval(n.args[0])))
        raise ValueError("Unsupported expression")

    return _eval(node)
