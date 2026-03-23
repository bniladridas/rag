"""
Tool definitions for the RAG agent
"""

import ast
import subprocess
from typing import Callable
import math
import os
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup
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
        tools = """Available tools:
CALC: Calculate a mathematical expression (e.g., CALC: 2 + 3 * 4)
WIKI: Search Wikipedia for information (e.g., WIKI: Machine Learning)
TIME: Get current date and time
SHELL: Execute a shell command and return output (e.g., SHELL: git status)"""
        if self.config.ENABLE_WEB:
            tools += "\nSEARCH: Search the web (e.g., SEARCH: latest LLM news)"
            tools += "\nWEB: Fetch a URL and extract readable text (e.g., WEB: https://example.com)"
        return tools

    def execute_tool(self, tool_call: str) -> str:
        """Execute a tool based on the tool call string"""
        tool_call_upper = tool_call.upper()
        if tool_call_upper.startswith("CALC:"):
            return self._execute_calc(tool_call)
        elif tool_call_upper.startswith("WIKI:"):
            return self._execute_wiki(tool_call)
        elif tool_call_upper.startswith("TIME:"):
            return self._execute_time(tool_call)
        elif tool_call_upper.startswith("SEARCH:"):
            return self._execute_search(tool_call)
        elif tool_call_upper.startswith("WEB:"):
            return self._execute_web(tool_call)
        elif tool_call_upper.startswith("SHELL:"):
            return self._execute_shell(tool_call)
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

    def _execute_search(self, tool_call: str) -> str:
        """Search the web for a query (best-effort; may require API key)."""
        if not self.config.ENABLE_WEB:
            return "Web tools are disabled. Set RAG_ENABLE_WEB=1 to enable SEARCH/WEB."

        query = tool_call[7:].strip()
        if not query:
            return "SEARCH tool requires a query, e.g. `SEARCH: transformers 4.57.3`."

        provider = (self.config.SEARCH_PROVIDER or "duckduckgo").lower()
        if provider == "brave":
            api_key = os.getenv("BRAVE_API_KEY", "")
            if not api_key:
                return "BRAVE search requires BRAVE_API_KEY. Set RAG_SEARCH_PROVIDER=duckduckgo for no-key search."
            return self._search_brave(query, api_key)
        if provider == "serper":
            api_key = os.getenv("SERPER_API_KEY", "")
            if not api_key:
                return "Serper search requires SERPER_API_KEY. Set RAG_SEARCH_PROVIDER=duckduckgo for no-key search."
            return self._search_serper(query, api_key)

        # Default: DuckDuckGo HTML (no API key; may be rate-limited).
        return self._search_duckduckgo(query)

    def _search_duckduckgo(self, query: str) -> str:
        try:
            url = "https://duckduckgo.com/html/"
            resp = self.session.get(url, params={"q": query}, timeout=10)
            if resp.status_code != 200:
                return f"Search failed with status {resp.status_code}."
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:5]:
                title = a.get_text(" ", strip=True)
                href = a.get("href", "")
                if title and href:
                    results.append((title, href))
            if not results:
                return "No search results found."
            lines = ["Top results:"]
            for i, (title, href) in enumerate(results, start=1):
                lines.append(f"{i}. {title} — {href}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching the web: {e}"

    def _search_brave(self, query: str, api_key: str) -> str:
        try:
            resp = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": "5"},
                headers={"X-Subscription-Token": api_key},
                timeout=10,
            )
            if resp.status_code != 200:
                return f"Brave search failed with status {resp.status_code}."
            data = resp.json()
            web = data.get("web", {}).get("results", [])
            lines = ["Top results:"]
            for i, r in enumerate(web[:5], start=1):
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"{i}. {title} — {url}\n   {desc}".rstrip())
            return "\n".join(lines) if len(lines) > 1 else "No search results found."
        except Exception as e:
            return f"Error searching with Brave: {e}"

    def _search_serper(self, query: str, api_key: str) -> str:
        try:
            resp = self.session.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
                timeout=10,
            )
            if resp.status_code != 200:
                return f"Serper search failed with status {resp.status_code}."
            data = resp.json()
            organic = data.get("organic", [])
            lines = ["Top results:"]
            for i, r in enumerate(organic[:5], start=1):
                title = r.get("title", "")
                link = r.get("link", "")
                snippet = r.get("snippet", "")
                lines.append(f"{i}. {title} — {link}\n   {snippet}".rstrip())
            return "\n".join(lines) if len(lines) > 1 else "No search results found."
        except Exception as e:
            return f"Error searching with Serper: {e}"

    def _execute_web(self, tool_call: str) -> str:
        """Fetch a URL and extract readable text."""
        if not self.config.ENABLE_WEB:
            return "Web tools are disabled. Set RAG_ENABLE_WEB=1 to enable SEARCH/WEB."

        url = tool_call[4:].strip()
        if not url:
            return "WEB tool requires a URL, e.g. `WEB: https://example.com`."
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            return "Only http(s) URLs are allowed."

        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return f"Fetch failed with status {resp.status_code}."

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            text = soup.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return "No readable text found on the page."

            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "…"

            if title:
                return f"Page title: {title}\n\n{text}"
            return text
        except Exception as e:
            return f"Error fetching URL: {e}"

    def _execute_shell(self, tool_call: str) -> str:
        """Execute a shell command safely."""
        command = tool_call[6:].strip()
        if not command:
            return "SHELL tool requires a command, e.g. `SHELL: git status`."
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip() if result.stdout else ""
            if not output:
                output = result.stderr.strip() if result.stderr else ""
            if not output:
                output = "(command completed with no output)"
            return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {e}"


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
