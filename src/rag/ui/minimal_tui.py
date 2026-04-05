"""
Minimal TUI - Application-style terminal interface
"""

import contextlib
import difflib
import io
import logging
import sys
from typing import TYPE_CHECKING, Optional

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .. import __version__

if TYPE_CHECKING:
    from ..rag_engine import RAGEngine


class MinimalTUI:
    def __init__(self, theme: str = "default", initial_query: str = ""):
        self.theme = theme
        self.initial_query = initial_query
        self.console = self._make_console()
        self.running = True
        self.history: list[str] = []
        self.rag_engine: Optional["RAGEngine"] = None

    def _make_console(self) -> Console:
        return Console(force_terminal=True, no_color=(self.theme == "minimal"))

    def _width(self) -> int:
        return max(72, min(self.console.size.width, 100))

    def _content_width(self) -> int:
        return max(60, min(self._width() - 6, 92))

    def _refresh_console(self) -> None:
        self.console = self._make_console()

    def _style(self, semantic: str) -> str:
        if self.theme == "minimal":
            return ""
        styles = {
            "title": "bold cyan",
            "muted": "dim",
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "bold red",
            "label": "bold white",
            "border": "cyan",
        }
        return styles.get(semantic, "")

    def _panel_style(self) -> str:
        return "white" if self.theme == "minimal" else "bright_black"

    def _status_line(self) -> str:
        if not self.rag_engine:
            return "engine unavailable"
        try:
            return self.rag_engine.current_backend_and_model()
        except Exception:
            return "backend/model unavailable"

    def _prompt_text(self) -> str:
        status = self._status_line()
        if len(status) > 28:
            status = status[:25] + "..."
        return f"[{status}] > "

    def _print_message(self, label: str, message: str, semantic: str = "info") -> None:
        style = self._style(semantic)
        prefix = f"{label}:"
        text = f"{prefix} {message}"
        self.console.print(text if not style else f"[{style}]{text}[/]")

    def _known_commands(self) -> list[str]:
        return [
            "help",
            "info",
            "clear",
            "exit",
            "backends",
            "models",
            "theme",
            "shortcuts",
            "shortcuts on",
            "shortcuts off",
            "ollama start",
            "ollama stop",
        ]

    def _maybe_handle_command_typo(self, query: str) -> bool:
        match = difflib.get_close_matches(
            query.lower(), self._known_commands(), n=1, cutoff=0.8
        )
        if not match:
            return False
        self._print_message(
            "Command",
            f"Unknown command `{query}`. Did you mean `{match[0]}`?",
            "warning",
        )
        return True

    def _generate_response_quietly(self, query: str) -> str:
        if not self.rag_engine:
            raise RuntimeError("RAG engine not initialized")

        root_logger = logging.getLogger()
        rag_logger = logging.getLogger("rag")
        rag_engine_logger = logging.getLogger("rag.rag_engine")
        previous_levels = {
            root_logger: root_logger.level,
            rag_logger: rag_logger.level,
            rag_engine_logger: rag_engine_logger.level,
        }

        capture = io.StringIO()
        try:
            root_logger.setLevel(logging.ERROR)
            rag_logger.setLevel(logging.ERROR)
            rag_engine_logger.setLevel(logging.ERROR)
            with (
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(capture),
            ):
                return self.rag_engine.generate_response(query)
        finally:
            for logger_obj, level in previous_levels.items():
                logger_obj.setLevel(level)

    def _card(self, title: str, body: str, width: Optional[int] = None) -> Panel:
        return Panel(
            body,
            title=title,
            title_align="left",
            width=width,
            box=ROUNDED,
            border_style=self._panel_style(),
            padding=(1, 1),
        )

    def _terminal_card(self, command: str) -> Panel:
        dots = Text("o o o", style="bright_black" if self.theme != "minimal" else "")
        body = Text()
        body.append("$ ", style="bold white" if self.theme != "minimal" else "")
        body.append(command, style="bold")
        return Panel(
            Align.left(Text.assemble(dots, "\n\n", body)),
            box=ROUNDED,
            border_style=self._panel_style(),
            padding=(0, 1),
            width=self._content_width(),
        )

    def _render_home(self) -> None:
        self.draw_header()
        self.draw_content()
        self.draw_footer()

    def init_engine(self) -> None:
        """Initialize the RAG engine."""
        try:
            root_logger = logging.getLogger()
            previous_level = root_logger.level
            capture = io.StringIO()
            with (
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(capture),
            ):
                root_logger.setLevel(logging.ERROR)
                try:
                    from ..rag_engine import RAGEngine

                    self.rag_engine = RAGEngine()
                finally:
                    root_logger.setLevel(previous_level)
        except Exception as e:
            self.console.print(f"Failed to initialize RAG engine: {e}")
            sys.exit(1)

    def clear_screen(self) -> None:
        self.console.clear()

    def draw_header(self) -> None:
        return

    def draw_content(self) -> None:
        return

    def draw_footer(self) -> None:
        self.console.print(
            Align.left(
                Text(
                    "Commands: help, info, clear, backends, models, "
                    "backend:<name>, model:<name>, shortcuts [on|off], "
                    "ollama start, ollama stop, theme",
                    style="dim",
                ),
                width=self._content_width(),
            )
        )

    def run_loop(self) -> None:  # noqa: C901
        while self.running:
            try:
                query = self.console.input(self._prompt_text()).strip()
                if not query:
                    continue

                self.history.append(query)

                if query.lower() in ["exit", "quit", "q"]:
                    self.running = False
                    self._print_message("Session", "Goodbye!", "success")
                    break

                if query.lower() == "clear":
                    self.clear_screen()
                    self._render_home()
                    continue

                if query.lower() == "help":
                    self.show_help()
                    continue

                if query.lower() == "info":
                    self.show_info()
                    continue

                if query.lower() == "backends":
                    self.show_backends()
                    continue

                if query.lower() == "theme":
                    self.theme = "minimal" if self.theme == "default" else "default"
                    self._refresh_console()
                    self.clear_screen()
                    self._render_home()
                    self._print_message("Theme", f"Switched to {self.theme}", "success")
                    continue

                if query.lower() == "models":
                    self.show_models()
                    continue

                if query.lower() == "shortcuts":
                    self.show_shortcuts_status()
                    continue

                if query.lower().startswith("shortcuts "):
                    mode = query.split(None, 1)[1].strip().lower()
                    self.set_shortcuts(mode)
                    continue

                if query.lower() == "ollama start":
                    self.start_ollama()
                    continue

                if query.lower() == "ollama stop":
                    self.stop_ollama()
                    continue

                if query.lower().startswith("backend:"):
                    backend = query.split(":", 1)[-1].strip()
                    self.switch_backend(backend)
                    continue

                if query.lower().startswith("model:"):
                    model = query.split(":", 1)[-1].strip()
                    self.switch_model(model)
                    continue

                if self._maybe_handle_command_typo(query):
                    continue

                # Process actual query through RAG engine
                if self.rag_engine:
                    self._print_message("You", query, "muted")
                    try:
                        self._print_message("Assistant", "Thinking...", "info")
                        with self.console.status("Thinking..."):
                            response = self._generate_response_quietly(query)
                        response_text = str(response).strip() or "(empty response)"
                        self.console.print()
                        self.console.print(
                            self._card(
                                "Assistant", response_text, width=self._content_width()
                            )
                        )
                    except Exception as e:
                        self._print_message("Error", str(e), "error")
                else:
                    self._print_message("Error", "RAG engine not initialized", "error")

            except (KeyboardInterrupt, EOFError):
                self.console.print("")
                self._print_message("Session", "Goodbye!", "success")
                break

    def show_help(self) -> None:
        help_text = "\n".join(
            [
                "Ask anything about ML, science fiction, or the cosmos.",
                "",
                "help, info, clear, exit: interface controls",
                "backends, backend:<name>: inspect or switch backend",
                "models, model:<name>: inspect or switch model",
                "shortcuts, shortcuts on, shortcuts off: control hardcoded replies",
                "ollama start, ollama stop: manage local Ollama server",
                "theme: toggle monochrome and accent styles",
                "",
                "Examples: WIKI: transformers | CALC: 2^10 | SHELL: git status",
            ]
        )
        self.console.print(self._card("Help", help_text, width=self._content_width()))

    def show_info(self) -> None:
        body = "\n".join(
            [
                f"Version: {__version__}",
                "Backends: local, openai, cerebras, ollama",
                "Tools: CALC:, TIME:, WIKI:, SHELL:, SEARCH:, WEB:",
                "Use `shortcuts on|off` to control deterministic shortcut replies.",
            ]
        )
        self.console.print()
        self.console.print(self._card("Info", body, width=self._content_width()))

    def show_backends(self) -> None:
        if self.rag_engine:
            backends = self.rag_engine.available_backends()
            current = self.rag_engine.current_backend_and_model()
            body = "\n".join(
                [
                    f"Current: {current}",
                    "Available: " + (", ".join(backends) if backends else "none"),
                ]
            )
            self.console.print(
                self._card("Backends", body, width=self._content_width())
            )
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def show_models(self) -> None:
        if self.rag_engine:
            models = self.rag_engine.available_models()
            current = self.rag_engine.current_backend_and_model()
            body = "\n".join(
                [
                    f"Current: {current}",
                    "Available: " + (", ".join(models) if models else "none"),
                ]
            )
            self.console.print(self._card("Models", body, width=self._content_width()))
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def show_shortcuts_status(self) -> None:
        if self.rag_engine:
            enabled = getattr(self.rag_engine, "shortcut_responses_enabled", True)
            state = "on" if enabled else "off"
            self.console.print(
                self._card(
                    "Shortcuts",
                    f"Deterministic shortcut replies are {state}.",
                    width=self._content_width(),
                )
            )
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def set_shortcuts(self, mode: str) -> None:
        if mode not in {"on", "off"}:
            self._print_message(
                "Error",
                "Use `shortcuts on` or `shortcuts off`.",
                "error",
            )
            return
        if self.rag_engine:
            msg = self.rag_engine.set_shortcut_responses_enabled(mode == "on")
            self._print_message("Shortcuts", msg, "success")
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def switch_backend(self, backend: str) -> None:
        if self.rag_engine:
            msg = self.rag_engine.set_backend(backend)
            current = self.rag_engine.current_backend_and_model()
            self.console.print(
                self._card(
                    "Backend",
                    f"{msg}\n\nNow using: {current}",
                    width=self._content_width(),
                )
            )
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def switch_model(self, model: str) -> None:
        if self.rag_engine:
            msg = self.rag_engine.set_active_model(model)
            current = self.rag_engine.current_backend_and_model()
            self.console.print(
                self._card(
                    "Model",
                    f"{msg}\n\nNow using: {current}",
                    width=self._content_width(),
                )
            )
        else:
            self._print_message("Error", "RAG engine not initialized", "error")

    def start_ollama(self) -> None:
        import subprocess
        import time

        self._print_message("Ollama", "Starting Ollama server...", "info")
        try:
            # Check if already running
            try:
                import requests

                r = requests.get("http://localhost:11434/api/tags", timeout=1)
                if r.status_code == 200:
                    self._print_message(
                        "Ollama",
                        "Already running on localhost:11434",
                        "success",
                    )
                    if self.rag_engine:
                        self.switch_backend("ollama")
                    return
            except Exception:
                pass

            # Start Ollama
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(2)
            self._print_message(
                "Ollama",
                "Server started on localhost:11434",
                "success",
            )
            if self.rag_engine:
                self.switch_backend("ollama")
        except FileNotFoundError:
            self._print_message(
                "Error",
                "Ollama not found. Install from ollama.ai",
                "error",
            )
        except Exception as e:
            self._print_message("Error", f"Failed to start Ollama: {e}", "error")

    def stop_ollama(self) -> None:
        import subprocess

        self._print_message("Ollama", "Stopping Ollama server...", "info")
        try:
            if sys.platform == "darwin":
                subprocess.run(
                    ["killall", "ollama"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", "ollama"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._print_message("Ollama", "Server stopped", "success")
        except Exception as e:
            self._print_message("Error", f"Failed to stop Ollama: {e}", "error")

    def run(self) -> None:
        self.init_engine()
        self.clear_screen()
        self._render_home()
        if self.initial_query:
            self.history.append(self.initial_query)
            self._print_message("You", self.initial_query, "muted")
            try:
                if self.rag_engine:
                    self._print_message("Assistant", "Thinking...", "info")
                    with self.console.status("Thinking..."):
                        response = self._generate_response_quietly(self.initial_query)
                    response_text = str(response).strip() or "(empty response)"
                    self.console.print()
                    self.console.print(
                        self._card(
                            "Assistant", response_text, width=self._content_width()
                        )
                    )
            except Exception as e:
                self._print_message("Error", str(e), "error")
        self.run_loop()


def run_minimal_tui(initial_query: str = "") -> None:
    tui = MinimalTUI(theme="minimal", initial_query=initial_query)
    tui.run()


if __name__ == "__main__":
    run_minimal_tui()
