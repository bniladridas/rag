"""
Minimal TUI - Application-style terminal interface
"""

from __future__ import annotations

import contextlib
import difflib
import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .. import __version__
from ..review import (
    ReviewReport,
    build_open_report,
    build_review_report,
    handle_thread_command,
)

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
        self.show_footer_hint = True
        self.last_review_report: Optional[ReviewReport] = None
        self.last_review_index: int = 0
        self.last_review_command: Optional[str] = None
        self.live_review_enabled = False
        self.last_review_path: Optional[Path] = None
        self.last_review_mtime: Optional[float] = None

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
            "?",
            "commands",
            "info",
            "clear",
            "exit",
            "backends",
            "models",
            "theme",
            "shortcuts",
            "shortcuts on",
            "shortcuts off",
            "review staged",
            "review current",
            "review diff",
            "review staged",
            "open",
            "live review on",
            "live review off",
            "threads",
            "thread add",
            "thread reply",
            "thread resolve",
            "next finding",
            "prev finding",
            "ollama start",
            "ollama stop",
        ]

    def _maybe_handle_command_typo(self, query: str) -> bool:
        query_lower = query.lower()
        if query_lower in self._known_commands():
            return False
        match = difflib.get_close_matches(
            query_lower, self._known_commands(), n=1, cutoff=0.6
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

    def _user_card(self, body: str) -> Panel:
        return Panel(
            body,
            title="You",
            title_align="left",
            width=self._content_width(),
            box=ROUNDED,
            border_style="dim" if self.theme != "minimal" else "",
            padding=(0, 1),
        )

    def _assistant_card(self, body: str) -> Panel:
        subtitle = Text(
            self._status_line(),
            style="" if self.theme == "minimal" else "dim",
        )

        if (
            body.startswith("diff --git")
            or body.startswith("@@ ")
            or "+++ " in body
            or "--- " in body
        ):
            syntax = Syntax(body, "diff", theme="monokai", line_numbers=False)
            content = Group(subtitle, Text(""), syntax)
        else:
            content = Group(subtitle, Text(""), Text(body))

        return Panel(
            content,
            width=self._content_width(),
            box=ROUNDED,
            border_style=self._panel_style(),
            padding=(0, 1),
        )

    def _review_card(self, report: ReviewReport) -> Panel:
        subtitle = Text(
            f"{report.label}  |  {report.summary}",
            style="" if self.theme == "minimal" else "dim",
        )
        excerpt = self._review_excerpt_text(report)
        content = Group(subtitle, Text(""), excerpt)
        return Panel(
            content,
            width=self._content_width(),
            box=ROUNDED,
            border_style=self._panel_style(),
            padding=(0, 1),
        )

    def _review_session_body(self) -> str:
        if not self.last_review_command:
            return ""
        lines = [f"Active: {self.last_review_command}"]
        lines.append(f"Live review: {'on' if self.live_review_enabled else 'off'}")
        if self.last_review_report and self.last_review_report.findings:
            current = self.last_review_index + 1
            total = len(self.last_review_report.findings)
            lines.append(f"Findings: {current}/{total}")
            lines.append(
                "Commands: review current, next finding, prev finding, threads"
            )
        else:
            lines.append("Commands: review current, open <path[:line]>, threads")
        return "\n".join(lines)

    def _review_session_card(self) -> Optional[Panel]:
        body = self._review_session_body()
        if not body:
            return None
        return self._card("Review Session", body, width=self._content_width())

    def _set_review_state(self, report: ReviewReport) -> None:
        self.last_review_report = report
        self.last_review_index = 0
        self._update_review_file_watch(report)

    def _remember_review_command(self, command: str) -> None:
        self.last_review_command = command

    def _update_review_file_watch(self, report: ReviewReport) -> None:
        if not self.rag_engine:
            return
        target = report.label.split("  | ", 1)[0].split(":", 1)[0]
        self.last_review_path = (self.rag_engine.config.PROJECT_ROOT / target).resolve()
        try:
            self.last_review_mtime = self.last_review_path.stat().st_mtime
        except Exception:
            self.last_review_mtime = None

    def _toggle_live_review(self, enabled: bool) -> None:
        self.live_review_enabled = enabled
        self._print_message(
            "Live Review",
            "enabled." if enabled else "disabled.",
            "success",
        )

    def _maybe_refresh_live_review(self) -> None:
        if (
            not self.live_review_enabled
            or not self.last_review_command
            or not self.last_review_path
            or not self.rag_engine
        ):
            return
        try:
            current_mtime = self.last_review_path.stat().st_mtime
        except Exception:
            return
        if (
            self.last_review_mtime is not None
            and current_mtime <= self.last_review_mtime
        ):
            return
        self.last_review_mtime = current_mtime
        report = None
        if self.last_review_command.startswith("review "):
            report = build_review_report(
                self.last_review_command, self.rag_engine.config.PROJECT_ROOT
            )
            if report is not None:
                self._set_review_state(report)
        elif self.last_review_command.startswith("open "):
            report = build_open_report(
                self.last_review_command, self.rag_engine.config.PROJECT_ROOT
            )
            if report is not None:
                self.last_review_report = report
                self.last_review_index = 0
                self._update_review_file_watch(report)
        if report is not None:
            self.console.print()
            self.console.print(self._review_card(report))
            session_card = self._review_session_card()
            if session_card is not None:
                self.console.print(session_card)

    def _finding_focus_report(self, step: int) -> Optional[ReviewReport]:
        report = self.last_review_report
        if report is None or not report.findings:
            return None
        self.last_review_index = (self.last_review_index + step) % len(report.findings)
        finding = report.findings[self.last_review_index]
        focused_lines = tuple(
            (line_no, content)
            for line_no, content in report.source_lines
            if finding.line - 2 <= line_no <= finding.line + 2
        )
        if not focused_lines:
            focused_lines = report.source_lines
        return ReviewReport(
            mode=report.mode,
            label=f"{report.label}  |  finding {self.last_review_index + 1}/{len(report.findings)}",
            findings=(finding,),
            source_lines=focused_lines,
            summary=f"{finding.severity} finding at line {finding.line}",
        )

    def _review_excerpt_text(self, report: ReviewReport) -> Syntax | Text:
        if not report.source_lines:
            return Text()

        gutter_width = max(
            (len(str(line_no)) for line_no, _ in report.source_lines), default=2
        )

        code_lines = []
        for line_no, content in report.source_lines:
            code_lines.append(f"{line_no:>{gutter_width}} | {content}")

        return Syntax(
            "\n".join(code_lines),
            "python",
            theme="monokai",
            line_numbers=True,
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

    def _erase_last_input_line(self) -> None:
        # Move to the submitted prompt line, clear it, and return to column 0.
        self.console.file.write("\x1b[1A\x1b[2K\r")
        self.console.file.flush()

    def draw_header(self) -> None:
        return

    def draw_content(self) -> None:
        card = self._review_session_card()
        if card is not None:
            self.console.print(card)

    def draw_footer(self) -> None:
        if not self.show_footer_hint:
            return
        self.console.print(
            Align.left(
                Text("? / commands", style="dim"),
                width=self._content_width(),
            )
        )
        self.show_footer_hint = False

    def run_loop(self) -> None:  # noqa: C901
        while self.running:
            try:
                self._maybe_refresh_live_review()
                query = self.console.input(self._prompt_text()).strip()
                if not query:
                    continue
                self._erase_last_input_line()

                self.history.append(query)

                # Display user query
                self.console.print()
                self.console.print(self._user_card(query))

                if query.lower() in ["exit", "quit", "q"]:
                    self.running = False
                    self._print_message("Session", "Goodbye!", "success")
                    break

                if query.lower() == "clear":
                    self.clear_screen()
                    self.show_footer_hint = True
                    self._render_home()
                    continue

                if query.lower() == "?":
                    self.show_help()
                    continue

                if query.lower() == "commands":
                    self.show_commands()
                    continue

                if query.lower() in {"review diff", "review staged", "review current"}:
                    from ..review import review_command

                    if self.rag_engine is None:
                        continue
                    cmd = query.lower()
                    result = review_command(cmd, self.rag_engine.config.PROJECT_ROOT)
                    self.console.print()
                    self.console.print(self._assistant_card(result))
                    continue

                if query.lower() == "review current":
                    if not self.last_review_command:
                        self._print_message(
                            "Review",
                            "Run `review <path>` or `open <path>` first.",
                            "warning",
                        )
                        continue
                    query = self.last_review_command

                if query.lower() in {"live review on", "live review off"}:
                    self._toggle_live_review(query.lower().endswith("on"))
                    continue

                if query.lower() == "info":
                    self.show_info()
                    continue

                if query.lower() in {"next finding", "prev finding"}:
                    step = 1 if query.lower() == "next finding" else -1
                    report = self._finding_focus_report(step)
                    if report is None:
                        self._print_message(
                            "Review",
                            "Run `review <path>` first to navigate findings.",
                            "warning",
                        )
                    else:
                        self.console.print()
                        self.console.print(self._review_card(report))
                        session_card = self._review_session_card()
                        if session_card is not None:
                            self.console.print(session_card)
                    continue

                if query.lower() == "backends":
                    self.show_backends()
                    continue

                if query.lower() == "theme":
                    self.theme = "minimal" if self.theme == "default" else "default"
                    self._refresh_console()
                    self.clear_screen()
                    self.show_footer_hint = True
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

                if query.lower() == "threads" or query.lower().startswith("thread "):
                    if not self.rag_engine:
                        self._print_message(
                            "Error", "RAG engine not initialized", "error"
                        )
                        continue
                    response = handle_thread_command(
                        query, self.rag_engine.config.PROJECT_ROOT
                    )
                    self.console.print()
                    self.console.print(self._assistant_card(response))
                    continue

                if query.lower().startswith("backend:"):
                    backend = query.split(":", 1)[-1].strip()
                    self.switch_backend(backend)
                    continue

                if query.lower().startswith("model:"):
                    model = query.split(":", 1)[-1].strip()
                    self.switch_model(model)
                    continue

                if query.lower().startswith("open "):
                    if self.rag_engine:
                        report = build_open_report(
                            query, self.rag_engine.config.PROJECT_ROOT
                        )
                        if report is None:
                            self._print_message(
                                "Open",
                                "Use `open <path[:line]>` for a readable text file inside the repo.",
                                "error",
                            )
                        else:
                            self._remember_review_command(query)
                            self.last_review_report = report
                            self.last_review_index = 0
                            self._update_review_file_watch(report)
                            self.console.print()
                            self.console.print(self._review_card(report))
                            session_card = self._review_session_card()
                            if session_card is not None:
                                self.console.print(session_card)
                    else:
                        self._print_message(
                            "Error", "RAG engine not initialized", "error"
                        )
                    continue

                if self._maybe_handle_command_typo(query):
                    continue

                # Process actual query through RAG engine
                if self.rag_engine:
                    try:
                        if query.lower().startswith("review "):
                            report = build_review_report(
                                query, self.rag_engine.config.PROJECT_ROOT
                            )
                            if report is not None:
                                self._remember_review_command(query)
                                self._set_review_state(report)
                                self.console.print()
                                self.console.print(self._review_card(report))
                                session_card = self._review_session_card()
                                if session_card is not None:
                                    self.console.print(session_card)
                                continue
                        # Show loading message while generating response
                        self.console.print("Generating response...")
                        response = self._generate_response_quietly(query)
                        response_text = str(response).strip() or "(empty response)"
                        self.console.print()
                        self.console.print(self._assistant_card(response_text))
                    except Exception as e:
                        self._print_message("Error", str(e), "error")
                else:
                    self._print_message("Error", "RAG engine not initialized", "error")

            except (KeyboardInterrupt, EOFError):
                self.console.print("")
                self._print_message("Session", "Goodbye!", "success")
                break

    def show_help(self) -> None:
        help_text = (
            "Ask anything about ML, sci-fi, or the cosmos.\n\n"
            "Try:\n"
            "  review staged\n"
            "  open file:10\n"
            "  thread add file:42 comment\n"
            "  CALC: 2+2 | WIKI: ML | SHELL: git status\n\n"
            f"Theme: {self.theme} (toggle: theme)\n"
            f"Shortcuts: {'on' if getattr(self.rag_engine, 'shortcut_responses_enabled', False) else 'off'} (toggle: shortcuts on/off)"
        )
        self.console.print(self._card("Help", help_text, width=self._content_width()))

    def show_commands(self) -> None:
        body = (
            "info, clear, exit\n"
            "backends, backend:name\n"
            "models, model:name\n"
            "shortcuts, shortcuts on/off\n"
            "review diff, review staged, review file:10\n"
            "open file:10, next finding, prev finding\n"
            "live review on, live review off\n"
            "threads, thread add/reply/resolve\n"
            "ollama start, ollama stop\n"
            "theme"
        )
        self.console.print(self._card("Commands", body, width=self._content_width()))

    def show_info(self) -> None:
        body = "\n".join(
            [
                f"Version: {__version__}",
                f"Theme: {self.theme} (toggle: theme)",
                "Backends: local, openai, cerebras, ollama",
                "Tools: CALC:, TIME:, WIKI:, SHELL:, SEARCH:, WEB:",
                "Review: review diff, review staged, or review path[:line[-line]]",
                "Navigation: open path[:line], next finding, prev finding",
                "Live: live review on|off",
                "Threads: threads, thread add <path:line> <comment>",
                "Shortcuts: shortcuts on|off",
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
            enabled = getattr(self.rag_engine, "shortcut_responses_enabled", False)
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
            try:
                if self.rag_engine:
                    self.console.print("Generating response...")
                    response = self._generate_response_quietly(self.initial_query)
                    response_text = str(response).strip() or "(empty response)"
                    self.console.print()
                    self.console.print(self._user_card(self.initial_query))
                    self.console.print()
                    self.console.print(self._assistant_card(response_text))
            except Exception as e:
                self._print_message("Error", str(e), "error")
        self.run_loop()


def run_minimal_tui(initial_query: str = "", theme: str = "default") -> None:
    tui = MinimalTUI(theme=theme, initial_query=initial_query)
    tui.run()


def main(args: Optional[list] = None) -> None:
    """Entry point for rag-agent with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(prog="rag-agent")
    parser.add_argument("--theme", choices=["default", "minimal"], default="default")
    parser.add_argument("initial_query", nargs="*", default=[""])
    parsed = parser.parse_args(args or [])
    query = " ".join(parsed.initial_query).strip()
    run_minimal_tui(initial_query=query, theme=parsed.theme)


if __name__ == "__main__":
    main()
