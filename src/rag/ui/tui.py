"""
Text User Interface for RAG Transformer
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..__version__ import __version__
from ..rag_engine import RAGEngine


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        return


def create_tui_parser() -> argparse.ArgumentParser:
    """Create argument parser for TUI mode"""
    parser = argparse.ArgumentParser(
        prog="rag-tui",
        description="RAG Transformer Text User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The TUI provides a rich terminal interface with enhanced formatting
and interactive features for the RAG Transformer assistant.

Examples:
  rag-tui                Start TUI mode
  rag-tui --no-color     Start TUI without colors
  rag-tui --help         Show this help message
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force TUI mode even in non-interactive environments",
    )

    parser.add_argument(
        "--theme",
        choices=["default", "minimal"],
        default="default",
        help="Theme: default (colored) or minimal (app-style)",
    )

    parser.add_argument(
        "initial_query",
        nargs="*",
        help="Optional initial query to run when the TUI starts",
    )

    return parser


def _display_welcome(console: Console, no_color: bool, theme: str = "default") -> None:
    """Display welcome message and instructions."""
    content_width = 92
    border_style = "white" if no_color or theme == "minimal" else "bright_black"
    heading = (
        Text("Run the interactive TUI:", style="bold")
        if not no_color and theme != "minimal"
        else Text("Run the interactive TUI:")
    )
    dots = Text("o o o", style="bright_black" if border_style != "white" else "")
    command = Text()
    command.append("$ ", style="bold white" if border_style != "white" else "")
    command.append("rag-tui", style="bold")
    command_hint = Text(
        "Commands: help, info, clear, backends, backend, backend:<name>, "
        "models, model, model:<name>, shortcuts [on|off], refresh, cache:clear, "
        "memory:clear, hf:clear, update",
        style="" if border_style == "white" else "dim",
    )

    console.print()
    console.print(Align.left(heading, width=content_width))
    console.print()
    console.print(
        Align.center(
            Panel(
                Align.left(Text.assemble(dots, "\n\n", command)),
                box=ROUNDED,
                border_style=border_style,
                padding=(0, 1),
                width=content_width,
            ),
            width=content_width,
        )
    )
    console.print()
    console.print(Align.left(command_hint, width=content_width))
    console.print("")


def _display_model_status(
    console: Console, rag_engine: RAGEngine, no_color: bool, theme: str = "default"
) -> None:
    """Display warnings for missing models."""
    status = rag_engine.get_status()
    if not status.get("embedding_model_loaded"):
        msg = "Warning: embedding model not loaded. Retrieval will be limited."
        console.print(
            msg if (no_color or theme == "minimal") else f"[yellow]{msg}[/yellow]"
        )
    if not status.get("generator_model_loaded"):
        msg = "Warning: generator model not loaded. Responses will be limited."
        console.print(
            msg if (no_color or theme == "minimal") else f"[yellow]{msg}[/yellow]"
        )


def _display_info(console: Console, theme: str = "default") -> None:
    """Display info panel like the web installation section."""
    content_width = 92
    card_width = 44
    monochrome = theme == "minimal"
    border_style = "white" if monochrome else "bright_black"
    intro = Text(
        "rag works out of the box with minimal configuration. "
        "Choose quick setup or manual installation.",
        style="" if monochrome else "dim",
    )
    quick_install = Panel(
        "bash SETUP.sh",
        title="Quick Install",
        title_align="left",
        width=card_width,
        box=ROUNDED,
        border_style=border_style,
        padding=(1, 1),
    )
    manual_install = Panel(
        "\n".join(
            [
                "python3 -m venv venv",
                "source venv/bin/activate",
                "pip install -r requirements.txt",
                "pip install -e .",
            ]
        ),
        title="Manual",
        title_align="left",
        width=card_width,
        box=ROUNDED,
        border_style=border_style,
        padding=(1, 1),
    )
    runtime = Panel(
        "\n".join(
            [
                f"Version: {__version__}",
                "Backends: local, openai, cerebras, ollama",
                "Tools: CALC:, TIME:, WIKI:, SHELL:, SEARCH:, WEB:",
            ]
        ),
        title="Runtime",
        title_align="left",
        width=content_width,
        box=ROUNDED,
        border_style=border_style,
        padding=(1, 1),
    )

    console.print()
    console.print(Align.left(intro, width=content_width))
    console.print()
    console.print(
        Align.center(
            Columns([quick_install, manual_install], expand=False, equal=True),
            width=content_width,
        )
    )
    console.print()
    console.print(Align.center(runtime, width=content_width))


def _display_help(console: Console, no_color: bool) -> None:
    """Display help message with available commands and features."""
    if no_color:
        help_text = (
            "RAG Transformer Help\n\n"
            "• Ask about Machine Learning, AI, and Data Science\n"
            "• Inquire about Science Fiction movies and plots\n"
            "• Explore Cosmos, astronomy, and space science\n\n"
            "Built-in Tools:\n"
            "• CALC: <expression>  (e.g., 'CALC: 2^10')\n"
            "• WIKI: <topic>       (e.g., 'WIKI: Quantum Computing')\n"
            "• TIME:               (current date and time)\n\n"
            "Optional Web Tools (enable with RAG_ENABLE_WEB=1):\n"
            "• SEARCH: <query>     (e.g., 'SEARCH: latest LLM news')\n"
            "• WEB: <url>          (e.g., 'WEB: https://example.com')\n\n"
            "LLM Backends (select with RAG_LLM_BACKEND):\n"
            "• local   (default)\n"
            "• openai   (OPENAI_API_KEY + OPENAI_MODEL)\n"
            "• cerebras (CEREBRAS_API_KEY + CEREBRAS_MODEL)\n"
            "• ollama   (OLLAMA_BASE_URL + OLLAMA_MODEL)\n\n"
            "Commands: 'exit'/'quit'/'q' to quit, 'help'/'h' for this message"
        )
        console.print(Panel(help_text, title="Help"))
    else:
        help_text = (
            "[bold]📚 RAG Transformer Help[/]\n\n"
            "• Ask about [blue]Machine Learning[/], AI, and Data Science\n"
            "• Inquire about [magenta]Science Fiction[/] movies and plots\n"
            "• Explore [green]Cosmos[/], astronomy, and space science\n\n"
            "[bold]Built-in Tools:[/]\n"
            "• [cyan]CALC:[/] <expression>  (e.g., 'CALC: 2^10')\n"
            "• [cyan]WIKI:[/] <topic>       (e.g., 'WIKI: Quantum Computing')\n"
            "• [cyan]TIME:[/]               (current date and time)\n\n"
            "[bold]Optional Web Tools[/] (enable with `RAG_ENABLE_WEB=1`):\n"
            "• [cyan]SEARCH:[/] <query>     (e.g., 'SEARCH: latest LLM news')\n"
            "• [cyan]WEB:[/] <url>          (e.g., 'WEB: https://example.com')\n\n"
            "[bold]LLM Backends[/] (select with `RAG_LLM_BACKEND`):\n"
            "• [cyan]local[/]   (default)\n"
            "• [cyan]openai[/]   (`OPENAI_API_KEY` + `OPENAI_MODEL`)\n"
            "• [cyan]cerebras[/] (`CEREBRAS_API_KEY` + `CEREBRAS_MODEL`)\n"
            "• [cyan]ollama[/]   (`OLLAMA_BASE_URL` + `OLLAMA_MODEL`)\n\n"
            "[bold]Commands:[/] 'exit'/'quit'/'q' to quit, 'help'/'h' for this message"
        )
        console.print(Panel(help_text, title="[blue]Help[/]", border_style="blue"))

    console.print(
        Panel(
            "MODEL: <name>   Switch active model for current backend\n"
            "MODELS          Show current backend/model\n"
            "BACKEND: <name> Switch backend (local|openai|cerebras|ollama)\n"
            "BACKENDS        Show available backends\n"
            "BACKEND         Pick backend from a list\n"
            "SHORTCUTS       Show deterministic shortcut reply status\n"
            "SHORTCUTS ON    Enable deterministic shortcut replies\n"
            "SHORTCUTS OFF   Disable deterministic shortcut replies\n"
            "REFRESH         Reload engine/session\n"
            "CACHE:CLEAR     Delete project .cache\n"
            "MEMORY:CLEAR    Clear conversation memory\n"
            "HF:CLEAR        Delete Hugging Face cache (~/.cache/huggingface)\n"
            "UPDATE          Show update commands",
            title="Models" if no_color else "[blue]Models[/]",
            border_style="white" if no_color else "blue",
        )
    )


def _process_query(
    rag_engine: RAGEngine,
    query: str,
    console: Console,
    no_color: bool,
    theme: str = "default",
) -> None:
    """Process a single query and display the response."""
    if no_color or theme == "minimal":
        with console.status("Processing your query..."):
            response = rag_engine.generate_response(query)
        console.print(Panel(response, title="Response"))
    else:
        with console.status("[bold green]Processing your query...[/]"):
            response = rag_engine.generate_response(query)
        console.print(
            Panel(
                Markdown(response), title="[bold]💡 Response[/]", border_style="green"
            )
        )


def _handle_exit(console: Console, no_color: bool) -> None:
    """Handle exit message display."""
    if no_color:
        console.print("\nGoodbye!")
    else:
        console.print("\n[yellow]👋 Goodbye![/yellow]")


def _pick_from_list(
    console: Console, title: str, options: list[str], no_color: bool
) -> Optional[str]:
    if not options:
        return None

    page_size = 10
    page = 0
    while True:
        start = page * page_size
        end = start + page_size
        chunk = options[start:end]
        if not chunk:
            page = 0
            continue

        header = f"{title} (page {page + 1}/{(len(options) - 1) // page_size + 1})"
        lines = [header, ""]
        for idx, item in enumerate(chunk, start=start + 1):
            lines.append(f"{idx}. {item}")
        lines.append("")
        lines.append("Choose number, 'n' next, 'p' prev, or 'q' cancel.")
        console.print(
            Panel("\n".join(lines), title="Select" if no_color else "[blue]Select[/]")
        )

        choice = console.input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice in {"n", "next"}:
            if end < len(options):
                page += 1
            continue
        if choice in {"p", "prev", "previous"}:
            if page > 0:
                page -= 1
            continue
        if choice.isdigit():
            pick = int(choice)
            if 1 <= pick <= len(options):
                return options[pick - 1]
            console.print(
                "Invalid selection."
                if no_color
                else "[yellow]Invalid selection.[/yellow]"
            )
            continue
        console.print(
            "Invalid input." if no_color else "[yellow]Invalid input.[/yellow]"
        )


def _resolve_hf_cache_dir() -> Path:
    # Prefer explicit cache env vars when provided.
    explicit = (
        os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
        or os.getenv("HF_HOME")
    )
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/huggingface").expanduser()


def run_tui(  # noqa: C901
    no_color: bool = False,
    force: bool = False,
    theme: str = "default",
    initial_query: Optional[str] = None,
) -> None:
    """Run the Text User Interface with CLI policy compliance.

    Args:
        no_color: If True, disable colored output
        force: If True, force TUI mode even in non-interactive environments
        theme: Theme to use - 'default' (colored) or 'minimal' (monochrome)
        initial_query: Optional query to process before entering the input loop
    """
    # Skip TUI in non-interactive environments unless forced
    if not sys.stdin.isatty() and not force and not os.getenv("FORCE_TUI"):
        print("RAG Transformer TUI - Text User Interface")
        print("Non-interactive environment detected. Use --force to override.")
        print("For non-interactive usage, try: rag --query 'your question'")
        return

    _load_env_file()
    console = Console(force_terminal=force, no_color=no_color)

    # Theme flag for internal use
    if theme == "minimal":
        no_color = True

    try:
        rag_engine = RAGEngine()
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG engine: {e}[/red]")
        sys.exit(1)

    _display_welcome(console, no_color, theme)
    _display_model_status(console, rag_engine, no_color, theme)

    if initial_query:
        _process_query(rag_engine, initial_query, console, no_color, theme)

    while True:
        try:
            prompt = (
                f"> {rag_engine.current_backend_and_model()} "
                if theme == "minimal"
                else f"[cyan]❯[/] [dim]{rag_engine.current_backend_and_model()}[/] "
            )
            query = console.input(prompt).strip()

            # Handle commands
            if query.lower() in ["exit", "quit", "q"]:
                _handle_exit(console, no_color)
                break

            if query.lower() in ["help", "h"]:
                _display_help(console, no_color)
                continue

            if query.lower() == "clear":
                console.clear()
                continue

            if query.lower() == "theme":
                theme = "minimal" if theme == "default" else "default"
                console.print(f"Theme switched to: {theme}")
                no_color = theme == "minimal"
                continue

            if query.lower() == "info":
                _display_info(console, theme)
                continue

            if query.lower() == "update":
                msg = (
                    "To update to the latest software, run outside the TUI:\n"
                    "  git pull\n"
                    "  ./venv/bin/pip install -e .\n"
                    "Then restart rag-tui."
                )
                console.print(
                    Panel(msg, title="Update" if no_color else "[blue]Update[/]")
                )
                continue

            if query.lower() == "refresh":
                console.print(
                    "Refreshing session (reloading engine/models/index)..."
                    if no_color
                    else "[dim]Refreshing session (reloading engine/models/index)...[/]"
                )
                try:
                    rag_engine = RAGEngine()
                    _display_model_status(console, rag_engine, no_color)
                    console.print(
                        "Refreshed." if no_color else "[green]Refreshed.[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"Refresh failed: {e}"
                        if no_color
                        else f"[red]Refresh failed: {e}[/red]"
                    )
                continue

            if query.lower() == "shortcuts":
                enabled = getattr(rag_engine, "shortcut_responses_enabled", True)
                msg = f"Shortcut responses: {'on' if enabled else 'off'}"
                console.print(msg if no_color else f"[dim]{msg}[/]")
                continue

            if query.lower() in {"shortcuts on", "shortcuts off"}:
                enabled = query.lower().endswith("on")
                msg = rag_engine.set_shortcut_responses_enabled(enabled)
                console.print(msg if no_color else f"[green]{msg}[/green]")
                continue

            if query.lower() == "memory:clear":
                try:
                    rag_engine.memory.clear()
                    console.print(
                        "Memory cleared."
                        if no_color
                        else "[green]Memory cleared.[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"Failed to clear memory: {e}"
                        if no_color
                        else f"[red]Failed to clear memory: {e}[/red]"
                    )
                continue

            if query.lower() == "cache:clear":
                cache_dir: Path = rag_engine.config.CACHE_DIR
                project_root: Path = rag_engine.config.PROJECT_ROOT
                try:
                    resolved = cache_dir.resolve()
                    resolved.relative_to(project_root.resolve())
                except Exception:
                    console.print(
                        f"Refusing to delete cache outside project root: {cache_dir}"
                        if no_color
                        else f"[red]Refusing to delete cache outside project root: {cache_dir}[/red]"
                    )
                    continue

                confirm = (
                    console.input(f"Delete cache at {resolved}? (y/N) ").strip().lower()
                )
                if confirm != "y":
                    console.print("Canceled." if no_color else "[dim]Canceled.[/]")
                    continue

                try:
                    if resolved.exists():
                        shutil.rmtree(resolved)
                    resolved.mkdir(parents=True, exist_ok=True)
                    console.print(
                        "Cache cleared. Run `refresh` to reload the engine."
                        if no_color
                        else "[green]Cache cleared.[/green] Run [bold]refresh[/] to reload the engine."
                    )
                except Exception as e:
                    console.print(
                        f"Failed to clear cache: {e}"
                        if no_color
                        else f"[red]Failed to clear cache: {e}[/red]"
                    )
                continue

            if query.lower() == "hf:clear":
                hf_dir = _resolve_hf_cache_dir()
                try:
                    resolved = hf_dir.resolve()
                except Exception:
                    resolved = hf_dir

                home = Path.home()
                try:
                    home_resolved = home.resolve()
                except Exception:
                    home_resolved = home

                # Safety checks: refuse obviously dangerous deletes.
                try:
                    resolved.relative_to(home_resolved)
                except Exception:
                    console.print(
                        f"Refusing to delete Hugging Face cache outside your home directory: {resolved}"
                        if no_color
                        else f"[red]Refusing to delete Hugging Face cache outside your home directory: {resolved}[/red]"
                    )
                    continue
                if resolved == home_resolved or str(resolved) in {"/", ""}:
                    console.print(
                        f"Refusing to delete unsafe path: {resolved}"
                        if no_color
                        else f"[red]Refusing to delete unsafe path: {resolved}[/red]"
                    )
                    continue

                msg = (
                    f"This will delete Hugging Face cache at:\n  {resolved}\n\n"
                    "This forces models to re-download next time.\n"
                    "Type DELETE to confirm (or anything else to cancel):"
                )
                console.print(
                    Panel(msg, title="HF Cache" if no_color else "[blue]HF Cache[/]")
                )
                confirm = console.input("> ").strip()
                if confirm != "DELETE":
                    console.print("Canceled." if no_color else "[dim]Canceled.[/]")
                    continue

                try:
                    if resolved.exists():
                        shutil.rmtree(resolved)
                    console.print(
                        "Hugging Face cache cleared. Run `refresh` to reload the engine."
                        if no_color
                        else "[green]Hugging Face cache cleared.[/green] Run [bold]refresh[/] to reload the engine."
                    )
                except Exception as e:
                    console.print(
                        f"Failed to clear Hugging Face cache: {e}"
                        if no_color
                        else f"[red]Failed to clear Hugging Face cache: {e}[/red]"
                    )
                continue

            if query.lower() == "models":
                available = rag_engine.available_models()
                msg = f"Current: {rag_engine.current_backend_and_model()}"
                console.print(msg if no_color else f"[dim]{msg}[/]")
                if available:
                    console.print(
                        ("Available models: " + ", ".join(available))
                        if no_color
                        else "[dim]Available models: " + ", ".join(available) + "[/]"
                    )
                else:
                    hint = rag_engine.models_hint()
                    if hint:
                        console.print(hint if no_color else f"[dim]{hint}[/]")
                continue

            if query.lower() == "backends":
                msg = "Backends: " + ", ".join(rag_engine.available_backends())
                console.print(msg if no_color else f"[dim]{msg}[/]")
                continue

            if query.lower().startswith("backend:"):
                backend = query.split(":", 1)[-1].strip()
                msg = rag_engine.set_backend(backend)
                console.print(msg if no_color else f"[green]{msg}[/green]")
                continue

            if query.lower().startswith("ollama:"):
                query_stripped = query[7:].strip()
                query_lower = query_stripped.lower()

                if query_lower == "start":
                    msg = rag_engine.start_ollama_server()
                    console.print(msg if no_color else f"[green]{msg}[/green]")
                    continue

                if query_lower == "stop":
                    msg = rag_engine.stop_ollama_server()
                    console.print(msg if no_color else f"[green]{msg}[/green]")
                    continue

                if query_stripped:
                    parts = query_stripped.split(None, 1)
                    model_name = parts[0] if parts else ""
                    actual_query = parts[1] if len(parts) > 1 else ""
                    msg = rag_engine.set_backend("ollama")
                    if "Switched" in msg:
                        rag_engine.config.OLLAMA_MODEL = model_name
                        msg = f"Switched to ollama with model {model_name}."
                        console.print(msg if no_color else f"[green]{msg}[/green]")
                        if actual_query:
                            query = actual_query
                        else:
                            continue
                    else:
                        console.print(msg if no_color else f"[red]{msg}[/red]")
                        continue

            if query.lower() == "backend":
                options = rag_engine.available_backends()
                picked = _pick_from_list(console, "Backends", options, no_color)
                if picked:
                    msg = rag_engine.set_backend(picked)
                    console.print(msg if no_color else f"[green]{msg}[/green]")
                continue

            if query.lower().startswith("model:"):
                model_name = query.split(":", 1)[-1].strip()
                msg = rag_engine.set_active_model(model_name)
                console.print(msg if no_color else f"[green]{msg}[/green]")
                continue

            if query.lower() == "model":
                options = rag_engine.available_models()
                picked = _pick_from_list(
                    console,
                    f"Models for {rag_engine.current_backend_and_model().split(':', 1)[0]}",
                    options,
                    no_color,
                )
                if picked:
                    msg = rag_engine.set_active_model(picked)
                    console.print(msg if no_color else f"[green]{msg}[/green]")
                continue

            if not query:
                console.print(
                    "Please enter a valid query."
                    if no_color
                    else "[yellow]Please enter a valid query.[/yellow]"
                )
                continue

            _process_query(rag_engine, query, console, no_color)

        except (KeyboardInterrupt, EOFError):
            _handle_exit(console, no_color)
            break
        except Exception as e:
            console.print(
                Panel(
                    (
                        f"An error occurred: {e}"
                        if no_color
                        else f"[red]An error occurred: {e}[/red]"
                    ),
                    title="Error" if no_color else "❌ Error",
                    border_style="white" if no_color else "red",
                )
            )


def main(args: Optional[list] = None) -> None:
    """Main entry point for TUI with argument parsing"""
    parser = create_tui_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.theme == "minimal":
        from .minimal_tui import run_minimal_tui

        run_minimal_tui(initial_query=" ".join(parsed_args.initial_query).strip())
    else:
        run_tui(
            no_color=parsed_args.no_color,
            force=parsed_args.force,
            theme=parsed_args.theme,
            initial_query=" ".join(parsed_args.initial_query).strip(),
        )


if __name__ == "__main__":
    main()
