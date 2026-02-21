"""
Text User Interface for RAG Transformer
"""

import argparse
import os
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..__version__ import __version__
from ..rag_engine import RAGEngine


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

    return parser


def _display_welcome(console: Console, no_color: bool) -> None:
    """Display welcome message and instructions."""
    if no_color:
        console.print(
            Panel.fit(
                "Agentic RAG Transformer\n"
                "ML, Sci-Fi, and Cosmos Assistant\n"
                f"Version {__version__}"
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold blue]🤖 Agentic RAG Transformer[/]\n"
                "[green]ML, Sci-Fi, and Cosmos Assistant[/]\n"
                f"[dim]Version {__version__}[/]"
            )
        )
    console.print("Type 'exit'/'quit' to quit, 'help' for instructions.\n")


def _display_model_status(
    console: Console, rag_engine: RAGEngine, no_color: bool
) -> None:
    """Display warnings for missing models."""
    status = rag_engine.get_status()
    if not status.get("embedding_model_loaded"):
        msg = "Warning: embedding model not loaded. Retrieval will be limited."
        console.print(msg if no_color else f"[yellow]{msg}[/yellow]")
    if not status.get("generator_model_loaded"):
        msg = "Warning: generator model not loaded. Responses will be limited."
        console.print(msg if no_color else f"[yellow]{msg}[/yellow]")


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
            "[bold]Commands:[/] 'exit'/'quit'/'q' to quit, 'help'/'h' for this message"
        )
        console.print(Panel(help_text, title="[blue]Help[/]", border_style="blue"))


def _process_query(
    rag_engine: RAGEngine, query: str, console: Console, no_color: bool
) -> None:
    """Process a single query and display the response."""
    if no_color:
        with console.status("Processing your query..."):
            response = rag_engine.generate_response(query)
        console.print(Panel(response, title="Response"))
    else:
        with console.status("[bold green]Processing your query...[/]"):
            response = rag_engine.generate_response(query)
        # Clean the response to ensure no unclosed tags
        clean_response = response.replace("[/", "").replace("[", "[")
        console.print(
            Panel(clean_response, title="[bold]💡 Response[/]", border_style="green")
        )


def _handle_exit(console: Console, no_color: bool) -> None:
    """Handle exit message display."""
    if no_color:
        console.print("\nGoodbye!")
    else:
        console.print("\n[yellow]👋 Goodbye![/yellow]")


def run_tui(no_color: bool = False, force: bool = False) -> None:
    """Run the Text User Interface with CLI policy compliance.

    Args:
        no_color: If True, disable colored output
        force: If True, force TUI mode even in non-interactive environments
    """
    # Skip TUI in non-interactive environments unless forced
    if not sys.stdin.isatty() and not force and not os.getenv("FORCE_TUI"):
        print("RAG Transformer TUI - Text User Interface")
        print("Non-interactive environment detected. Use --force to override.")
        print("For non-interactive usage, try: rag --query 'your question'")
        return

    console = Console(force_terminal=force, no_color=no_color)

    try:
        rag_engine = RAGEngine()
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG engine: {e}[/red]")
        sys.exit(1)

    _display_welcome(console, no_color)
    _display_model_status(console, rag_engine, no_color)

    while True:
        try:
            query = Prompt.ask("[cyan]❯[/]").strip()

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

    run_tui(no_color=parsed_args.no_color, force=parsed_args.force)


if __name__ == "__main__":
    main()
