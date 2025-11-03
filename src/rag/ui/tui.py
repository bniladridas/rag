"""
Text User Interface for RAG Transformer
"""

import os
import sys
import argparse
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..rag_engine import RAGEngine
from ..__version__ import __version__


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


def run_tui(no_color: bool = False, force: bool = False) -> None:
    """Run the Text User Interface with CLI policy compliance"""

    # Skip TUI in non-interactive environments unless forced
    if not sys.stdin.isatty() and not force and not os.getenv("FORCE_TUI"):
        print("RAG Transformer TUI - Text User Interface")
        print("Non-interactive environment detected. Use --force to override.")
        print("For non-interactive usage, try: rag --query 'your question'")
        return

    # Initialize console with color settings
    console = Console(force_terminal=force, no_color=no_color)

    try:
        rag_engine = RAGEngine()
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG engine: {e}[/red]")
        sys.exit(1)

    # Welcome panel
    if no_color:
        console.print(
            Panel.fit(
                "Agentic RAG Transformer\n"
                "ML, Sci-Fi, and Cosmos Assistant\n"
                f"Version {__version__}"
            )
        )
        console.print("Type 'exit'/'quit' to quit, 'help' for instructions.\n")
    else:
        console.print(
            Panel.fit(
                "[bold blue]🤖 Agentic RAG Transformer[/bold blue]\n"
                "[green]ML, Sci-Fi, and Cosmos Assistant[/green]\n"
                f"[dim]Version {__version__}[/dim]"
            )
        )
        console.print("Type 'exit'/'quit' to quit, 'help' for instructions.\n")

    while True:
        try:
            query = Prompt.ask("[bold cyan]❯[/bold cyan]").strip()

            if query.lower() in ["exit", "quit", "q"]:
                if no_color:
                    console.print("Goodbye!")
                else:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                break

            if query.lower() in ["help", "h"]:
                if no_color:
                    console.print(
                        Panel(
                            "RAG Transformer Help\n\n"
                            "• Ask about Machine Learning, AI, and Data Science\n"
                            "• Inquire about Science Fiction movies and plots\n"
                            "• Explore Cosmos, astronomy, and space science\n\n"
                            "Built-in Tools:\n"
                            "• CALC: <expression>  (e.g., 'CALC: 2^10')\n"
                            "• WIKI: <topic>       (e.g., 'WIKI: Quantum Computing')\n"
                            "• TIME:               (current date and time)\n\n"
                            "Commands: 'exit'/'quit'/'q' to quit, 'help'/'h' for this message",
                            title="Help",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            "[bold]📚 RAG Transformer Help[/bold]\n\n"
                            "• Ask about [blue]Machine Learning[/blue], AI, and Data Science\n"
                            "• Inquire about [magenta]Science Fiction[/magenta] movies and plots\n"
                            "• Explore [green]Cosmos[/green], astronomy, and space science\n\n"
                            "[bold]Built-in Tools:[/bold]\n"
                            "• [cyan]CALC:[/cyan] <expression>  (e.g., 'CALC: 2^10')\n"
                            "• [cyan]WIKI:[/cyan] <topic>       (e.g., 'WIKI: Quantum Computing')\n"
                            "• [cyan]TIME:[/cyan]               (current date and time)\n\n"
                            "[bold]Commands:[/bold] 'exit'/'quit'/'q' to quit, "
                            "'help'/'h' for this message",
                            title="Help",
                            border_style="blue",
                        )
                    )
                continue

            if query.lower() == "clear":
                console.clear()
                continue

            if not query:
                if no_color:
                    console.print("Please enter a valid query.")
                else:
                    console.print("[yellow]Please enter a valid query.[/yellow]")
                continue

            # Show processing indicator for longer queries
            if no_color:
                with console.status("Processing your query..."):
                    response = rag_engine.generate_response(query)
                console.print(
                    Panel(
                        response,
                        title="Response",
                    )
                )
            else:
                with console.status("[bold green]Processing your query..."):
                    response = rag_engine.generate_response(query)
                console.print(
                    Panel(
                        f"[green]{response}[/green]",
                        title="💡 Response",
                        border_style="green",
                    )
                )

        except KeyboardInterrupt:
            if no_color:
                console.print("\nGoodbye!")
            else:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except EOFError:
            if no_color:
                console.print("\nGoodbye!")
            else:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(
                Panel(
                    f"An error occurred: {e}" if no_color else f"[red]An error occurred: {e}[/red]",
                    title="Error" if no_color else "❌ Error",
                    border_style=None if no_color else "red",
                )
            )


def main(args: Optional[list] = None) -> None:
    """Main entry point for TUI with argument parsing"""
    parser = create_tui_parser()
    parsed_args = parser.parse_args(args)

    run_tui(no_color=parsed_args.no_color, force=parsed_args.force)


if __name__ == "__main__":
    main()
