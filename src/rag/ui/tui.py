"""
Text User Interface for RAG Transformer
"""

import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..rag_engine import RAGEngine


def run_tui():
    """Run the Text User Interface"""

    # Skip TUI in non-interactive environments (CI/Docker), unless forced for testing
    if not sys.stdin.isatty() and not os.getenv("FORCE_TUI"):
        print("Non-interactive environment detected. Skipping TUI.")
        return

    console = Console()
    rag_engine = RAGEngine()

    console.print(
        Panel.fit(
            "[bold blue]Agentic RAG Transformer[/bold blue]\n"
            "[green]ML, Sci-Fi, and Cosmos Assistant[/green]"
        )
    )
    console.print("Type 'exit' to quit, 'help' for instructions.\n")

    while True:
        try:
            query = Prompt.ask("[bold cyan]Query[/bold cyan]").strip()

            if query.lower() == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break

            if query.lower() == "help":
                console.print(
                    Panel(
                        "This AI Assistant covers:\n"
                        "- Machine Learning concepts\n"
                        "- Science Fiction Movies\n"
                        "- Cosmos and Astronomy\n"
                        "Ask about AI, movies, space, or scientific topics!",
                        title="Help",
                    )
                )
                continue

            if not query:
                console.print("[red]Please enter a valid query.[/red]")
                continue

            response = rag_engine.generate_response(query)
            console.print(Panel(f"[green]{response}[/green]", title="Response"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
