"""
Main entry point for the RAG Transformer application
"""

import argparse
import os
import sys
import traceback
from typing import Optional

from .__version__ import __version__
from .rag_engine import RAGEngine


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # Optional: dotenv is a convenience only.
        return


def should_use_color(no_color: bool = False) -> bool:
    """Determine if colored output should be used"""
    # Respect --no-color flag
    if no_color:
        return False

    # Respect NO_COLOR environment variable (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False

    # Check if output is to a terminal
    return sys.stdout.isatty()


def format_message(message: str, emoji: str = "", no_color: bool = False) -> str:
    """Format message with or without emoji based on color settings"""
    if should_use_color(no_color) and emoji:
        return f"{emoji} {message}"
    return message


def _print_model_status(rag_engine: RAGEngine, no_color: bool) -> None:
    status = rag_engine.get_status()
    if not status.get("embedding_model_loaded"):
        print(
            format_message(
                "Warning: embedding model not loaded. Retrieval will be limited.",
                "⚠️",
                no_color,
            )
        )
    if not status.get("generator_model_loaded"):
        print(
            format_message(
                "Warning: generator model not loaded. Responses will be limited.",
                "⚠️",
                no_color,
            )
        )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with CLI policies"""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rag                           Start interactive mode
  rag --query "What is ML?"     Ask a single question
  rag --quiet --query "test"    Ask a question with minimal output
  rag --version                 Show version information
  rag --help                    Show this help message

For more information, visit: https://github.com/harpertoken/rag
        """,
    )

    parser.add_argument("--version", action="version", version=f"version {__version__}")

    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question and exit (non-interactive mode)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output for debugging",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-essential output"
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    parser.add_argument(
        "--force-interactive",
        action="store_true",
        help="Force interactive mode even in non-TTY environments (for testing)",
    )

    return parser


def print_welcome_message(
    verbose: bool = False, quiet: bool = False, no_color: bool = False
) -> None:
    """Print welcome message with optional verbose information"""
    if not quiet:
        print(
            format_message(
                "Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant",
                "🤖",
                no_color,
            )
        )

        if verbose:
            print(f"Version: {__version__}")
            print("Knowledge areas: Machine Learning, Science Fiction, Cosmos")
            print("Available tools: Calculator, Wikipedia, Time/Date")

        print("Type 'exit' to quit, 'help' for instructions")


def handle_single_query(
    query: str, verbose: bool = False, quiet: bool = False, no_color: bool = False
) -> None:
    """Handle a single query in non-interactive mode"""
    if verbose and not quiet:
        print(f"Processing query: {query}")

    try:
        rag_engine = RAGEngine()
        if not quiet:
            _print_model_status(rag_engine, no_color)
        response = rag_engine.generate_response(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def interactive_mode(  # noqa: C901
    verbose: bool = False,
    quiet: bool = False,
    no_color: bool = False,
    force_interactive: bool = False,
) -> None:
    """Run the interactive CLI mode"""
    # Detect non-interactive environment (e.g., CI or Docker run)
    if not sys.stdin.isatty() and not force_interactive:
        if not quiet:
            print(
                format_message(
                    "Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant",
                    "🤖",
                    no_color,
                )
            )
            print(
                "Non-interactive environment detected. Use --query for single questions."
            )
        return

    print_welcome_message(verbose, quiet, no_color)

    try:
        rag_engine = RAGEngine()
        if not quiet:
            _print_model_status(rag_engine, no_color)
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            # Use colored or plain prompt
            model_tag = ""
            if not quiet:
                try:
                    model_tag = f"[{rag_engine.current_backend_and_model()}] "
                except Exception:
                    model_tag = ""
            prompt = (
                f"❯ {model_tag}" if should_use_color(no_color) else f"> {model_tag}"
            )
            query = input(prompt).strip()

            if query.lower() in ["exit", "quit", "q"]:
                print(format_message("Goodbye!", "👋", no_color))
                break

            if query.lower() in ["help", "h"]:
                help_title = format_message("RAG Transformer Help:", "📚", no_color)
                print(f"\n{help_title}")
                print("• Ask about Machine Learning, AI, and Data Science")
                print("• Inquire about Science Fiction movies and plots")
                print("• Explore Cosmos, astronomy, and space science")
                print("• Use built-in tools:")
                print("  - CALC: <expression>  (e.g., 'CALC: 2^10')")
                print("  - WIKI: <topic>       (e.g., 'WIKI: Quantum Computing')")
                print("  - TIME:               (current date and time)")
                print(
                    "• Commands: 'exit'/'quit'/'q' to quit, 'help'/'h' for this message"
                )
                continue

            if query.lower() == "models":
                print(f"Current: {rag_engine.current_backend_and_model()}")
                continue

            if query.lower().startswith("model:"):
                model_name = query.split(":", 1)[-1].strip()
                print(rag_engine.set_active_model(model_name))
                continue

            if not query:
                print("Please enter a valid query.")
                continue

            if verbose:
                print(f"Processing: {query}")

            response = rag_engine.generate_response(query)
            response_msg = format_message(response, "💡", no_color)
            print(f"\n{response_msg}")

        except KeyboardInterrupt:
            print(f"\n{format_message('Goodbye!', '👋', no_color)}")
            break
        except EOFError:
            print(f"\n{format_message('Goodbye!', '👋', no_color)}")
            break
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            if verbose:
                error_msg += f"\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)


def main(args: Optional[list] = None) -> None:
    """Main entry point with argument parsing and CLI policy enforcement"""
    _load_env_file()
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Handle single query mode
    if parsed_args.query:
        handle_single_query(
            parsed_args.query,
            parsed_args.verbose,
            parsed_args.quiet,
            parsed_args.no_color,
        )
        return

    # Handle interactive mode
    interactive_mode(
        parsed_args.verbose,
        parsed_args.quiet,
        parsed_args.no_color,
        parsed_args.force_interactive,
    )


if __name__ == "__main__":
    main()
