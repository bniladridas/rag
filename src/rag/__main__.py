"""
Main entry point for the RAG Transformer application
"""

import sys
import argparse
from typing import Optional

from .rag_engine import RAGEngine
from .__version__ import __version__


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
  rag --version                 Show version information
  rag --help                    Show this help message

For more information, visit: https://github.com/harpertoken/rag
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Ask a single question and exit (non-interactive mode)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output for debugging"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    return parser


def print_welcome_message(verbose: bool = False) -> None:
    """Print welcome message with optional verbose information"""
    print("🤖 Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    
    if verbose:
        print(f"Version: {__version__}")
        print("Knowledge areas: Machine Learning, Science Fiction, Cosmos")
        print("Available tools: Calculator, Wikipedia, Time/Date")
    
    print("Type 'exit' to quit, 'help' for instructions")


def handle_single_query(query: str, verbose: bool = False) -> None:
    """Handle a single query in non-interactive mode"""
    if verbose:
        print(f"Processing query: {query}")
    
    try:
        rag_engine = RAGEngine()
        response = rag_engine.generate_response(query)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def interactive_mode(verbose: bool = False) -> None:
    """Run the interactive CLI mode"""
    # Detect non-interactive environment (e.g., CI or Docker run)
    if not sys.stdin.isatty():
        print("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
        print("Non-interactive environment detected. Use --query for single questions.")
        return

    print_welcome_message(verbose)
    
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            query = input("\n❯ ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() in ["help", "h"]:
                print("\n📚 RAG Transformer Help:")
                print("• Ask about Machine Learning, AI, and Data Science")
                print("• Inquire about Science Fiction movies and plots")
                print("• Explore Cosmos, astronomy, and space science")
                print("• Use built-in tools:")
                print("  - CALC: <expression>  (e.g., 'CALC: 2^10')")
                print("  - WIKI: <topic>       (e.g., 'WIKI: Quantum Computing')")
                print("  - TIME:               (current date and time)")
                print("• Commands: 'exit'/'quit'/'q' to quit, 'help'/'h' for this message")
                continue

            if not query:
                print("Please enter a valid query.")
                continue

            if verbose:
                print(f"Processing: {query}")

            response = rag_engine.generate_response(query)
            print(f"\n💡 {response}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            if verbose:
                import traceback
                error_msg += f"\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)


def main(args: Optional[list] = None) -> None:
    """Main entry point with argument parsing and CLI policy enforcement"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Handle single query mode
    if parsed_args.query:
        handle_single_query(parsed_args.query, parsed_args.verbose)
        return
    
    # Handle interactive mode
    interactive_mode(parsed_args.verbose)


if __name__ == "__main__":
    main()
