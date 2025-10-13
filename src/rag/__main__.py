"""
Main entry point for the RAG Transformer application
"""

import sys

from .rag_engine import RAGEngine


def main():
    """Main interactive loop or help display in non-interactive mode"""

    # Detect non-interactive environment (e.g., CI or Docker run)
    if not sys.stdin.isatty():
        print("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
        print("Non-interactive environment detected. Exiting safely.")
        return

    print("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")

    rag_engine = RAGEngine()

    while True:
        try:
            query = input("\nEnter your query (or 'exit'/'help' to interact): ").strip()

            if query.lower() == "exit":
                print("Exiting RAG Transformer. Goodbye!")
                break

            if query.lower() == "help":
                print("\nThis is an Agentic AI Assistant covering:")
                print("- Machine Learning concepts")
                print("- Science Fiction Movies")
                print("- Cosmos and Astronomy")
                print("I can perform calculations and search Wikipedia.")
                print("Ask about AI, movies, space, or scientific topics!")
                continue

            if not query:
                print("Please enter a valid query.")
                continue

            response = rag_engine.generate_response(query)
            print(f"\nResponse: {response}")

        except KeyboardInterrupt:
            print("\nExiting RAG Transformer. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
