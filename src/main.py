"""
Main entry point for the RAG Transformer application
"""
from .rag_engine import RAGEngine


def main():
    """Main interactive loop"""
    print("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")

    # Initialize RAG engine
    rag_engine = RAGEngine()

    while True:
        try:
            # Get user input
            query = input("\nEnter your query (or 'exit'/'help' to interact): ").strip()

            # Handle special commands
            if query.lower() == 'exit':
                print("Exiting RAG Transformer. Goodbye!")
                break

            if query.lower() == 'help':
                print("\nThis is an Agentic AI Assistant covering:")
                print("- Machine Learning concepts")
                print("- Science Fiction Movies")
                print("- Cosmos and Astronomy")
                print("I can perform calculations and search Wikipedia.")
                print("Ask about AI, movies, space, or scientific topics!")
                continue

            # Check for empty query
            if not query:
                print("Please enter a valid query.")
                continue

            # Generate and print response
            response = rag_engine.generate_response(query)
            print(f"\nResponse: {response}")

        except KeyboardInterrupt:
            print("\nExiting RAG Transformer. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
