"""
RAG Engine for retrieval-augmented generation
"""
import os
import json
import faiss
import re
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import Config
from .tools import ToolExecutor


class RAGEngine:
    """Retrieval-Augmented Generation engine"""

    def __init__(self):
        self.config = Config()
        self.tool_executor = ToolExecutor()

        # Initialize models
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.GENERATOR_MODEL)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.config.GENERATOR_MODEL)

        # Knowledge base
        self.knowledge_base = []
        self.index = None

        # Query embedding cache
        self.query_cache = {}

        # Load knowledge base
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load documents from knowledge base file"""
        kb_path = os.path.join(self.config.DATASET_DIR, self.config.KNOWLEDGE_BASE_FILE)
        try:
            with open(kb_path, 'r') as f:
                documents = json.load(f)
                self.add_documents(documents)
                print(f"Loaded {len(documents)} documents from knowledge base")
        except FileNotFoundError:
            print(f"Knowledge base not found at {kb_path}. Run data collection first.")
            # Fallback minimal knowledge
            fallback_docs = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Science fiction explores futuristic concepts and advanced technology."
            ]
            self.add_documents(fallback_docs)

    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base and create FAISS index"""
        if not documents:
            return

        # Embed documents
        embeddings = self.embedding_model.encode(documents)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Store documents
        self.knowledge_base.extend(documents)

    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve most relevant documents for a query"""
        if not self.index or len(self.knowledge_base) == 0:
            return []

        # Handle very short queries
        if len(query.split()) < 2:
            return self.knowledge_base[:self.config.TOP_K_RETRIEVAL]

        # Embed query and search (with caching)
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.embedding_model.encode([query])
            self.query_cache[query] = query_embedding

        distances, indices = self.index.search(query_embedding, self.config.TOP_K_RETRIEVAL)

        # Retrieve documents
        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        return retrieved_docs

    def generate_response(self, query: str) -> str:
        """Generate response using RAG with tool support"""
        # Clean query
        query = query.strip().strip('"').strip("'")

        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().split()[0] in greetings:
            return "Hello! I'm an agentic AI assistant with knowledge about machine learning, sci-fi movies, and cosmos. I can use tools like calculations. How can I help you today?"

        # Handle direct tool calls
        if query.upper().startswith(("CALC:", "WIKI:", "TIME:")):
            return self.tool_executor.execute_tool(query)

        # Handle direct calculations
        if 'calculate' in query.lower() or re.search(r'\d+\s*[\+\-\*/]\s*\d+', query):
            expr_match = re.search(r'calculate\s+(.+)', query, re.IGNORECASE)
            if expr_match:
                expr = expr_match.group(1).strip()
            else:
                expr = re.sub(r'[^\d\+\-\*/\.\(\)\s]', '', query).strip()

            if expr:
                return self.tool_executor.execute_tool(f"CALC: {expr}")

        # Retrieve context and generate response
        context_docs = self.retrieve_context(query)
        context = " ".join(context_docs)

        # Agent loop with tool usage
        for _ in range(self.config.MAX_ITERATIONS):
            input_text = (
                f"Context information: {context}\n\n"
                f"{self.tool_executor.get_available_tools()}\n\n"
                f"Question: {query}\n\n"
                f"Answer the question using the context. If you need external\n"
                f"information, use a tool by responding with the tool command.\n"
                f"Otherwise, provide a direct answer."
            )

            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.generator.generate(
                **inputs,
                max_length=self.config.MAX_LENGTH,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check for tool usage
            if response.upper().startswith(("CALC:", "WIKI:", "TIME:")):
                tool_result = self.tool_executor.execute_tool(response)
                context += f"\nTool result: {tool_result}"
                continue  # Continue loop to generate final response

            # Return final response
            if not response or len(response.split()) < 3:
                response = (
                    "I apologize, but I couldn't generate a specific response.\n"
                    "Could you please rephrase your query about machine learning,\n"
                    "sci-fi movies, or cosmos?"
                )
            return response

        # Fallback
        return "I used tools but couldn't finalize a response. Please try a different query."
