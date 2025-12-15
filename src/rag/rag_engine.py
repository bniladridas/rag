"""
RAG Engine for retrieval-augmented generation
"""

import json
import logging
import os
import re
import sys
from typing import List

import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import Config
from .tools import ToolExecutor

logger = logging.getLogger(__name__)

# Optional: use fallback embeddings if SentenceTransformer not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore
    logger.warning("sentence_transformers not installed. Using fallback embeddings.")


class RAGEngine:
    """Retrieval-Augmented Generation engine"""

    def __init__(self):
        self.config = Config()
        self.tool_executor = ToolExecutor()

        # Handle non-interactive CI/Docker environment
        if not sys.stdin.isatty():
            self.config.MAX_ITERATIONS = 1

        # Initialize models safely
        self.embedding_model = None
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            except Exception:
                logger.warning("Failed to load embedding model. Using fallback.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.GENERATOR_MODEL)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.GENERATOR_MODEL
            )
        except Exception:
            logger.warning("Failed to load generator model. Responses may be limited.")
            self.tokenizer = None
            self.generator = None

        # Knowledge base
        self.knowledge_base = []
        self.index = None
        self.query_cache = {}

        # Load knowledge base
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load documents from knowledge base file"""
        kb_path = os.path.join(self.config.DATASET_DIR, self.config.KNOWLEDGE_BASE_FILE)
        try:
            with open(kb_path, "r") as f:
                documents = json.load(f)
                self.add_documents(documents)
                logger.info(f"Loaded {len(documents)} documents from knowledge base")
        except FileNotFoundError:
            logger.warning(
                f"Knowledge base not found at {kb_path}. Using fallback docs."
            )
            fallback_docs = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Science fiction explores futuristic concepts and advanced technology.",
            ]
            self.add_documents(fallback_docs)

    def add_documents(self, documents: List[str]):
        """Add documents to knowledge base and create FAISS index if embeddings exist"""
        if not documents:
            return

        self.knowledge_base.extend(documents)

        if self.embedding_model:
            embeddings = self.embedding_model.encode(documents)
            dimension = int(embeddings.shape[1])  # type: ignore
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)  # type: ignore
        else:
            self.index = None  # fallback: no index

    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve most relevant documents for a query"""
        if not self.index or len(self.knowledge_base) == 0:
            return self.knowledge_base[: self.config.TOP_K_RETRIEVAL]

        if len(query.split()) < 2:
            return self.knowledge_base[: self.config.TOP_K_RETRIEVAL]

        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            assert self.embedding_model is not None
            query_embedding = self.embedding_model.encode([query])
            self.query_cache[query] = query_embedding

        distances, indices = self.index.search(
            query_embedding, self.config.TOP_K_RETRIEVAL
        )  # type: ignore

        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        return retrieved_docs

    def generate_response(self, query: str) -> str:
        """Generate response using RAG with tool support"""
        query = query.strip().strip('"').strip("'")

        greetings = ["hi", "hello", "hey", "greetings"]
        if query.lower().split()[0] in greetings:
            return (
                "Hello! I'm an agentic AI assistant with knowledge about "
                "machine learning, sci-fi movies, and cosmos. I can use tools "
                "like calculations. How can I help you today?"
            )

        if query.upper().startswith(("CALC:", "WIKI:", "TIME:")):
            return self.tool_executor.execute_tool(query)

        if "calculate" in query.lower() or re.search(r"\d+\s*[\+\-\*/]\s*\d+", query):
            expr_match = re.search(r"calculate\s+(.+)", query, re.IGNORECASE)
            expr = (
                expr_match.group(1).strip()
                if expr_match
                else re.sub(r"[^\d\+\-\*/\.\(\)\s]", "", query).strip()
            )
            if expr:
                return self.tool_executor.execute_tool(f"CALC: {expr}")

        context_docs = self.retrieve_context(query)
        context = " ".join(context_docs)

        # If generator model not loaded, return context as fallback
        if not self.tokenizer or not self.generator:
            return (
                context_docs[0]
                if context_docs
                else "No response available in CI environment."
            )

        for _ in range(self.config.MAX_ITERATIONS):
            input_text = (
                f"Context information: {context}\n\n"
                f"{self.tool_executor.get_available_tools()}\n\n"
                f"Question: {query}\n\n"
                f"Answer the question using the context. If you need external\n"
                f"information, use a tool by responding with the tool\n"
                f"command. Otherwise, provide a direct answer."
            )

            inputs = self.tokenizer(
                input_text, return_tensors="pt", max_length=512, truncation=True
            )
            outputs = self.generator.generate(
                **inputs,
                max_length=self.config.MAX_LENGTH,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response.upper().startswith(("CALC:", "WIKI:", "TIME:")):
                tool_result = self.tool_executor.execute_tool(response)
                context += f"\nTool result: {tool_result}"
                continue

            if not response or len(response.split()) < 3:
                response = (
                    "I couldn't generate a detailed response. "
                    "Please rephrase your query about machine learning, "
                    "sci-fi movies, or cosmos."
                )
            return response

        return "I used tools but couldn't finalize a response. Try a different query."
