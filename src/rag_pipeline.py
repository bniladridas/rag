import os
import json
import torch
import faiss
import numpy as np
import re
import math
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class RAGTransformer:
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize RAG Transformer with embedding and generation models
        
        Args:
            embedding_model (str): Model for creating document embeddings
            generator_model (str): Model for text generation
        """
        # Embedding Model for Retrieval
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Generator Model
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        
        # Knowledge Base
        self.knowledge_base = []
        self.index = None
    
    def load_datasets(self, dataset_dir: str = 'src/datasets'):
        """
        Load datasets from JSON files in the specified directory

        Args:
            dataset_dir (str): Directory containing dataset JSON files
        """
        knowledge_base_path = os.path.join(dataset_dir, 'knowledge_base.json')
        try:
            with open(knowledge_base_path, 'r') as f:
                documents = json.load(f)
                print(f"Loaded {len(documents)} documents from knowledge base")
                self.add_documents(documents)
        except FileNotFoundError:
            print(f"Knowledge base not found at {knowledge_base_path}. Run data collector first.")
            # Fallback to minimal knowledge
            fallback_docs = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Science fiction explores futuristic concepts and advanced technology."
            ]
            self.add_documents(fallback_docs)
    
    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base and create FAISS index
        
        Args:
            documents (List[str]): List of documents to add
        """
        # Embed documents
        embeddings = self.embedding_model.encode(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Store documents
        self.knowledge_base.extend(documents)

    def get_tools(self) -> str:
        """Get available tools description"""
        return """Available tools:
CALC: Calculate a mathematical expression (e.g., CALC: 2 + 3 * 4)
WIKI: Search Wikipedia for information (e.g., WIKI: Machine Learning)
TIME: Get current date and time"""

    def execute_tool(self, tool_call: str) -> str:
        """Execute a tool call"""
        if tool_call.startswith("CALC:"):
            expr = tool_call[5:].strip()
            try:
                # Safe eval for simple math with basic functions
                allowed_names = {
                    "__builtins__": None,
                    "sqrt": math.sqrt,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "log": math.log,
                    "exp": math.exp,
                    "pi": math.pi,
                    "e": math.e
                }
                result = eval(expr, allowed_names)
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Invalid calculation: {e}"
        elif tool_call.startswith("WIKI:"):
            topic = tool_call[5:].strip().replace(' ', '_')
            try:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
                headers = {'User-Agent': 'RAG-Transformer/1.0 (educational project)'}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    extract = data.get('extract', 'No summary available')
                    return f"Wikipedia summary for '{topic.replace('_', ' ')}': {extract}"
                else:
                    return f"No Wikipedia page found for '{topic.replace('_', ' ')}'"
            except Exception as e:
                return f"Error fetching Wikipedia: {e}"
        elif tool_call.startswith("TIME:"):
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Current date and time: {current_time}"
        return "Unknown tool"
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant documents for a given query
        
        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve
        
        Returns:
            List[str]: Most relevant documents
        """
        # Handle very short or non-informative queries
        if len(query.split()) < 2:
            return self.knowledge_base[:top_k]
        
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents
        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        return retrieved_docs
    
    def generate_response(self, query: str) -> str:
        """
        Generate an agentic response using retrieved context and tools

        Args:
            query (str): Input query

        Returns:
            str: Generated response
        """
        # Clean the query
        query = query.strip().strip('"').strip("'")

        # Handle greeting-like queries
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().split()[0] in greetings:
            return "Hello! I'm an agentic AI assistant with knowledge about machine learning, sci-fi movies, and cosmos. I can use tools like calculations. How can I help you today?"

        # Handle direct tool calls
        if query.upper().startswith("CALC:"):
            return self.execute_tool(query)
        if query.upper().startswith("WIKI:"):
            return self.execute_tool(query)
        if query.upper().startswith("TIME:"):
            return self.execute_tool(query)

        # Handle direct calculation queries
        if 'calculate' in query.lower() or re.search(r'\d+\s*[\+\-\*/]\s*\d+', query):
            # Extract mathematical expression
            expr_match = re.search(r'calculate\s+(.+)', query, re.IGNORECASE)
            if expr_match:
                expr = expr_match.group(1).strip()
            else:
                # Try to find math expression in the query
                expr = re.sub(r'[^\d\+\-\*/\.\(\)\s]', '', query).strip()

            if expr:
                try:
                    # Safe eval for simple math
                    allowed_names = {"__builtins__": None}
                    result = eval(expr, allowed_names)
                    return f"Calculation result: {result}"
                except Exception as e:
                    return f"Invalid calculation: {e}"

        # Retrieve context
        context_docs = self.retrieve_context(query)
        context = " ".join(context_docs)

        # Agent loop
        max_iterations = 3
        for _ in range(max_iterations):
            input_text = f"Context information: {context}\n\nAvailable tools: {self.get_tools()}\n\nQuestion: {query}\n\nAnswer the question using the context. If you need external information, use a tool by responding with the tool command (e.g., WIKI: topic). Otherwise, provide a direct answer."

            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.generator.generate(**inputs,
                                              max_length=150,
                                              num_return_sequences=1,
                                              do_sample=True,
                                              temperature=0.7)

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check for tool call
            if response.startswith("CALC:"):
                tool_result = self.execute_tool(response)
                context += f"\nTool result: {tool_result}"
                continue  # Continue loop to generate final response

            # Final response
            if not response or len(response.split()) < 3:
                response = "I apologize, but I couldn't generate a specific response. Could you please rephrase your query about machine learning, sci-fi movies, or cosmos?"
            return response

        # Fallback if loop exhausted
        return "I used tools but couldn't finalize a response. Please try a different query."

def main():
    # Initialize RAG Transformer
    rag = RAGTransformer()
    
    # Load datasets
    rag.load_datasets()
    
    # Interactive query loop
    print("Agentic RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")
    
    while True:
        # Get user input
        query = input("\nEnter your query (or 'exit'/'help' to interact): ").strip()
        
        # Check for special commands
        if query.lower() == 'exit':
            print("Exiting RAG Transformer. Goodbye!")
            break
        
        if query.lower() == 'help':
            print("\nThis is an Agentic AI Assistant covering:")
            print("- Machine Learning concepts")
            print("- Science Fiction Movies")
            print("- Cosmos and Astronomy")
            print("I can perform calculations and use tools.")
            print("Ask about AI, movies, space, or scientific topics!")
            continue
        
        # Check for empty query
        if not query:
            print("Please enter a valid query.")
            continue
        
        # Generate and print response
        try:
            response = rag.generate_response(query)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
