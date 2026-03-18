"""
RAG Engine for retrieval-augmented generation
"""

import json
import logging
import os
import re
import sys
from typing import List, Optional

import faiss
import requests

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore[assignment]
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import Config
from .memory import MemoryStore, format_memory_context
from .tools import ToolExecutor

logger = logging.getLogger(__name__)

# Optional: use fallback embeddings if SentenceTransformer not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore[assignment,misc]
    logger.warning("sentence_transformers not installed. Using fallback embeddings.")


class RAGEngine:
    """Retrieval-Augmented Generation engine"""

    def __init__(self) -> None:  # noqa: C901
        self.config = Config()
        self.tool_executor = ToolExecutor()
        self.user_name: Optional[str] = None
        self.memory = MemoryStore.build_for_mode(
            self.config.MEMORY_MODE, self.config.CACHE_DIR, self.config.MEMORY_DB
        )
        self.system_instructions = self._load_skill_instructions()
        self.model_status = {
            "embedding_model_loaded": False,
            "generator_model_loaded": False,
        }
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if sys.platform == "darwin":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            if torch is not None:
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
            if sys.version_info >= (3, 14):
                logger.warning(
                    "Python 3.14 on macOS can be unstable with torch/libomp. "
                    "Use Python 3.12/3.13/3.14 or set RAG_DEVICE=cpu and OMP_NUM_THREADS=1."
                )

        # Handle non-interactive CI/Docker environment
        if not sys.stdin.isatty():
            self.config.MAX_ITERATIONS = 1

        # Initialize models safely
        self.embedding_model = None
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(
                    self.config.EMBEDDING_MODEL, device=self.config.DEVICE
                )
                self.model_status["embedding_model_loaded"] = True
            except Exception:
                logger.warning("Failed to load embedding model. Using fallback.")

        self.tokenizer = None
        self.generator = None

        self.llm_backend = self.config.LLM_BACKEND
        self.openai_client = None
        self._http_session = requests.Session()

        if self.llm_backend == "openai":
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.getenv("OPENAI_API_KEY", "")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.model_status["generator_model_loaded"] = True
                else:
                    logger.warning(
                        "OPENAI_API_KEY not set. Falling back to local generator."
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        elif self.llm_backend == "cerebras":
            if os.getenv("CEREBRAS_API_KEY", ""):
                self.model_status["generator_model_loaded"] = True
            else:
                logger.warning(
                    "CEREBRAS_API_KEY not set. Falling back to local generator."
                )
        elif self.llm_backend == "ollama":
            # Assume local server; we mark as loaded and report runtime error if unreachable.
            self.model_status["generator_model_loaded"] = True

        if self.openai_client is None and self.llm_backend not in {
            "cerebras",
            "ollama",
        }:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.GENERATOR_MODEL
                )
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.GENERATOR_MODEL
                )
                self.model_status["generator_model_loaded"] = True
            except Exception:
                logger.warning(
                    "Failed to load generator model. Responses may be limited."
                )
                self.tokenizer = None
                self.generator = None

        # Knowledge base
        self.knowledge_base: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.query_cache: dict = {}

        # Load knowledge base
        self.load_knowledge_base()

    def get_status(self) -> dict:
        """Return model status flags for user-facing messaging."""
        return dict(self.model_status)

    def load_knowledge_base(self) -> None:
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

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to knowledge base and create FAISS index if embeddings exist"""
        if not documents:
            return

        self.knowledge_base.extend(documents)

        if self.embedding_model and self.config.USE_FAISS:
            embeddings = self.embedding_model.encode(documents)
            dimension = int(embeddings.shape[1])
            self.index = faiss.IndexFlatL2(dimension)
            assert self.index is not None
            self.index.add(embeddings)
        else:
            self.index = None  # fallback: no index

    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve most relevant documents for a query"""
        if not self.index:
            return self.knowledge_base[: self.config.TOP_K_RETRIEVAL]

        if len(self.knowledge_base) == 0:
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
        )

        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        return retrieved_docs

    def generate_response(self, query: str) -> str:  # noqa: C901
        """Generate response using RAG with tool support"""
        query = query.strip().strip('"').strip("'")

        if not query:
            return "Please enter a valid query."

        lowered = query.lower()

        # Very small session memory: remember the user's name if they share it.
        # This helps in TUI mode, where the same engine instance is reused.
        name_match = re.search(
            r"\b(?:my name is|i am|i'm|im)\s+([A-Za-z][A-Za-z0-9_\-']{0,31})\b",
            query,
            flags=re.IGNORECASE,
        )
        if name_match:
            shared_name = name_match.group(1).strip().strip(".,!?:;")
            if shared_name:
                self.user_name = shared_name
                self.memory.set_fact("user_name", shared_name)
                response = (
                    f"Nice to meet you, {shared_name}. "
                    "Want to ask about machine learning, sci-fi, or the cosmos?"
                )
                self._remember_turn(query, response)
                return response

        # Handle "what's my name?" style queries deterministically (avoid model drift).
        if ("my name" in lowered or lowered in {"who am i", "who am i?"}) and any(
            token in lowered for token in ("what", "tell", "who", "?")
        ):
            fact = self.memory.get_fact("user_name")
            remembered = getattr(self, "user_name", None) or (
                fact.value if fact else None
            )
            if remembered:
                response = f"You told me your name is {remembered}."
                self._remember_turn(query, response)
                return response
            response = "I don't know yet—tell me with “I am <name>”."
            self._remember_turn(query, response)
            return response

        greetings = ["hi", "hello", "hey", "greetings"]
        if lowered.split()[0] in greetings:
            response = (
                "Hello! I'm an agentic AI assistant with knowledge about "
                "machine learning, sci-fi movies, and cosmos. I can use tools "
                "like calculations. How can I help you today?"
            )
            self._remember_turn(query, response)
            return response

        # Basic small-talk handling for non-domain queries
        if lowered in {"am", "uh", "um"}:
            response = "Could you share a bit more detail so I can help?"
            self._remember_turn(query, response)
            return response
        if lowered in {"how are you", "how are you?", "how r u", "how r u?"}:
            response = "Doing well—thanks. What would you like to explore?"
            self._remember_turn(query, response)
            return response
        if "math" in lowered:
            response = "Sure—ask a math question or use CALC:, e.g. `CALC: 2^10`."
            self._remember_turn(query, response)
            return response

        if re.search(
            r"\b(what\s+is\s+)?(the\s+)?(current\s+)?(date|time)\b", lowered
        ) or lowered in {"time", "date", "time now", "date today", "what time is it"}:
            response = self.tool_executor.execute_tool("TIME:")
            self._remember_turn(query, response)
            return response

        if query.upper().startswith(("CALC:", "WIKI:", "TIME:", "SEARCH:", "WEB:")):
            response = self.tool_executor.execute_tool(query)
            self._remember_turn(query, response)
            return response

        if "calculate" in query.lower() or re.search(r"\d+\s*[\+\-\*/]\s*\d+", query):
            expr_match = re.search(r"calculate\s+(.+)", query, re.IGNORECASE)
            expr = (
                expr_match.group(1).strip()
                if expr_match
                else re.sub(r"[^\d\+\-\*/\.\(\)\s]", "", query).strip()
            )
            if expr:
                response = self.tool_executor.execute_tool(f"CALC: {expr}")
                self._remember_turn(query, response)
                return response

        context_docs = self.retrieve_context(query)
        memory_facts = self.memory.search_facts(query)
        memory_msgs = self.memory.recent_messages(limit=8)
        memory_context = format_memory_context(memory_facts, memory_msgs)
        context = " ".join(context_docs)
        if memory_context:
            context = f"{memory_context}\n\nKnowledge base:\n{context}".strip()

        # If generator model not loaded, return context as fallback
        if (
            (not self.tokenizer or not self.generator)
            and self.openai_client is None
            and self.llm_backend not in {"cerebras", "ollama"}
        ):
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

            response = self._generate_text(input_text)

            if response.upper().startswith(("CALC:", "WIKI:", "TIME:")):
                tool_result = self.tool_executor.execute_tool(response)
                context += f"\nTool result: {tool_result}"
                continue
            if response.upper().startswith(("SEARCH:", "WEB:")):
                tool_result = self.tool_executor.execute_tool(response)
                context += f"\nTool result: {tool_result}"
                continue

            if not response or len(response.split()) < 3:
                response = (
                    "I didn't get enough signal to answer that. "
                    "Try a question about machine learning, sci-fi, or the cosmos."
                )
            self._remember_turn(query, response)
            return response

        return "I used tools but couldn't finalize a response. Try a different query."

    def _remember_turn(self, user: str, assistant: str) -> None:
        self.memory.add_message("user", user)
        self.memory.add_message("assistant", assistant)

    def _generate_text(self, prompt: str) -> str:
        """Generate text with the configured backend."""
        backend = (self.llm_backend or "local").lower()

        if backend == "openai":
            if self.openai_client is None:
                return "OpenAI backend is selected but OPENAI_API_KEY is not set."
            model = self.config.OPENAI_MODEL or os.getenv("OPENAI_MODEL", "")
            if not model:
                # Keep this explicit to avoid guessing model names.
                return "OpenAI backend is enabled but OPENAI_MODEL is not set."
            try:
                resp = self.openai_client.responses.create(
                    model=model,
                    instructions=self.system_instructions,
                    input=prompt,
                    max_output_tokens=max(64, int(self.config.MAX_LENGTH)),
                )
                text = getattr(resp, "output_text", None)
                if text:
                    return str(text).strip()
            except Exception:
                # Fallback to chat.completions for older surfaces.
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": self.system_instructions},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    return f"OpenAI request failed: {e}"

        if backend == "cerebras":
            return self._generate_text_cerebras(prompt)

        if backend == "ollama":
            return self._generate_text_ollama(prompt)

        if backend != "local":
            return f"Unknown backend: {backend}"

        if self.tokenizer is None or self.generator is None:
            err = self._ensure_local_generator_loaded()
            if err:
                return err

        assert self.tokenizer is not None and self.generator is not None
        prompt = f"{self.system_instructions}\n\n{prompt}".strip()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = self.generator.generate(
            **inputs,
            max_length=self.config.MAX_LENGTH,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
        )
        return str(self.tokenizer.decode(outputs[0], skip_special_tokens=True)).strip()

    def current_backend_and_model(self) -> str:
        backend = (self.llm_backend or "local").lower()
        if backend == "openai":
            return f"openai:{self.config.OPENAI_MODEL}"
        if backend == "cerebras":
            return f"cerebras:{self.config.CEREBRAS_MODEL}"
        if backend == "ollama":
            return f"ollama:{self.config.OLLAMA_MODEL}"
        return f"local:{self.config.GENERATOR_MODEL}"

    def available_backends(self) -> list[str]:
        return ["local", "openai", "cerebras", "ollama"]

    def set_backend(self, backend: str) -> str:
        backend = (backend or "").strip().lower()
        if backend not in set(self.available_backends()):
            return f"Unknown backend: {backend}"

        # Validate credentials before switching to remote backends.
        if backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return "OPENAI_API_KEY is not set; backend unchanged."
            try:
                from openai import OpenAI  # type: ignore

                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                return f"Failed to initialize OpenAI client: {e}"

        if backend == "cerebras":
            if not os.getenv("CEREBRAS_API_KEY", ""):
                return "CEREBRAS_API_KEY is not set; backend unchanged."

        if backend == "local":
            err = self._ensure_local_generator_loaded()
            if err:
                return err

        # Switch backend last so failures above don't leave us half-switched.
        self.llm_backend = backend
        self.config.LLM_BACKEND = backend
        # If leaving OpenAI, we keep client cached; harmless and makes switching faster.
        return f"Switched backend to {backend} ({self.current_backend_and_model()})."

    def set_active_model(self, model_name: str) -> str:
        """Set model for the active backend (best-effort)."""
        model_name = (model_name or "").strip()
        if not model_name:
            return "Model name is required."

        backend = (self.llm_backend or "local").lower()
        if backend == "openai":
            self.config.OPENAI_MODEL = model_name
            return f"Switched OpenAI model to {model_name}."
        if backend == "cerebras":
            self.config.CEREBRAS_MODEL = model_name
            return f"Switched Cerebras model to {model_name}."
        if backend == "ollama":
            self.config.OLLAMA_MODEL = model_name
            return f"Switched Ollama model to {model_name}."
        return "Model switching is not supported for the local backend."

    def available_models(self) -> list[str]:
        """Return a best-effort list of models for the active backend."""
        backend = (self.llm_backend or "local").lower()
        if backend == "cerebras":
            return list(self.config.CEREBRAS_MODELS or [])
        if backend == "ollama":
            return self._ollama_list_models() or (
                [self.config.OLLAMA_MODEL] if self.config.OLLAMA_MODEL else []
            )
        if backend == "openai":
            # Model catalogs change frequently; keep it explicit.
            return [self.config.OPENAI_MODEL] if self.config.OPENAI_MODEL else []
        # Local backend does not have a discoverable model catalog here.
        return []

    def models_hint(self) -> str:
        backend = (self.llm_backend or "local").lower()
        if backend == "local":
            return (
                "Local backend doesn't expose a model list. "
                "To use Llama models, switch to Ollama (in TUI: `backend: ollama`)."
            )
        if backend == "openai":
            return (
                "OpenAI backend model discovery isn't enabled; "
                "set `OPENAI_MODEL` explicitly if you want to switch."
            )
        if backend == "cerebras":
            return "Set `CEREBRAS_MODELS` (comma-separated) to control the picker list."
        if backend == "ollama":
            return "Install models with `ollama pull <name>`; the picker lists installed tags."
        return ""

    def _ensure_local_generator_loaded(self) -> Optional[str]:
        """Load local tokenizer/generator on demand."""
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.GENERATOR_MODEL
                )
            if self.generator is None:
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.GENERATOR_MODEL
                )
            self.model_status["generator_model_loaded"] = True
            return None
        except Exception as e:
            self.tokenizer = None
            self.generator = None
            self.model_status["generator_model_loaded"] = False
            return f"Failed to load local generator model '{self.config.GENERATOR_MODEL}': {e}"

    def _generate_text_cerebras(self, prompt: str) -> str:
        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            return "Cerebras backend is enabled but CEREBRAS_API_KEY is not set."
        model = self.config.CEREBRAS_MODEL or (
            self.config.CEREBRAS_MODELS[0] if self.config.CEREBRAS_MODELS else ""
        )
        if not model:
            return "Cerebras backend is enabled but CEREBRAS_MODEL is not set."

        url = f"{self.config.CEREBRAS_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max(64, int(self.config.MAX_LENGTH)),
            "stream": False,
        }
        try:
            resp = self._http_session.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            if resp.status_code != 200:
                return f"Cerebras request failed with status {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            ) or "Empty Cerebras response."
        except Exception as e:
            return f"Cerebras request failed: {e}"

    def _generate_text_ollama(self, prompt: str) -> str:
        url = f"{self.config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": self.config.OLLAMA_MODEL or "llama3",
            "messages": [
                {"role": "system", "content": self.system_instructions},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7},
        }
        try:
            resp = self._http_session.post(url, json=payload, timeout=60)
            if resp.status_code != 200:
                return f"Ollama request failed with status {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            msg = data.get("message", {}) or {}
            return (msg.get("content", "") or "").strip() or "Empty Ollama response."
        except Exception as e:
            return f"Ollama request failed: {e}"

    def _ollama_list_models(self) -> list[str]:
        try:
            url = f"{self.config.OLLAMA_BASE_URL}/api/tags"
            resp = self._http_session.get(url, timeout=5)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            for m in data.get("models", []) or []:
                name = (m.get("name") or "").strip()
                if name:
                    models.append(name)
            # Prefer de-duplicated order.
            seen = set()
            out = []
            for n in models:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
            return out
        except Exception:
            return []

    def _load_skill_instructions(self) -> str:
        try:
            path = self.config.SKILL_FILE
            if path and path.exists():
                # Keep it short for small local models.
                text = path.read_text(encoding="utf-8").strip()
                return text[:6000]
        except Exception:
            pass
        return (
            "You are the Agentic RAG Transformer assistant. "
            "Prefer concise, correct answers. "
            "When using tools, respond with exactly one tool command like `CALC: 2+2`."
        )
