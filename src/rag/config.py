"""
Configuration for RAG Transformer
"""

import logging
import os
import sys
from pathlib import Path


class Config:
    """Project configuration for RAG Transformer."""

    def __init__(self) -> None:
        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]
        # Runtime device selection (defaults to CPU on macOS for stability)
        default_device = "cpu" if sys.platform == "darwin" else "cpu"
        self.DEVICE = os.getenv("RAG_DEVICE", default_device)
        self.USE_FAISS = os.getenv("USE_FAISS", "1") not in {"0", "false", "False"}
        # Models
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google/flan-t5-small")

        # API Keys
        self.TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
        self.NASA_API_KEY = os.getenv("NASA_API_KEY", "")

        # Dataset and knowledge base
        self.DATASET_DIR = self._resolve_project_path(
            os.getenv("DATASET_DIR", "datasets"),
            fallback="datasets",
        )
        self.KNOWLEDGE_BASE_FILE = self._resolve_project_path(
            os.getenv("KNOWLEDGE_BASE_FILE", "knowledge_base.json"),
            allow_file=True,
            fallback="knowledge_base.json",
        )
        self.MOVIE_PAGES = self._get_int_env("MOVIE_PAGES", 5)
        self.COSMOS_DAYS = self._get_int_env("COSMOS_DAYS", 7)

        # Output and cache directories
        self.OUTPUT_DIR = self._resolve_project_path(
            os.getenv("OUTPUT_DIR", "outputs"),
            fallback="outputs",
        )
        self.CACHE_DIR = self._resolve_project_path(
            os.getenv("CACHE_DIR", ".cache"),
            fallback=".cache",
        )

        # Memory (privacy-sensitive; defaults to in-session only)
        # off: no memory, session: in-memory, persist: sqlite on disk in CACHE_DIR
        self.MEMORY_MODE = os.getenv("RAG_MEMORY_MODE", "session").lower()
        self.MEMORY_DB = os.getenv("RAG_MEMORY_DB", "memory.sqlite3")

        # Web tools (disabled by default)
        self.ENABLE_WEB = os.getenv("RAG_ENABLE_WEB", "0") in {"1", "true", "True"}
        self.SEARCH_PROVIDER = os.getenv("RAG_SEARCH_PROVIDER", "duckduckgo").lower()

        # LLM backend selection (ollama by default)
        self.LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "ollama").lower()
        openai_model = os.getenv("OPENAI_MODEL", "").strip()
        if openai_model in {"gpt5.3", "gpt-5.3"}:
            openai_model = "gpt-5.3-chat-latest"
        self.OPENAI_MODEL = openai_model or "gpt-5.3-chat-latest"
        self.SKILL_FILE = self._resolve_project_path(
            os.getenv("RAG_SKILL_FILE", "skill.asc"),
            allow_file=True,
            fallback="skill.asc",
        )

        # Cerebras backend (OpenAI-compatible HTTP API)
        self.CEREBRAS_BASE_URL = os.getenv(
            "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
        ).rstrip("/")
        self.CEREBRAS_MODELS = [
            m.strip()
            for m in os.getenv(
                "CEREBRAS_MODELS",
                "llama3.1-8b,qwen-3-235b-a22b-instruct-2507",
            ).split(",")
            if m.strip()
        ]
        self.CEREBRAS_MODEL = os.getenv(
            "CEREBRAS_MODEL", self.CEREBRAS_MODELS[0] if self.CEREBRAS_MODELS else ""
        )

        # Ollama backend (local server)
        self.OLLAMA_BASE_URL = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

        # System settings
        self.MAX_WORKERS = self._get_int_env("MAX_WORKERS", 5)
        self.TOP_K_RETRIEVAL = self._get_int_env("TOP_K_RETRIEVAL", 3)
        self.MAX_ITERATIONS = self._get_int_env("MAX_ITERATIONS", 3)
        self.MAX_LENGTH = self._get_int_env("MAX_LENGTH", 150)

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        self._setup_logging()

        # Validate paths
        self._validate_paths()

    def _get_int_env(self, var_name: str, default: int) -> int:
        """Helper to safely get an integer environment variable."""
        try:
            return int(os.getenv(var_name, default))
        except ValueError:
            logging.warning(f"Invalid integer for {var_name}, defaulting to {default}")
            return default

    def _setup_logging(self) -> None:
        """Set up logging configuration and avoid duplicate third-party logs.

        Several upstream libraries (notably Hugging Face / Transformers) may attach
        their own handlers. If our root logger also has a handler, messages can be
        emitted twice (once via the library handler and once via propagation to
        root). We prefer a single, consistent stream, so we clear known
        third-party handlers and let records propagate to root.
        """
        level = getattr(logging, self.LOG_LEVEL, logging.INFO)

        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
        else:
            root_logger.setLevel(level)

        self._dedupe_third_party_loggers()

    def _dedupe_third_party_loggers(self) -> None:
        """Remove extra handlers from noisy third-party loggers to avoid repeats."""
        for logger_name in (
            "huggingface_hub",
            "transformers",
            "sentence_transformers",
            "urllib3",
        ):
            third_party_logger = logging.getLogger(logger_name)
            if third_party_logger.handlers and third_party_logger.propagate:
                third_party_logger.handlers.clear()
            third_party_logger.propagate = True

    def _resolve_project_path(
        self, value: str, allow_file: bool = False, fallback: str = "datasets"
    ) -> Path:
        """Resolve a path safely within the project root."""
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.PROJECT_ROOT / candidate).resolve()
        try:
            candidate.relative_to(self.PROJECT_ROOT)
        except ValueError:
            fallback_path = self.PROJECT_ROOT / fallback
            logging.warning(
                f"Path '{value}' is outside project root. Using '{fallback_path}'."
            )
            return fallback_path
        return candidate

    def _validate_paths(self) -> None:
        """Ensure directories exist, create if needed."""
        for path in [self.DATASET_DIR, self.OUTPUT_DIR, self.CACHE_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.KNOWLEDGE_BASE_FILE.exists():
            logging.warning(
                f"Knowledge base file '{self.KNOWLEDGE_BASE_FILE}' does not exist."
            )


# Example usage
# config = Config()
# print(config.EMBEDDING_MODEL, config.DATASET_DIR)
