"""
Configuration for RAG Transformer
"""

import logging
import os
from pathlib import Path


class Config:
    """Project configuration for RAG Transformer."""

    def __init__(self):
        # Models
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google/flan-t5-small")

        # API Keys
        self.TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
        self.NASA_API_KEY = os.getenv("NASA_API_KEY", "")

        # Dataset and knowledge base
        self.DATASET_DIR = Path(os.getenv("DATASET_DIR", "datasets"))
        self.KNOWLEDGE_BASE_FILE = Path(
            os.getenv("KNOWLEDGE_BASE_FILE", "knowledge_base.json")
        )
        self.MOVIE_PAGES = self._get_int_env("MOVIE_PAGES", 5)
        self.COSMOS_DAYS = self._get_int_env("COSMOS_DAYS", 7)

        # Output and cache directories
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))

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

    def _setup_logging(self):
        """Set up basic logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL, logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _validate_paths(self):
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
