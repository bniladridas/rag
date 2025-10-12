"""
Configuration settings for the RAG Transformer
"""
import os
from dotenv import load_dotenv

# Safely load .env if it exists
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

class Config:
    """Configuration class for API keys and settings"""

    # API Keys (empty defaults for CI and Docker)
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
    NASA_API_KEY = os.getenv('NASA_API_KEY', '')

    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    GENERATOR_MODEL = 'google/flan-t5-small'

    # Data settings
    BASE_DIR = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
    KNOWLEDGE_BASE_FILE = os.path.join(DATASET_DIR, 'knowledge_base.json')

    # Ensure directories exist
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Fetching settings
    MAX_WORKERS = 5
    MOVIE_PAGES = 5
    COSMOS_DAYS = 30

    # RAG settings
    TOP_K_RETRIEVAL = 3
    MAX_ITERATIONS = 3
    MAX_LENGTH = 150
