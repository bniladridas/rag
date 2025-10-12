"""
Configuration settings for the RAG Transformer
"""
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class Config:
    """Configuration class for API keys and settings"""

    # API Keys
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
    NASA_API_KEY = os.getenv('NASA_API_KEY', '')

    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    GENERATOR_MODEL = 'google/flan-t5-small'

    # Data settings
    DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
    KNOWLEDGE_BASE_FILE = 'knowledge_base.json'

    # Fetching settings
    MAX_WORKERS = 5
    MOVIE_PAGES = 5
    COSMOS_DAYS = 30

    # RAG settings
    TOP_K_RETRIEVAL = 3
    MAX_ITERATIONS = 3
    MAX_LENGTH = 150
