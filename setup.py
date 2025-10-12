"""
Setup script for RAG Transformer
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rag=src.main:main',
            'rag-collect=src.data_fetcher:main',
        ],
    },
    author="RAG Transformer Team",
    description="Agentic RAG system for ML, Sci-Fi, and Cosmos knowledge",
    python_requires=">=3.8",
)