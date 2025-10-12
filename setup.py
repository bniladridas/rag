"""
Setup script for RAG Transformer
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description from README
long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="rag",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rag=rag:main',
            'rag-tui=ui.tui:run_tui',
            'rag-collect=rag.data_fetcher:main',
        ],
    },
    author="RAG Transformer Team",
    description="Agentic RAG system for ML, Sci-Fi, and Cosmos knowledge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    zip_safe=False,
)
