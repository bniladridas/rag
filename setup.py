"""
Setup script for RAG Transformer
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description from README
long_description = Path("README.asc").read_text(encoding="utf-8")

setup(
    name="rag",
    version="1.6.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag=rag:main",
            "rag-tui=rag.ui.tui:main",
            "rag-collect=rag.data_fetcher:main",
        ],
    },
    author="RAG Transformer Team",
    description="Agentic RAG system for ML, Sci-Fi, and Cosmos knowledge",
    long_description=long_description,
    long_description_content_type="text/asciidoc",
    python_requires=">=3.8",
    zip_safe=False,
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    project_urls={
        "Source": "https://github.com/bniladridas/rag",
        "Bug Reports": "https://github.com/bniladridas/rag/issues",
    },
)
