#!/bin/bash

set -e

echo "Running CI tests locally..."

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install flake8 mypy rich pytest

# Run linting
python scripts/lint.py

# Run tests
python scripts/test.py

# Run additional checks
python setup.py check
python -c "import src.rag.__main__; import src.rag.rag_engine; import src.rag.config; import src.rag.tools; import src.rag.data_fetcher; print('All imports successful')"

echo "CI tests completed successfully."