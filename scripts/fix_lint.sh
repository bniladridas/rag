#!/bin/bash

set -e

echo "Fixing linting issues..."

# Install fixers if not present
pip install --quiet black isort autoflake

# Fix imports: remove unused, sort
autoflake --remove-all-unused-imports --recursive --in-place src/ tests/ --exclude=__pycache__
isort src/ tests/

# Format code
black src/ tests/ --line-length=100

echo "Linting fixes applied."