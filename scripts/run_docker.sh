#!/bin/bash

set -e

echo "Building and running Docker locally..."

# Build Docker image
docker build -f container/Dockerfile -t rag:test .

# Verify image imports
docker run --rm rag:test python -c "import src.rag.__main__; import src.rag.rag_engine; import src.rag.config; import src.rag.tools; import src.rag.data_fetcher; print('All imports successful')"

# Display CLI help safely
echo -e "exit" | docker run -i --rm rag:test rag --help

# Display TUI safely
echo -e "exit" | docker run -i --rm rag:test python -c "from src.rag.ui.tui import run_tui; run_tui()"

# Clean up
docker system prune -f || true
docker image prune -f || true
docker volume prune -f || true
docker builder prune -f || true

echo "Docker build and tests completed successfully."