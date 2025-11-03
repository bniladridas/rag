#!/bin/bash

set -e

echo "Running E2E tests locally..."

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run data collection (if API keys available)
if [ -n "$TMDB_API_KEY" ] && [ -n "$NASA_API_KEY" ]; then
  python -m src.rag.data_fetcher
else
  echo "API keys not available, skipping data collection"
fi

# Run e2e tests
PYTHONPATH=src python -m pytest tests/integration/test_e2e.py -v

# Test full application flow
echo -e "hello\ncalculate 2+3\nhelp\nexit" | python -m src.rag.__main__ | grep -E "(Hello|Result|This is an Agentic|Exiting)"

# Test TUI application flow
echo -e "hello\nexit" | PYTHONPATH=src FORCE_TUI=1 python -c "from src.rag.ui.tui import run_tui; run_tui()" | grep -E "(Query|Response|Goodbye)"

echo "E2E tests completed successfully."
