#!/bin/bash
# Development environment setup script for RAG Transformer

set -e

echo "Setting up RAG Transformer development environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Not in a virtual environment. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
    echo ""
fi

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Run initial checks
echo "Running initial checks..."
python -c "import rag; print('Package import successful')"
python -m pytest tests/test_cli.py -v --tb=short

echo ""
echo "Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  • Run tests: python -m pytest"
echo "  • Run linting: python scripts/lint.py"
echo "  • Start interactive mode: rag"
echo "  • Start TUI mode: rag-tui"
echo ""
