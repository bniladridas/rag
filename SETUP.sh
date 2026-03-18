#!/bin/bash
# One-command RAG setup
# Usage: bash SETUP.sh

set -e

echo "=== RAG Setup ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create a starter .env (never overwrites an existing one)
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp ".env.example" ".env"
    echo "Created .env from .env.example (fill in keys as needed)"
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt -q
pip install -e . -q

# Add to shell config
SHELL_CONFIG="$HOME/.zshrc"
[ -f "$HOME/.bashrc" ] && SHELL_CONFIG="$HOME/.bashrc"

if ! grep -q "venv/bin" "$SHELL_CONFIG" 2>/dev/null; then
    echo "export PATH=\"$SCRIPT_DIR/venv/bin:\$PATH\"" >> "$SHELL_CONFIG"
    echo "Added to $SHELL_CONFIG"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Run 'source $SHELL_CONFIG' or open a new terminal."
echo "Then test with: rag --help"
