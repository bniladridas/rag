#!/bin/bash

# RAG Setup Script
# This script sets up the RAG assistant on a new computer

set -e  # Exit on error

echo "=== Setting up RAG Assistant ==="

# Get repo directory (where this script is located)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/venv"

# Create a starter .env (never overwrites an existing one)
if [ ! -f "$REPO_DIR/.env" ] && [ -f "$REPO_DIR/.env.example" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    echo "Created $REPO_DIR/.env from .env.example (fill in keys as needed)"
fi

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists. Skipping creation."
fi

# 2. Install dependencies
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install -r "$REPO_DIR/requirements.txt" -q

# 3. Install the package in editable mode
echo "Installing RAG package..."
pip install -e "$REPO_DIR" -q

# 4. Add to PATH in shell config
echo "Adding RAG to PATH..."

# Detect shell and config file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

# Check if PATH entry already exists
if ! grep -q "$VENV_DIR/bin" "$SHELL_CONFIG" 2>/dev/null; then
    echo "export PATH=\"$VENV_DIR/bin:\$PATH\"" >> "$SHELL_CONFIG"
    echo "Added RAG to $SHELL_CONFIG"
else
    echo "RAG already in $SHELL_CONFIG"
fi

# 5. Create helper scripts in /usr/local/bin
echo "Creating helper scripts in /usr/local/bin..."

for cmd in rag rag-tui rag-collect; do
    cat > "/tmp/${cmd}_wrapper" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
$cmd "\$@"
EOF
    chmod +x "/tmp/${cmd}_wrapper"
    if sudo mv "/tmp/${cmd}_wrapper" "/usr/local/bin/$cmd" 2>/dev/null; then
        echo "Created /usr/local/bin/$cmd"
    else
        echo "Could not create /usr/local/bin/$cmd (requires sudo)"
        echo "You can manually create it with:"
        echo "  sudo mv /tmp/${cmd}_wrapper /usr/local/bin/$cmd"
    fi
done

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "You can now run 'rag', 'rag-tui', and 'rag-collect' from any terminal."
echo ""
echo "To use immediately in this session, run:"
echo "  source $SHELL_CONFIG"
echo ""
echo "Or open a new terminal window."
echo ""
echo "Test with: rag --help"
