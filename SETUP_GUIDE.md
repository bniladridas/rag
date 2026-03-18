# RAG Setup Guide

## Quick Start (Recommended)

**Option 1: Automated Script (Easiest)**
```bash
# 1. Clone the repository
git clone <repo-url> && cd rag

# 2. Run the setup script (no sudo required)
bash setup_rag.sh

# 3. Open a new terminal or run:
source ~/.zshrc  # or ~/.bashrc if using bash

# 4. Test it
rag --help
```

**Option 2: One-Liner Setup**
```bash
git clone <repo-url> && cd rag && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -q && pip install -e . -q && echo 'export PATH="'$(pwd)'/venv/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc && rag --help
```

## Manual Setup

If you prefer manual control:

```bash
# 1. Clone the repository
git clone <repo-url>
cd rag

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install RAG in editable mode
pip install -e .

# 6. Add to your shell's PATH
echo 'export PATH="/path/to/rag/venv/bin:$PATH"' >> ~/.zshrc
# Replace /path/to/rag with the actual path, e.g., /Users/yourname/rag

# 7. Apply the changes
source ~/.zshrc

# 8. Test it
rag --help
```

## Available Commands

Once installed, you can use these commands from any terminal:

- `rag` - Main CLI interface (interactive mode)
- `rag --query "your question"` - Ask a single question
- `rag-tui` - Text-based user interface (TUI)
- `rag-collect` - Data collection tool

## Troubleshooting

### Command not found after setup
Make sure you've sourced your shell config or opened a new terminal:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### Virtual environment issues
If you get import errors, reactivate the venv:
```bash
source venv/bin/activate
```

### Updating dependencies
To update to the latest dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
pip install -e . --upgrade
```

## What the Setup Does

1. **Creates a virtual environment** (`./venv`) to isolate dependencies
2. **Installs all required packages** from `requirements.txt`
3. **Installs RAG in editable mode** so code changes are immediately available
4. **Adds the venv bin directory to your PATH** so commands work from anywhere
5. **Creates wrapper scripts** for easy command access

## File Locations

- Virtual environment: `./venv/` (in the repo directory)
- Shell config: `~/.zshrc` (zsh) or `~/.bashrc` (bash)
- Commands: Available via PATH after setup
