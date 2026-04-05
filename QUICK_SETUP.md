# Quick RAG Setup

## Fastest Method (Copy-Paste)

Run these commands in order:

```bash
# 1. Clone and enter repo
git clone <repo-url> && cd rag

# 2. Run setup script (takes ~2-3 minutes)
bash setup_rag.sh

# 3. Open a new terminal or run:
source ~/.zshrc  # or ~/.bashrc

# 4. Test it
rag --help
```

## Alternative: Manual Setup

If you prefer manual control:

```bash
# 1. Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Add to PATH
echo 'export PATH="/path/to/rag/venv/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## One-Liner Setup

For the absolute fastest setup on a new Mac:

```bash
git clone <repo-url> && cd rag && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt -q && pip install -e . -q && echo 'export PATH="'$(pwd)'/venv/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc && rag --help
```

## Notes

- The setup script creates wrapper scripts in `/usr/local/bin` so `rag` works from anywhere
- Virtual environment is located at `./venv` in the repo directory
- All three commands (`rag`, `rag-tui --theme minimal`, `rag-collect`) will be available
