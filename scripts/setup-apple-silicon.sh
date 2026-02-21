#!/usr/bin/env bash
set -euo pipefail

# Helper for macOS Apple Silicon setup. This mirrors the workflow steps locally.

python -m venv .venv315 >/dev/null
source .venv315/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Pip packages installed. Checking PyTorch device options:"
python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

echo "Running `rag --version` to ensure the entry point is discoverable."
rag --version

echo "To run the TUI on Apple Silicon:"
echo "  source .venv315/bin/activate"
echo "  python -m rag.ui.tui"
