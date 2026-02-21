#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=$(command -v python3 || true)
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 is required on Apple Silicon; please install it first."
  exit 1
fi

export PYTHONBREAKSYSTEMPACKAGES=1

fail_install() {
  echo "WARNING: $1 failed; run the script again with network access to finish."
}

echo "Creating '.venv315' via ${PYTHON_BIN}..."
"$PYTHON_BIN" -m venv .venv315 >/dev/null
source .venv315/bin/activate
"$PYTHON_BIN" -m pip install --break-system-packages --upgrade pip || fail_install "pip upgrade"
python -m pip install --break-system-packages -r requirements.txt || fail_install "pip install requirements"
python -m pip install --break-system-packages -e . || fail_install "pip install editable package"

echo "Pip packages installed. Checking PyTorch device options..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

echo "Running 'rag --version' to ensure the entry point is discoverable."
rag --version

echo "To run the TUI on Apple Silicon:"
echo "  source .venv315/bin/activate"
echo "  python -m rag.ui.tui"
