#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-/tmp/pyinstaller-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

cd "$ROOT_DIR"

build_app() {
    local app_name="$1"
    local entrypoint="$2"
    local dmg_name="$3"

    pyinstaller --noconfirm --clean --onedir --name "$app_name" --windowed "$entrypoint"
    APP_BUNDLE_PATH="$ROOT_DIR/dist/$app_name.app" \
        dmgbuild -s "$ROOT_DIR/scripts/dmg_settings.py" "$app_name" "$ROOT_DIR/$dmg_name"
}

build_app "RAG Minimal TUI" "src/rag/ui/minimal_tui.py" "RAG_Minimal_TUI.dmg"
build_app "RAG Transformer" "src/rag/__main__.py" "RAG_Transformer.dmg"
