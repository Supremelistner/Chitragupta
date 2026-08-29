#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chitragupta — WSL install helper
# -----------------------------------------------------------------------------
# Why this script exists:
#   The project lives on /mnt/c (a 9P mount from Windows). `python3 -m venv`
#   on /mnt/c completes but the venv is unreliable for `pip install` — pip
#   gets stuck in 'D' (uninterruptible disk-wait) state because 9P is very
#   slow for thousands of small file operations. Installing under /tmp works
#   but /tmp is wiped on every WSL reboot, so we put the venv in
#   $HOME/.local/chitragupta-wsl-venv (persistent, on the native ext4 FS).
#
# `activate.py` auto-detects the venv via three paths:
#   1. $CHITRAGUPTA_VENV env var (if set)
#   2. .venv-wsl/  inside the project (a symlink we create here)
#   3. $HOME/.local/chitragupta-wsl-venv/  (the canonical install location)
#
# Usage (from a WSL shell, inside the project):
#     ./scripts/install-wsl.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# Strip stray Windows line endings (\r) so bash can run on files edited
# from the Windows side. This is a no-op on LF-only files.
if grep -q $'\r' "$0"; then
    TMP="$(mktemp)"; tr -d '\r' < "$0" > "$TMP"; mv "$TMP" "$0"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_SRC="$HOME/.local/chitragupta-wsl-venv"
VENV_LINK="$PROJECT_DIR/.venv-wsl"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: $REQUIREMENTS not found" >&2
    exit 1
fi

echo "=== Removing old venv ==="
rm -rf "$VENV_SRC"

echo "=== Creating venv on native WSL FS ($VENV_SRC) ==="
mkdir -p "$(dirname "$VENV_SRC")"
python3 -m venv "$VENV_SRC"

# shellcheck disable=SC1091
source "$VENV_SRC/bin/activate"

echo "=== Upgrading pip / wheel / setuptools ==="
python -m pip install --upgrade pip wheel setuptools

echo "=== Installing project dependencies from requirements.txt ==="
python -m pip install -r "$REQUIREMENTS"

echo "=== Verifying orchestrator can import its deps ==="
PYTHONPATH="$PROJECT_DIR/src" python -c "import requests, fastapi, uvicorn; print('OK', requests.__version__, fastapi.__version__, uvicorn.__version__)"

# Try to symlink the project .venv-wsl/ to the venv. This is a hint for
# `activate.py` and for editors/linters, but not required.
if [ -d "$PROJECT_DIR" ]; then
    echo "=== Linking $VENV_LINK -> $VENV_SRC ==="
    rm -rf "$VENV_LINK" 2>/dev/null || true
    ln -sfn "$VENV_SRC" "$VENV_LINK" 2>/dev/null || \
        echo "   (could not create symlink at $VENV_LINK — that is fine)"
fi

echo ""
echo "Done."
echo "  venv:    $VENV_SRC"
echo "  run:     python activate.py --bg"
echo "  (or:    CHITRAGUPTA_VENV=\"$VENV_SRC\" python activate.py --bg)"

