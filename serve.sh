#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cleanup() {
  set +e
  trap - INT TERM
  echo ""
  echo "Interrupted. Stopping..."
  kill -- -$$ >/dev/null 2>&1
  exit 130
}
trap cleanup INT TERM

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PY="$VENV_DIR/bin/python"
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  fi

  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python not found. Install Python 3.7+ and retry." >&2
    exit 1
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtualenv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  PY="$VENV_DIR/bin/python"
fi

exec "$PY" serve.py "$@"

