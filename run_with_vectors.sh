#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_SERVER=0
SERVER_ARGS=()

cleanup() {
  set +e
  trap - INT TERM
  echo ""
  echo "Interrupted. Stopping..."
  kill -- -$$ >/dev/null 2>&1
  exit 130
}
trap cleanup INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-mermaid)
      export CHATGPT_HISTORY_RENDER_MERMAID=0
      shift
      ;;
    --stream)
      export CHATGPT_HISTORY_STREAM=1
      shift
      ;;
    --count-total)
      export CHATGPT_HISTORY_COUNT_TOTAL=1
      shift
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --workers requires a number argument" >&2
        exit 2
      fi
      export CHATGPT_HISTORY_WORKERS="$2"
      shift 2
      ;;
    --chrome-path)
      if [[ $# -lt 2 ]]; then
        echo "Error: --chrome-path requires a path argument" >&2
        exit 2
      fi
      export CHATGPT_HISTORY_CHROME_PATH="$2"
      shift 2
      ;;
    --serve)
      RUN_SERVER=1
      shift
      ;;
    --open)
      RUN_SERVER=1
      SERVER_ARGS+=("--open")
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./run_with_vectors.sh [options] [--serve] [--open]

Options:
  --no-mermaid           Disable Mermaid diagram rendering (faster, fewer deps)
  --stream               Stream conversations.json (low-memory mode)
  --count-total          In --stream mode, pre-count conversations for progress (reads file twice)
  --workers <n>          Convert conversations in parallel (I/O bound; try 4-8)
  --chrome-path <path>   Use a specific Chrome/Chromium binary for pyppeteer
  --serve                Start the local web viewer after conversion
  --open                 Open the viewer in your default browser (implies --serve)
EOF
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      echo "Run ./run_with_vectors.sh --help for usage." >&2
      exit 2
      ;;
  esac
done

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

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: venv python not found at $VENV_PY" >&2
  exit 1
fi

echo "Installing dependencies from requirements.txt"
"$VENV_PY" -m pip install -r requirements.txt

echo "Installing sentence-transformers for vector search"
"$VENV_PY" -m pip install sentence-transformers

if [[ ! -f "chatgpt-history-source/conversations.json" ]]; then
  echo "Warning: expected chatgpt-history-source/conversations.json but it was not found." >&2
  echo "Extract your ChatGPT export into chatgpt-history-source/ and re-run." >&2
fi

echo "Running converter"
"$VENV_PY" converter.py

if [[ "$RUN_SERVER" -eq 1 ]]; then
  exec "$VENV_PY" serve.py "${SERVER_ARGS[@]}"
fi
