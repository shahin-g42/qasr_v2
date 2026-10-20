#!/usr/bin/env bash
# Run on each training node, or use --hosts-file to reach actual machines via SSH.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GQASR_CLEANUP_PYTHON:-$PROJECT_DIR/gemma_speech/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
exec "$PYTHON" "$PROJECT_DIR/gemma_speech/gqasr/src/gqasr/cleanup.py" \
  --project-root "$PROJECT_DIR" "$@"
