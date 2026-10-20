#!/usr/bin/env bash
# Run once on the shared checkout, not simultaneously on all nodes.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR="${GQASR_TRAIN_ENV:-$ROOT/.venv2}"
PYTHON_BIN="${PYTHON:-python3}"
[[ ! -e "$ENV_DIR" ]] || { echo "Refusing to overwrite existing environment: $ENV_DIR. Use a fresh GQASR_TRAIN_ENV path." >&2; exit 2; }
"$PYTHON_BIN" - <<'PY'
import platform, sys, sysconfig
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Cluster Linux x86_64 required'
assert (3, 11) <= sys.version_info[:2] < (3, 15), 'Use Python 3.11–3.14 via PYTHON=/path/to/python'
assert not sysconfig.get_config_var('Py_GIL_DISABLED'), 'Use standard CPython, not a free-threaded interpreter'
PY
mkdir -p "$ROOT/logs"
exec > >(tee "$ROOT/logs/setup-training-env2-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
# Do not inherit external Python packages or user site packages from Conda.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
# CUDA 12.9 uses minor-version compatibility on driver 570; do not select CUDA 13 wheels.
# Resolve newest stable compatible versions; freeze the result for this run.
"$PY" -m pip install --upgrade --only-binary=:all: 'torch>=2.13,<3' --index-url https://download.pytorch.org/whl/cu129
"$PY" - <<'PY' > "$ENV_DIR/torch-constraint.txt"
from importlib.metadata import version
import torch
assert torch.version.cuda == '12.9', f'Expected CUDA 12.9 build, got {torch.version.cuda}'
print('torch==' + version('torch'))
PY
"$PY" -m pip install --upgrade --only-binary=:all: -c "$ENV_DIR/torch-constraint.txt" -e "$ROOT/gqasr"
"$PY" -m pip check
"$PY" -m gqasr.env_check --write-profile
"$PY" -m pip freeze --exclude-editable > "$ENV_DIR/requirements.lock.txt"
"$PY" -m pip inspect > "$ENV_DIR/pip-inspect.json"
printf '\nTraining environment ready: %s\nSelect it on every node with:\nexport GEMMA_SPEECH_ENV="%s"\n' "$ENV_DIR" "$ENV_DIR"
echo 'No model downloads or cluster training launched. The original .venv is untouched.'
