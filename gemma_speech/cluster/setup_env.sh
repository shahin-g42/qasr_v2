#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR="${GEMMA_SPEECH_ENV:-$ROOT/.venv}"
mkdir -p "$ROOT/logs"
exec > >(tee "$ROOT/logs/setup-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
"${PYTHON:-python3}" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$ROOT[dev]"
python -m pip check
python -m pip freeze > "$ROOT/logs/environment.lock.txt"
python - <<'PY'
import torch, transformers
print({'torch': str(torch.__version__), 'transformers': transformers.__version__,
       'cuda_runtime': torch.version.cuda, 'visible_gpus': torch.cuda.device_count()})
PY
