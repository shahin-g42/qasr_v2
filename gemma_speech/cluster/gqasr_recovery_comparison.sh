#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m gqasr.recovery_comparison --root "$ROOT" --gpus "${GPU_IDS:-0,1,2,3,4,5,6,7}" --workers-per-gpu "${WORKERS_PER_GPU:-4}" "$@"
