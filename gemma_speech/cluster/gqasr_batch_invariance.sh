#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m gqasr.batch_invariance \
  --comparison "${COMPARISON_DIR:-$ROOT/gqasr/verification/comparison-20261016T210052Z}" \
  --output "$ROOT/gqasr/verification/batch-invariance-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
