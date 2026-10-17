#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN="${GQASR_RUN_DIR:-$ROOT/gqasr/outputs/full_b40_frozen}"
python -m gqasr.batch_audit --run-config "$RUN/config.json" \
  --output "$ROOT/gqasr/verification/batch-audit-$(date -u +%Y%m%dT%H%M%SZ)" "$@"
