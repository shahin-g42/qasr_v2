#!/usr/bin/env bash
# Export once, then launch one fresh-training process group on four nodes.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
cd "$PROJECT_DIR"
export GEMMA_SPEECH_ENV="${GEMMA_SPEECH_ENV:-$PROJECT_DIR/gemma_speech/.venv2}"
# An inherited resume config must not silently select the older base model.
export GQASR_CONFIG="${GQASR_FRESH_CONFIG:-$PROJECT_DIR/gemma_speech/gqasr/configs/04_encoder12_b120_fresh_4node.yaml}"
export GQASR_LAUNCH_NNODES=4 NNODES=4 NPROC_PER_NODE=8
export MASTER_ADDR="${MASTER_ADDR:-inception-H100-hpc-020}"
export MASTER_PORT="${MASTER_PORT:-29614}"
export GQASR_LOG_DIR="${GQASR_LOG_DIR:-/tmp/gqasr-${USER:-user}-logs}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CPU_THREADS="${GQASR_CPU_THREADS:-1}"
[[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'GQASR_CPU_THREADS must be positive.' >&2; exit 2; }
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS" NUMBA_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
if [[ "${1:-}" == export ]]; then
  [[ $# == 1 ]] || { echo "Usage: $0 export" >&2; exit 2; }
  SOURCE="${GQASR_EXPORT_CHECKPOINT:-$PROJECT_DIR/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500}"
  exec "$GEMMA_SPEECH_ENV/bin/python" -m gqasr.export_checkpoint \
    --checkpoint "$SOURCE" --config "$GQASR_CONFIG"
fi
[[ $# -ge 1 && $# -le 2 && "$1" =~ ^[0-3]$ ]] || {
  echo "Usage: $0 export | NODE_RANK [MASTER_ADDR]  (ranks 0 through 3)" >&2; exit 2;
}
[[ $# == 1 || "$2" != --* ]] || { echo 'Fresh training does not accept resume flags.' >&2; exit 2; }
echo "Fresh training: config=$GQASR_CONFIG optimizer=new update=0 logs=$GQASR_LOG_DIR"
exec bash "$PROJECT_DIR/scripts/run_gqasr_12node.sh" "$@" --stage encoder
