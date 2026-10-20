#!/usr/bin/env bash
# Optional CPU check: bash scripts/resume_gqasr_encoder.sh check
# Recovery: bash scripts/resume_gqasr_encoder.sh NODE_RANK [MASTER_ADDR]
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
export GQASR_LAUNCH_NNODES="${GQASR_LAUNCH_NNODES:-4}"
export NNODES="$GQASR_LAUNCH_NNODES"
export MASTER_ADDR="${MASTER_ADDR:-inception-H100-hpc-020}"
cd "$PROJECT_DIR"
export GQASR_CONFIG="${GQASR_CONFIG:-$PROJECT_DIR/gemma_speech/gqasr/configs/03_encoder12_b120_resume_4node.yaml}"
CHECKPOINT="${GQASR_RESUME_CHECKPOINT:-$PROJECT_DIR/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500}"
export GQASR_LOG_DIR="${GQASR_LOG_DIR:-/tmp/gqasr-${USER:-user}-logs}"
# Conda/base environments can already export OPENBLAS_NUM_THREADS=32. A fallback
# default leaves 64 data workers spawning thousands of threads on a 96-core node.
# Use one explicit budget for this launch, before Python/NumPy import.
CPU_THREADS="${GQASR_CPU_THREADS:-1}"
[[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'GQASR_CPU_THREADS must be positive.' >&2; exit 2; }
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"
export NUMBA_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
if [[ "${1:-}" == check ]]; then
  [[ $# == 1 ]] || { echo 'Usage: bash scripts/resume_gqasr_encoder.sh check' >&2; exit 2; }
  ENV_DIR="${GEMMA_SPEECH_ENV:-$PROJECT_DIR/gemma_speech/.venv}"
  export HF_HUB_OFFLINE=1
  exec "$ENV_DIR/bin/python" -m gqasr.resume --config "$GQASR_CONFIG" \
    --checkpoint "$CHECKPOINT" --allow-batch-change --allow-world-size-change
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -s "$CHECKPOINT/training_state.pt" && -s "$CHECKPOINT/model.safetensors" ]] || {
    echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 2;
  }
  mkdir -p "$GQASR_LOG_DIR"
fi
echo "Recovery: checkpoint=$CHECKPOINT config=$GQASR_CONFIG logs=$GQASR_LOG_DIR cpu_threads=$CPU_THREADS"
exec bash "$PROJECT_DIR/scripts/run_gqasr_12node.sh" "$@" \
  --stage encoder --resume "$CHECKPOINT" --allow-batch-change --allow-world-size-change
