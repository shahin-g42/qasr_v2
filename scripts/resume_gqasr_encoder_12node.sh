#!/usr/bin/env bash
# CPU check once: bash scripts/resume_gqasr_encoder_12node.sh check
# Recovery: bash scripts/resume_gqasr_encoder_12node.sh NODE_RANK [MASTER_ADDR]
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_DIR
cd "$PROJECT_DIR"
export GQASR_CONFIG="${GQASR_CONFIG:-$PROJECT_DIR/gemma_speech/gqasr/configs/03_encoder12_b20_acc4_12node.yaml}"
CHECKPOINT="${GQASR_RESUME_CHECKPOINT:-$PROJECT_DIR/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500}"
export GQASR_LOG_DIR="${GQASR_LOG_DIR:-/tmp/gqasr-${USER:-user}-logs}"
# A fresh, CPU-only server preloads the data imports once per GPU process,
# instead of every one of the 64 data workers importing from Lustre independently.
export GQASR_WORKER_START_METHOD="${GQASR_WORKER_START_METHOD:-forkserver}"
[[ "$GQASR_WORKER_START_METHOD" == forkserver || "$GQASR_WORKER_START_METHOD" == spawn ]] || {
  echo 'GQASR_WORKER_START_METHOD must be forkserver or spawn.' >&2; exit 2;
}
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
# Keep diagnostic dumps on node-local storage if shared storage stops responding.
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_FR_DUMP_TEMP_FILE="${TORCH_FR_DUMP_TEMP_FILE:-${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-$GQASR_LOG_DIR/nccl_trace_rank_}}"
if [[ "${1:-}" == check ]]; then
  [[ $# == 1 ]] || { echo 'Usage: bash scripts/resume_gqasr_encoder_12node.sh check' >&2; exit 2; }
  ENV_DIR="${GEMMA_SPEECH_ENV:-$PROJECT_DIR/gemma_speech/.venv}"
  export HF_HUB_OFFLINE=1
  exec "$ENV_DIR/bin/python" -m gqasr.resume --config "$GQASR_CONFIG" \
    --checkpoint "$CHECKPOINT" --allow-microbatch-change
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -s "$CHECKPOINT/training_state.pt" && -s "$CHECKPOINT/model.safetensors" ]] || {
    echo "Incomplete checkpoint: $CHECKPOINT; run the check command first." >&2; exit 2;
  }
  mkdir -p "$GQASR_LOG_DIR"
fi
echo "Recovery: checkpoint=$CHECKPOINT config=$GQASR_CONFIG logs=$GQASR_LOG_DIR cpu_threads=$CPU_THREADS workers_start=$GQASR_WORKER_START_METHOD"
exec bash "$PROJECT_DIR/scripts/run_gqasr_12node.sh" "$@" \
  --stage encoder --resume "$CHECKPOINT" --allow-microbatch-change
