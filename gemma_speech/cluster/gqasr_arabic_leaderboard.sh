#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE="${1:?Stages: test source-smoke stream-check stream-smoke stream-prepare doctor references download-casablanca preflight prepare smoke tune run}"
shift
ENV_DIR="${GEMMA_SPEECH_ENV:-$ROOT/.venv}"
case "${GQASR_BENCH_INPUT:-local}" in
  local) DEFAULT_CONFIG="$ROOT/gqasr/configs/arabic_leaderboard_6set.yaml" ;;
  stream) DEFAULT_CONFIG="$ROOT/gqasr/configs/arabic_leaderboard_streaming.yaml" ;;
  *) echo "GQASR_BENCH_INPUT must be local or stream" >&2; exit 2 ;;
esac
CONFIG="${GQASR_BENCH_CONFIG:-$DEFAULT_CONFIG}"
LABEL="${GQASR_BENCH_MODEL_NAME:-GQASR}"
source "$ENV_DIR/bin/activate"
# Dedicated full-node benchmark: keep doctor and worker selection consistent even
# when activation inherits an earlier single-GPU training/debug session.
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
mkdir -p "$ROOT/logs"
exec > >(tee "$ROOT/logs/${LABEL}-arabic-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GQASR_BENCH_EXPECTED_GPUS="${GQASR_BENCH_EXPECTED_GPUS:-8}"
BENCH="$ENV_DIR/bin/gqasr-arabic-benchmark"

echo "$LABEL Arabic benchmark stage=$STAGE host=$(hostname) time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
case "$STAGE" in
  test)
    export HF_HUB_OFFLINE=1
    "$ENV_DIR/bin/python" -m pytest "$ROOT/gqasr/tests/test_arabic_benchmark.py" \
      "$ROOT/gqasr/tests/test_arabic_benchmark_streaming.py" "$@"
    ;;
  qasr-test)
    export HF_HUB_OFFLINE=1
    "$ENV_DIR/bin/python" -m pytest "$ROOT/gqasr/tests/test_arabic_benchmark_qasr.py" "$@"
    ;;
  source-smoke)
    export HF_HUB_OFFLINE=0
    "$BENCH" source-smoke --output "${GQASR_BENCH_SOURCE_SMOKE_OUTPUT:-$ROOT/gqasr/verification/arabic-source-smoke}" "$@"
    ;;
  doctor)
    export HF_HUB_OFFLINE=1
    if [[ "$LABEL" == QASR ]]; then
      "$BENCH" doctor --config "$CONFIG" --workers-per-gpu "${WORKERS_PER_GPU:-4}" "$@"
    else
      "$BENCH" doctor "$@"
    fi
    ;;
  stream-check|stream-smoke|stream-prepare)
    export HF_HUB_OFFLINE=0
    "$BENCH" "$STAGE" --work-dir "${GQASR_BENCH_WORK:?Set GQASR_BENCH_WORK for the prepared test cache}" \
      --workers "${STREAM_WORKERS:-16}" "$@"
    ;;
  references)
    export HF_HUB_OFFLINE=0
    "$BENCH" references --config "$CONFIG" "$@"
    ;;
  download-casablanca)
    export HF_HUB_OFFLINE=0
    "$BENCH" download-casablanca --config "$CONFIG" --workers "${DOWNLOAD_WORKERS:-8}" "$@"
    ;;
  preflight)
    export HF_HUB_OFFLINE=1
    "$BENCH" preflight --config "$CONFIG" "$@"
    ;;
  prepare)
    export HF_HUB_OFFLINE=1
    "$BENCH" prepare --config "$CONFIG" --workers "${PREPARE_WORKERS:-96}" "$@"
    ;;
  smoke)
    export HF_HUB_OFFLINE=1
    "$BENCH" run --config "$CONFIG" --gpus "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
      --workers-per-gpu "${WORKERS_PER_GPU:-11}" --batch-size "${BATCH_SIZE:-1}" \
      --limit-per-dataset "${SMOKE_PER_DATASET:-128}" \
      --modes "${DECODE_MODES:-arabic,auto}" \
      --output "${GQASR_BENCH_SMOKE_OUTPUT:-${GQASR_BENCH_OUTPUT}-smoke}" "$@"
    ;;
  tune)
    export HF_HUB_OFFLINE=1
    "$BENCH" tune --config "$CONFIG" --gpus "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
      --workers-per-gpu "${WORKERS_PER_GPU:-11}" \
      --limit-per-dataset "${TUNE_PER_DATASET:-64}" \
      --output "${GQASR_BENCH_TUNING_OUTPUT:-${GQASR_BENCH_OUTPUT}-tuning}" "$@"
    ;;
  run)
    export HF_HUB_OFFLINE=1
    "$BENCH" run --config "$CONFIG" --gpus "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
      --workers-per-gpu "${WORKERS_PER_GPU:-11}" --batch-size "${BATCH_SIZE:-1}" \
      --modes "${DECODE_MODES:-arabic,auto}" "$@"
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
