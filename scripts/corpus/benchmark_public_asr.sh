#!/usr/bin/env bash
# Benchmark public checkpoints on the same frozen audio; publish only completed runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
STAGE="${1:-all}"
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2}"
export ARABIC_REF_ROOT="${ARABIC_REF_ROOT:-$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full}"
export ARABIC_REF_PLAN="${ARABIC_REF_PLAN:-$ARABIC_REF_ROOT/plan}"
export ARABIC_REF_ANNOTATIONS_BUNDLE="${ARABIC_REF_ANNOTATIONS_BUNDLE:-$ARABIC_REF_ROOT/dataset-release-v4.4}"
export ARABIC_REF_ANNOTATIONS_ARCHIVE="${ARABIC_REF_ANNOTATIONS_ARCHIVE:-$ARABIC_REF_ROOT/arabic-reference-v4.4-cluster-upload.tar.gz}"
export ARABIC_REF_DATASET_RECEIPT="${ARABIC_REF_DATASET_RECEIPT:-$ARABIC_REF_ROOT/publication-v4.5-audio-001.json}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-audarai/asr_ideal_bench}"
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-}"
export HF_SPACE_REPO="${HF_SPACE_REPO:-shahink/asr_ideal_leaderboard}"
export HF_SPACE_TITLE="${HF_SPACE_TITLE:-Arabic ASR Benchmark}"
export PUBLIC_MODELS="${PUBLIC_MODELS:-qwen3-asr-0.6b qwen3-asr-1.7b cohere-arabic cohere-multilingual whisper-large-v3}"
export PUBLIC_MODEL_CACHE="${PUBLIC_MODEL_CACHE:-$ARABIC_REF_ROOT/public-models}"
export PUBLIC_BENCH_ROOT="${PUBLIC_BENCH_ROOT:-$ARABIC_REF_ROOT/public-benchmarks-v1}"
export PUBLIC_PREPARED_WORK="${PUBLIC_PREPARED_WORK:-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache}"
# Independent replicas, eight per H100 by default; ordinary WORKERS_PER_GPU does not override this.
export PUBLIC_GPUS="${PUBLIC_GPUS:-0,1,2,3,4,5,6,7}"
export PUBLIC_WORKERS_PER_GPU="${PUBLIC_WORKERS_PER_GPU:-8}"
export PUBLIC_BATCH_SIZE="${PUBLIC_BATCH_SIZE:-32}"
# Start ready replicas immediately; 'all' permits every requested replica to load together.
export PUBLIC_MODEL_LOAD_CONCURRENCY="${PUBLIC_MODEL_LOAD_CONCURRENCY:-all}"
BOARD_DEFAULT="v4.7-public-001"
if [[ "$STAGE" == rescore ]]; then BOARD_DEFAULT="v4.7-existing-001"; fi
export ARABIC_REF_BOARD_RELEASE_ID="${ARABIC_REF_BOARD_RELEASE_ID:-$BOARD_DEFAULT}"
export ARABIC_REF_SPACE_BUNDLE="${ARABIC_REF_SPACE_BUNDLE:-$ARABIC_REF_ROOT/leaderboard-$ARABIC_REF_BOARD_RELEASE_ID}"
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src:$REPO/gemma_speech/gqasr/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# The download stage needs network access. Worker subprocesses enforce offline loading.
if [[ "$STAGE" == download || "$STAGE" == publish || "$STAGE" == all ]]; then export HF_HUB_OFFLINE=0; fi
exec "$PY" -m data_processing.arabic_reference.public_benchmark "$STAGE"
