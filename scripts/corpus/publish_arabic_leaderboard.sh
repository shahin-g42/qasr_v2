#!/usr/bin/env bash
# Private static leaderboard. The Space destination is independent of the dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

export HF_SPACE_REPO="${HF_SPACE_REPO:-shahink/asr_ideal_leaderboard}"
export HF_SPACE_TITLE="${HF_SPACE_TITLE:-Arabic ASR Benchmark}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-audarai/asr_ideal_bench}"
# Prefer the local publication receipt; set a SHA explicitly if the receipt is elsewhere.
export HF_DATASET_REVISION="${HF_DATASET_REVISION:-}"
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2}"
export ARABIC_REF_ROOT="${ARABIC_REF_ROOT:-$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full}"
export ARABIC_REF_QASR_RUN="${ARABIC_REF_QASR_RUN-$ARABIC_REF_SOURCE_REPO/verification/qasr-stage02-arabic}"
export ARABIC_REF_GQASR_RUN="${ARABIC_REF_GQASR_RUN-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-final-streamed}"
export ARABIC_REF_PLAN="${ARABIC_REF_PLAN:-$ARABIC_REF_ROOT/plan}"
export ARABIC_REF_ANNOTATIONS_BUNDLE="${ARABIC_REF_ANNOTATIONS_BUNDLE:-$ARABIC_REF_ROOT/dataset-release-v4.4}"
export ARABIC_REF_ANNOTATIONS_ARCHIVE="${ARABIC_REF_ANNOTATIONS_ARCHIVE:-$ARABIC_REF_ROOT/arabic-reference-v4.4-cluster-upload.tar.gz}"
export ARABIC_REF_DATASET_RECEIPT="${ARABIC_REF_DATASET_RECEIPT:-$ARABIC_REF_ROOT/publication-v4.5-audio-001.json}"
export ARABIC_REF_BOARD_RELEASE_ID="${ARABIC_REF_BOARD_RELEASE_ID:-v4.6-board-001}"
export ARABIC_REF_SPACE_BUNDLE="${ARABIC_REF_SPACE_BUNDLE:-$ARABIC_REF_ROOT/leaderboard-$ARABIC_REF_BOARD_RELEASE_ID}"

PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src:$REPO/gemma_speech/gqasr/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
# all = verify published dataset binding, build, publish privately, verify uploaded files.
# build = offline preview; check = read-only Hub validation; verify = retry receipt audit.
exec "$PY" -m data_processing.arabic_reference.cluster_leaderboard "${1:-all}"
