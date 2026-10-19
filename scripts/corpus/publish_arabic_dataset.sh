#!/usr/bin/env bash
# Private Arabic audio dataset release. Override these settings with environment variables.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

# Hugging Face destination and presentation.
export HF_OWNER="${HF_OWNER:-shahink}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-asr_bench}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-$HF_OWNER/$HF_DATASET_NAME}"
export HF_DATASET_TITLE="${HF_DATASET_TITLE:-Arabic ASR Bench · Enhanced References}"
# Dataset configurations to include; each keeps its canonical source name.
export REF_DATASETS="${REF_DATASETS:-sada commonvoice masc_clean masc_noisy mgb2 casablanca}"

# Cluster input paths and a new immutable output version.
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2}"
export ARABIC_REF_ROOT="${ARABIC_REF_ROOT:-$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full}"
export ARABIC_REF_ANNOTATIONS_BUNDLE="${ARABIC_REF_ANNOTATIONS_BUNDLE:-$ARABIC_REF_ROOT/dataset-release-v4.4}"
export ARABIC_REF_AUDIO_ROOT="${ARABIC_REF_AUDIO_ROOT:-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache/prepared/audio}"
export ARABIC_REF_RELEASE_ID="${ARABIC_REF_RELEASE_ID:-v4.5-audio-001}"
export ARABIC_REF_DATASET_BUNDLE="${ARABIC_REF_DATASET_BUNDLE:-$ARABIC_REF_ROOT/dataset-release-$ARABIC_REF_RELEASE_ID}"
export REF_AUDIO_WORKERS="${REF_AUDIO_WORKERS:-90}"
export REF_AUDIO_SHARD_MIB="${REF_AUDIO_SHARD_MIB:-256}"

# Use the active cluster environment. No GPUs or running corrector are needed.
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src:$REPO/gemma_speech/gqasr/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec "$PY" -m data_processing.arabic_reference.cluster_release "${1:-all}"
