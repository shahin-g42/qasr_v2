#!/usr/bin/env bash
# Reserve one idle H100 for the complete serial comparison.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2}"
export ARABIC_REF_ROOT="${ARABIC_REF_ROOT:-$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full}"
export ARABIC_REF_PLAN="${ARABIC_REF_PLAN:-$ARABIC_REF_ROOT/plan}"
export PUBLIC_PREPARED_WORK="${PUBLIC_PREPARED_WORK:-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache}"
export PUBLIC_MODEL_CACHE="${PUBLIC_MODEL_CACHE:-$ARABIC_REF_ROOT/public-models}"
export PUBLIC_BENCH_ROOT="${PUBLIC_BENCH_ROOT:-$ARABIC_REF_ROOT/public-benchmarks-w8-b32-20261019-223610}"
export LATENCY_QASR_CHECKPOINT="${LATENCY_QASR_CHECKPOINT:-$ARABIC_REF_SOURCE_REPO/output/v81/full_12node}"
export GQASR_REFRESH_CHECKPOINT="${GQASR_REFRESH_CHECKPOINT:-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500}"
export LATENCY_MODELS="${LATENCY_MODELS:-qasr gqasr-encoder12-b80-step7500 qwen3-asr-0.6b qwen3-asr-1.7b cohere-arabic cohere-multilingual whisper-large-v3 jais-asr}"
export LATENCY_ROOT="${LATENCY_ROOT:-$ARABIC_REF_ROOT/latency-b1-c1-v1}"
export LATENCY_GPU="${LATENCY_GPU:-0}"
export LATENCY_PER_DATASET="${LATENCY_PER_DATASET:-100}"
export LATENCY_REPEATS="${LATENCY_REPEATS:-3}"
export LATENCY_WARMUP="${LATENCY_WARMUP:-12}"
export LATENCY_SEED="${LATENCY_SEED:-20261020}"
export PYTHONPATH="$REPO/src:$REPO/gemma_speech/gqasr/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="${PYTHON:-$REPO/.venv-public-asr/bin/python}"
STAGE="${1:-run}"
if [[ "$STAGE" == test ]]; then
  exec "$PY" -m pytest tests/test_asr_latency.py tests/test_public_asr_benchmark.py -q
fi
exec "$PY" -m data_processing.arabic_reference.latency_runner "$STAGE"
