#!/usr/bin/env bash
# Fresh accuracy run, preserving the historical GQASR candidates and references.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ARABIC_REF_SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2}"
export ARABIC_REF_ROOT="${ARABIC_REF_ROOT:-$ARABIC_REF_SOURCE_REPO/verification/arabic-reference-v4.1-full}"
export GQASR_REFRESH_CHECKPOINT="${GQASR_REFRESH_CHECKPOINT:-$ARABIC_REF_SOURCE_REPO/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500}"
export PUBLIC_BENCH_ROOT="${PUBLIC_BENCH_ROOT:-$ARABIC_REF_ROOT/public-benchmarks-w8-b32-20261019-223610}"
export PUBLIC_MODELS="gqasr-encoder12-b80-step7500"
export PYTHON="${PYTHON:-$PWD/.venv-public-asr/bin/python}"
STAGE="${1:-all}"
if [[ "$STAGE" == all ]]; then
  bash scripts/corpus/benchmark_public_asr.sh check
  bash scripts/corpus/benchmark_public_asr.sh smoke
  exec bash scripts/corpus/benchmark_public_asr.sh run
fi
case "$STAGE" in
  check|smoke|run) exec bash scripts/corpus/benchmark_public_asr.sh "$STAGE" ;;
  *) echo "Usage: $0 {check|smoke|run|all}" >&2; exit 2 ;;
esac
