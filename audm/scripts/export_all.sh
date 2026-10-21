#!/usr/bin/env bash
set -euo pipefail
AUD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$AUD_ROOT/audm/src:$AUD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
: "${QASR_CHECKPOINT:?Set QASR_CHECKPOINT to your trained QASR directory}"
: "${GQASR_CHECKPOINT:?Set GQASR_CHECKPOINT to your trained GQASR directory}"
: "${AUD_EXPORT_ROOT:?Set AUD_EXPORT_ROOT to a NEW destination parent}"
AUD_PYTHON="${AUD_PYTHON:-python}"

# One CPU process per export. No torchrun: each artifact is built once.
"$AUD_PYTHON" -m audm export --variant audpro --qasr-checkpoint "$QASR_CHECKPOINT" --output "$AUD_EXPORT_ROOT/AudPro"
"$AUD_PYTHON" -m audm export --variant audturbo --qasr-checkpoint "$QASR_CHECKPOINT" --output "$AUD_EXPORT_ROOT/AudTurbo" \
  --encoder-source "${AUD_CAPTIONER_SOURCE:-Qwen/Qwen3-Omni-30B-A3B-Captioner}" --encoder-revision "${AUD_CAPTIONER_REVISION:-main}"
"$AUD_PYTHON" -m audm export --variant audflash --gqasr-checkpoint "$GQASR_CHECKPOINT" --output "$AUD_EXPORT_ROOT/AudFlash" \
  --encoder-source "${AUD_GEMMA_SOURCE:-google/gemma-4-E2B-it}" --encoder-revision "${AUD_GEMMA_REVISION:-main}"

# Each exported model fits on one H100; verification is intentionally sequential.
for AUD_VARIANT in AudPro AudTurbo AudFlash; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$AUD_PYTHON" -m audm verify \
    "$AUD_EXPORT_ROOT/$AUD_VARIANT" --device "${AUD_VERIFY_DEVICE:-cuda:0}"
done
