#!/usr/bin/env bash
# CPU-only transcript audit. Run once, without torchrun or Slurm.
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
cd "$AUD_REPO_ROOT"
exec "${AUD_PYTHON:-python}" -m audm.audit_transcripts \
  --config "${AUDFLASH_ADAPTER_CONFIG:-$AUD_REPO_ROOT/audm/configs/audflash_adapter_1node.yaml}" \
  --model "${AUDFLASH_MODEL:-$AUD_REPO_ROOT/models/AudFlash-Gemma4-v1}" "$@"
