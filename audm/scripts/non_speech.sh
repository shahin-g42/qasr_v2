#!/usr/bin/env bash
# Cluster-only corpus preparation; no Slurm, GPU or distributed launch required.
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
AUD_NON_SPEECH_ROOT="${AUD_NON_SPEECH_ROOT:-/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/aud-data/non-speech-v1}"
AUD_PYTHON="${AUD_PYTHON:-python}"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  download)
    exec "$AUD_PYTHON" -m audm.prepare_non_speech download --root "$AUD_NON_SPEECH_ROOT" "$@"
    ;;
  prepare)
    exec "$AUD_PYTHON" -m audm.prepare_non_speech prepare --root "$AUD_NON_SPEECH_ROOT" \
      --output "$AUD_NON_SPEECH_ROOT/candidates-v1" "$@"
    ;;
  finalize)
    exec "$AUD_PYTHON" -m audm.prepare_non_speech finalize \
      --candidates "$AUD_NON_SPEECH_ROOT/candidates-v1/candidates.jsonl" \
      --decisions "$AUD_NON_SPEECH_ROOT/decisions.jsonl" \
      --output "$AUD_NON_SPEECH_ROOT/corpus-v1" "$@"
    ;;
  help|-h|--help)
    printf '%s\n' 'Usage: bash audm/scripts/non_speech.sh download|prepare|finalize [options]' \
      "Data root: $AUD_NON_SPEECH_ROOT" 'See audm/NON_SPEECH_DATA.md for listening review and training setup.'
    ;;
  *) printf 'Unknown action: %s\n' "$ACTION" >&2; exit 2 ;;
esac
