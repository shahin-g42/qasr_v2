#!/usr/bin/env bash
# One-node reference pilot; reuse the Yodas3 corrector and saved ASR predictions.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
SOURCE_REPO="${ARABIC_REF_SOURCE_REPO:-$REPO}"
export PYTHONPATH="$REPO/src:$REPO/gemma_speech/gqasr/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-$SOURCE_REPO/gemma_speech/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="${PYTHON:-python3}"; fi
ROOT="${ARABIC_REF_ROOT:-$SOURCE_REPO/verification/arabic-reference-pilot-v4}"
QASR_RESULTS="${QASR_RESULTS:-$SOURCE_REPO/verification/qasr-stage02-arabic}"
GQASR_RESULTS="${GQASR_RESULTS:-$SOURCE_REPO/gemma_speech/gqasr/verification/arabic-final-streamed}"
AUDIO_ROOT="${ARABIC_REF_AUDIO_ROOT:-$SOURCE_REPO/gemma_speech/gqasr/verification/arabic-stream-cache/prepared/audio}"
LLM_URL="${LLM_URL:-http://localhost:8010}"
MODEL_NAME="${LLM_MODEL_NAME:-corrector}"
CHECKPOINT_ARGS=()
if [[ -n "${ARABIC_REF_CHECKPOINT:-}" ]]; then CHECKPOINT_ARGS=(--checkpoint "$ARABIC_REF_CHECKPOINT"); fi
COMMAND="${1:?Expected plan|plan-full|doctor|smoke|run|status|review-pack|import-reviews|score|export-dataset|leaderboard|publish-dataset|publish-leaderboard|test|llm-print|llm-start}"
shift
CLI=("$PY" -m data_processing.arabic_reference)
case "$COMMAND" in
  plan|plan-full)
    EXTRA=(); if [[ "$COMMAND" == plan-full ]]; then EXTRA=(--full-suite); fi
    exec "${CLI[@]}" prepare --prediction "qasr=$QASR_RESULTS/predictions-arabic.jsonl" \
      --prediction "gqasr=$GQASR_RESULTS/predictions-arabic.jsonl" --output "$ROOT/plan" ${EXTRA[@]+"${EXTRA[@]}"} "$@" ;;
  doctor)
    exec "${CLI[@]}" doctor --plan "$ROOT/plan" --audio-root "$AUDIO_ROOT" \
      --url "$LLM_URL" --model "$MODEL_NAME" ${CHECKPOINT_ARGS[@]+"${CHECKPOINT_ARGS[@]}"} "$@" ;;
  smoke|run)
    EXTRA=(); DEST="$ROOT/proposals"
    if [[ "$COMMAND" == smoke ]]; then EXTRA=(--smoke); DEST="$ROOT/smoke-v4"; fi
    exec "${CLI[@]}" propose --plan "$ROOT/plan" --output "$DEST" \
      --url "$LLM_URL" --model "$MODEL_NAME" ${CHECKPOINT_ARGS[@]+"${CHECKPOINT_ARGS[@]}"} \
      --revision "${LLM_REVISION:-unrecorded}" --concurrency "${REF_CONCURRENCY:-16}" \
      --max-tokens "${REF_MAX_TOKENS:-4096}" ${EXTRA[@]+"${EXTRA[@]}"} "$@" ;;
  review-pack)
    exec "${CLI[@]}" review-pack --plan "$ROOT/plan" --proposals "$ROOT/proposals" \
      --audio-root "$AUDIO_ROOT" --output "$ROOT/listening" "$@" ;;
  import-reviews)
    exec "${CLI[@]}" import-reviews --plan "$ROOT/plan" --output "$ROOT/reviewed-v1.jsonl" "$@" ;;
  score)
    exec "${CLI[@]}" score --plan "$ROOT/plan" --reviewed "$ROOT/reviewed-v1.jsonl" \
      --output "$ROOT/scores-v1.json" "$@" ;;
  export-dataset|leaderboard)
    EXTRA=()
    if [[ -f "$ROOT/reviewed-v1.jsonl" ]]; then EXTRA+=(--reviewed "$ROOT/reviewed-v1.jsonl"); fi
    if [[ "$COMMAND" == export-dataset ]]; then
      if [[ -f "$ROOT/proposals/run.json" ]]; then EXTRA+=(--proposals "$ROOT/proposals"); fi
      exec "${CLI[@]}" export-dataset --plan "$ROOT/plan" \
        --output "${ARABIC_REF_DATASET_BUNDLE:-$ROOT/dataset-release}" \
        --release-id "${ARABIC_REF_RELEASE_ID:-v4-draft-001}" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
    else
      exec "${CLI[@]}" leaderboard --plan "$ROOT/plan" \
        --output "${ARABIC_REF_SPACE_BUNDLE:-$ROOT/leaderboard-release}" \
        --release-id "${ARABIC_REF_RELEASE_ID:-v4-board-001}" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
    fi ;;
  publish-dataset)
    exec "${CLI[@]}" publish --bundle "${ARABIC_REF_DATASET_BUNDLE:-$ROOT/dataset-release}" \
      --repo-id "${HF_DATASET_REPO:?Set HF_DATASET_REPO to your private dataset ID}" "$@" ;;
  publish-leaderboard)
    exec "${CLI[@]}" publish --bundle "${ARABIC_REF_SPACE_BUNDLE:-$ROOT/leaderboard-release}" \
      --repo-id "${HF_SPACE_REPO:?Set HF_SPACE_REPO to your private Space ID}" "$@" ;;
  status)
    EXTRA=(); if [[ -f "$ROOT/proposals/run.json" ]]; then EXTRA=(--proposals "$ROOT/proposals"); fi
    exec "${CLI[@]}" status --plan "$ROOT/plan" ${EXTRA[@]+"${EXTRA[@]}"} "$@" ;;
  test) exec "$PY" -m pytest tests/test_arabic_reference.py "$@" ;;
  llm-print|llm-start)
    exec bash scripts/corpus/serve_yodas3_corrector.sh "${COMMAND#llm-}" "$@" ;;
  *) echo "Unknown stage: $COMMAND" >&2; exit 1 ;;
esac
