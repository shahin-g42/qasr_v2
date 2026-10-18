#!/usr/bin/env bash
# 11 local correctors + 2 shared ASR endpoints; additive manifests/indexes.
# See the Yodas3 section of scripts/corpus/README.md for the full sequence.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-yodas3_v1}"
ROOT="${ROOT:-$REPO/asr_cleaned_manifests/$RUN_ID}"
CS_ROOT="${CS_ROOT:-$REPO/code_switch_indexes/cs_$RUN_ID}"
PY="${PYTHON:-python3}"
QASR_CONFIG="${QASR_CONFIG:-configs/v8.1/02_full_12node.yaml}"
GQASR_CONFIG="${GQASR_CONFIG:-gemma_speech/gqasr/configs/02_full_12node.yaml}"
COMMAND="${1:?expected plan|run|status|verify|peek|assemble|publish|configs|cs-plan|cs-run|cs-verify|cs-assemble|gqasr-append|asr-start|llm-start}"
shift
JOBS="${JOBS:-8}"

base_index() {
  "$PY" - "$GQASR_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1]).resolve()
print((p.parent / yaml.safe_load(p.read_text())['index_path']).resolve())
PY
}

case "$COMMAND" in
  plan)
    exec "$PY" -m data_processing.yodas3 plan --run-root "$ROOT" \
      --sources "${SOURCES_CONFIG:-configs/data_processing/yodas3_asr_sources.yaml}" --jobs "$JOBS" "$@"
    ;;
  run)
    "$PY" -m data_processing.yodas3 check-run --run-root "$ROOT"
    : "${ASR_URLS:?Set ASR_URLS to the TWO ASR base URLs separated by a space}"
    read -r -a ENDPOINTS <<< "$ASR_URLS"
    [[ ${#ENDPOINTS[@]} == 2 && "${ENDPOINTS[0]}" != "${ENDPOINTS[1]}" ]] || {
      echo 'ASR_URLS must contain two distinct base URLs' >&2; exit 1;
    }
    [[ $# == 0 ]] || { echo 'Run policy is frozen; tune throughput with environment variables' >&2; exit 1; }
    # 88 processes total by default, with 16 in-flight ASR requests each.
    # Each corrector gets only its own node's requests; both ASRs share/fail over.
    export RUN_ID ROOT ASR_URLS
    export PROCS="${PROCS:-8}" FOREGROUND="${FOREGROUND:-0}"
    export LLM_URL="${LLM_URL:-http://localhost:8010}"
    exec bash scripts/corpus/run_asr_clean.sh run \
      --asr-model "${ASR_MODEL:-qasr}" --llm-model "${LLM_MODEL_NAME:-corrector}" \
      --asr-concurrency "${ASR_CONCURRENCY:-16}" --llm-concurrency "${LLM_CONCURRENCY:-16}" \
      --flush-every "${FLUSH_EVERY:-256}" --windows-in-flight "${WINDOWS_IN_FLIGHT:-2}" \
      --llm-context "${LLM_CONTEXT:-16384}" --thinking never \
      --dialect-guard quarantine --ar-diacritics critical --max-duration 35
    ;;
  status|verify|peek|assemble|stop)
    export RUN_ID ROOT JOBS
    exec bash scripts/corpus/run_asr_clean.sh "$COMMAND" "$@"
    ;;
  publish)
    exec "$PY" -m data_processing.yodas3 publish --run-root "$ROOT" \
      --destination "${DESTINATION:-$REPO/asr_cleaned_manifests/full_v1/ar/data}" \
      --base-index "${BASE_GQASR_INDEX:-$(base_index)}" --jobs "$JOBS" "$@"
    ;;
  configs)
    exec "$PY" -m data_processing.yodas3 configs --run-root "$ROOT" \
      --qasr-config "$QASR_CONFIG" --gqasr-config "$GQASR_CONFIG" --cs-root "$CS_ROOT" --repo "$REPO" "$@"
    ;;
  cs-plan)
    BASE_CS_INDEX="${BASE_CS_INDEX:-$("$PY" - "$QASR_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
print(Path(yaml.safe_load(Path(sys.argv[1]).read_text())['code_switch_index']).resolve())
PY
)}"
    exec "$PY" - "$CS_ROOT" "$BASE_CS_INDEX" "$ROOT/updates/additions.json" <<'PY'
import json, sys
from data_processing.code_switch.append import plan_append
plan_append(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))['train_manifest']['ar'])
PY
    ;;
  cs-run|cs-status|cs-verify|cs-assemble|cs-peek|cs-stop)
    STAGE="${COMMAND#cs-}"
    export ROOT="$CS_ROOT" RUN_ID="cs_$RUN_ID" JOBS
    exec bash scripts/corpus/run_code_switch.sh "$STAGE" "$@"
    ;;
  gqasr-append)
    # Run in the GQASR environment (or set GQASR_PYTHON to its interpreter).
    export PYTHONPATH="$REPO/gemma_speech/gqasr/src:$PYTHONPATH"
    ARGS=(--config "$ROOT/updates/gqasr.yaml" --source-index "${BASE_GQASR_INDEX:-$(base_index)}"
          --workers "$JOBS" --report "$ROOT/updates/gqasr_append_report.json")
    if [[ -n "${SCAN_TEMP_DIR:-}" ]]; then ARGS+=(--temp-dir "$SCAN_TEMP_DIR"); fi
    exec "${GQASR_PYTHON:-$PY}" -m gqasr append-manifests "${ARGS[@]}" "$@"
    ;;
  asr-start|asr-preflight|asr-print)
    export CONTAINER="${CONTAINER:-qasr-asr-yodas3}"
    STAGE="${COMMAND#asr-}"
    [[ "$STAGE" != start ]] || STAGE=serve
    exec bash scripts/serve_asr_vllm.sh "$STAGE" "$@"
    ;;
  llm-start|llm-print)
    exec bash scripts/corpus/serve_yodas3_corrector.sh "${COMMAND#llm-}" "$@"
    ;;
  *) echo "Unknown command: $COMMAND" >&2; exit 1 ;;
esac
