#!/usr/bin/env bash
# ASR-assisted cleaning of every training/eval manifest (data_processing.asr_clean).
#
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh plan      # once, any node
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh run       # on EACH of the 8 LLM nodes
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh status    # anywhere, any time
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh peek -n 5 [--rejects] [--only-changed]
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh assemble  # when sources finish
#   RUN_ID=asrc1 bash scripts/corpus/run_asr_clean.sh stop      # on a node: stop its workers
#
# All 8 LLM nodes at once from the allocation (one task per node, foreground):
#   srun --nodelist=<the 8 corrector nodes> --ntasks-per-node=1 \
#        env RUN_ID=asrc1 FOREGROUND=1 bash scripts/corpus/run_asr_clean.sh run
#
# Each node runs PROCS worker processes (stable ids host:0..PROCS-1, so a
# restarted node resumes exactly its own chunks). Workers claim chunks
# dynamically from the shared plan, call the ASR on hpc-029 and the corrector
# on localhost:8010, and commit FLUSH_EVERY records at a time.
#
# Output: $ROOT/<lang>/<stem>/part-NNNNN.jsonl with exactly
#   audio_filepath, duration, text, org_text, asr_text
# and after `assemble`: $ROOT/<lang>/<stem>.jsonl. Rejects (with the reason)
# under $ROOT/_rejects/. Re-running `run` resumes; it never redoes or loses work.

set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${RUN_ID:?set RUN_ID (one id per cleaning run, the same on every node)}"
ROOT="${ROOT:-$REPO/asr_cleaned_manifests/$RUN_ID}"
# Source manifests: the flat data/ folder (<split>_<lang>_<name>.json) by
# default. The v7.6 training_manifests tree is NOT used (corrupted, per user
# 2026-10-09); set CONFIG=<training yaml> to take a config's manifests instead.
DATA_DIR="${DATA_DIR:-$REPO/data}"
CONFIG="${CONFIG:-}"
# Space-separated; requests round-robin with failover across all of them.
ASR_URLS="${ASR_URLS:-http://inception-H100-hpc-029.inception.ai:8020}"
LLM_URL="${LLM_URL:-http://localhost:8010}"
PROCS="${PROCS:-16}"
PY="${PYTHON:-python3}"
EXTRA=("${@:2}")

log() { printf '[%s %s] %s\n' "$(date +%H:%M:%S)" "$(hostname -s)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info >= (3, 10)' || die "$PY is older than 3.10 (set PYTHON=)"

case "${1:-}" in
  plan)
    "$PY" -c 'import numpy' 2>/dev/null || die "plan needs numpy for the dedup ($PY)"
    if [ -n "$CONFIG" ]; then SRC=(--config "$CONFIG"); else SRC=(--data-dir "$DATA_DIR"); fi
    log "planning ${SRC[*]} -> $ROOT"
    exec "$PY" -m data_processing.asr_clean plan --run-root "$ROOT" "${SRC[@]}" \
      --jobs "${JOBS:-64}" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  run)
    [ -f "$ROOT/_state/plan.json" ] || die "no plan at $ROOT -- run 'plan' first (same RUN_ID)"
    curl -sf --max-time 10 "$LLM_URL/health" >/dev/null || die "corrector not healthy at $LLM_URL"
    # Several ASR servers fail over between each other: start if ANY is healthy.
    ASR_ARGS=()
    up=0
    for u in $ASR_URLS; do
      if curl -sf --max-time 10 "$u/health" >/dev/null; then up=$((up + 1)); else log "WARNING: ASR not healthy at $u (will fail over)"; fi
      ASR_ARGS+=(--asr-url "$u")
    done
    [ "$up" -gt 0 ] || die "no ASR server is healthy: $ASR_URLS"
    ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    mkdir -p "$ROOT/_logs"
    CMD=("$PY" -m data_processing.asr_clean run --run-root "$ROOT" "${ASR_ARGS[@]}"
         --llm-url "$LLM_URL" --procs "$PROCS" ${EXTRA[@]+"${EXTRA[@]}"})
    if [ "${FOREGROUND:-0}" = 1 ]; then
      log "${CMD[*]}"
      exec "${CMD[@]}"
    fi
    nohup "${CMD[@]}" > "$ROOT/_logs/$(hostname -s).out" 2>&1 &
    echo $! > "$ROOT/_logs/$(hostname -s).pid"
    log "started $PROCS workers (pid $!): tail -f $ROOT/_logs/$(hostname -s)-*.log"
    ;;
  status)
    exec "$PY" -m data_processing.asr_clean status --run-root "$ROOT" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  peek)
    exec "$PY" -m data_processing.asr_clean peek --run-root "$ROOT" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  assemble)
    exec "$PY" -m data_processing.asr_clean assemble --run-root "$ROOT" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  stop)
    pidf="$ROOT/_logs/$(hostname -s).pid"
    [ -f "$pidf" ] || die "no pid file $pidf on this node"
    # The parent and its workers; committed work is safe, in-flight windows are redone.
    pkill -TERM -P "$(cat "$pidf")" 2>/dev/null || true
    kill -TERM "$(cat "$pidf")" 2>/dev/null || true
    rm -f "$pidf"
    log "stopped"
    ;;
  *)
    die "usage: RUN_ID=... $0 {plan|run|status|peek|assemble|stop} [extra args]"
    ;;
esac
