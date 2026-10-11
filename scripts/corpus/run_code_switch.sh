#!/usr/bin/env bash
# Prepare an immutable short-clip index; audio is composed online in training.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
RUN_ID="${RUN_ID:?set RUN_ID, identical on all machines}"
ROOT="${ROOT:-$REPO/code_switch_indexes/$RUN_ID}"
PY="${PYTHON:-python3}"
COMMAND="${1:?expected plan|run|stop|status|verify|assemble|peek|lock-test}"
shift
ARGS=("$COMMAND" --run-root "$ROOT")
case "$COMMAND" in
  plan) ARGS+=(--config "${CONFIG:-configs/code_switch/full_v1.yaml}" --jobs "${JOBS:-8}") ;;
  run) ARGS+=(--procs "${PROCS:-8}") ;;
  verify|assemble) ARGS+=(--jobs "${JOBS:-16}") ;;
  stop|status|peek|lock-test) ;;
  *) echo "Unknown command: $COMMAND" >&2; exit 1 ;;
esac
if [[ "$COMMAND" == run && "${FOREGROUND:-0}" != 1 ]]; then
  mkdir -p "$ROOT/_logs"
  LOG="$ROOT/_logs/$(hostname -s)-$(date +%Y%m%d-%H%M%S)-$$.log"
  nohup "$PY" -m data_processing.code_switch "${ARGS[@]}" "$@" > "$LOG" 2>&1 &
  echo "Supervisor started (pid $!); check $LOG for startup/lock errors."
else
  exec "$PY" -m data_processing.code_switch "${ARGS[@]}" "$@"
fi
