#!/usr/bin/env bash
# Evaluate the completed QASR Stage 2 artifact using the shared Arabic benchmark.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_DIR="${GEMMA_SPEECH_ENV:-$ROOT/gemma_speech/.venv}"
STAGE="${1:?Stages: test doctor preflight smoke tune run}"
shift
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export QASR_BENCH_TRAIN_CONFIG="${QASR_BENCH_TRAIN_CONFIG:-$ROOT/configs/v8.1/02_full_12node.yaml}"
if [[ -z "${QASR_BENCH_CHECKPOINT:-}" ]]; then
  QASR_BENCH_CHECKPOINT=$("$ENV_DIR/bin/python" - "$QASR_BENCH_TRAIN_CONFIG" <<'PY'
import sys
import yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text())
# Use the completed training output, never the model_name_or_path initialization.
print(config["output_dir"])
PY
)
fi
export GQASR_BENCH_CHECKPOINT="$QASR_BENCH_CHECKPOINT"
export GQASR_BENCH_WORK="${QASR_BENCH_WORK:-${GQASR_BENCH_WORK:-$ROOT/gemma_speech/gqasr/verification/arabic-stream-cache}}"
export GQASR_BENCH_OUTPUT="${QASR_BENCH_OUTPUT:-$ROOT/gemma_speech/gqasr/verification/qasr-stage02-arabic}"
export GQASR_BENCH_SMOKE_OUTPUT="${QASR_BENCH_SMOKE_OUTPUT:-${GQASR_BENCH_OUTPUT}-smoke}"
export GQASR_BENCH_TUNING_OUTPUT="${QASR_BENCH_TUNING_OUTPUT:-${GQASR_BENCH_OUTPUT}-tuning}"
export GQASR_BENCH_CONFIG="$ROOT/gemma_speech/gqasr/configs/arabic_leaderboard_qasr.yaml"
export GQASR_BENCH_MODEL_NAME=QASR
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
# Six sets x 256 clips / batch 16 = 96 tasks: enough to start all 64
# replicas for the requested 8 workers/GPU, with the production memory share.
export SMOKE_PER_DATASET="${SMOKE_PER_DATASET:-256}"
echo "QASR checkpoint: $GQASR_BENCH_CHECKPOINT"
echo "Shared prepared audio: $GQASR_BENCH_WORK"
echo "QASR output: $GQASR_BENCH_OUTPUT"
if [[ "$STAGE" == test ]]; then
  bash "$ROOT/gemma_speech/cluster/gqasr_arabic_leaderboard.sh" qasr-test "$@"
  exec bash "$ROOT/gemma_speech/cluster/gqasr_arabic_leaderboard.sh" test "$@"
fi
exec bash "$ROOT/gemma_speech/cluster/gqasr_arabic_leaderboard.sh" "$STAGE" "$@"
