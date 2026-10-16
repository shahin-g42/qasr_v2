#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
STAGE="${1:?Use download, run or train}"
shift
MODEL_DIR="${QWEN_BASELINE_MODEL:-$ROOT/models/qwen3-asr-1.7b-hf}"
RUN_DIR="${GQASR_RUN_DIR:-$ROOT/gqasr/outputs/full_b40_frozen}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
case "$STAGE" in
  download)
    HF_HUB_OFFLINE=0 python - "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = "Qwen/Qwen3-ASR-1.7B-hf"
revision = HfApi().model_info(repo).sha
path = snapshot_download(repo, revision=revision, local_dir=sys.argv[1])
Path(path, "baseline_source.json").write_text(json.dumps({"repository": repo, "revision": revision}, indent=2))
print(path)
PY
    ;;
  run)
    export HF_HUB_OFFLINE=1
    export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
    python -m gqasr.qwen_baseline \
      --run-config "$RUN_DIR/config.json" \
      --predictions "$RUN_DIR/predictions-${EVAL_STEP:-7294}.jsonl" \
      --model "$MODEL_DIR" \
      --output "${BASELINE_OUTPUT:-$ROOT/gqasr/verification/qwen17b-zh-${EVAL_STEP:-7294}-$(date -u +%Y%m%dT%H%M%SZ)}" \
      "$@"
    ;;
  train)
    export HF_HUB_OFFLINE=1
    export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
    python -m gqasr.qwen_baseline \
      --run-config "$RUN_DIR/config.json" --train \
      --model "$MODEL_DIR" \
      --output "${BASELINE_OUTPUT:-$ROOT/gqasr/verification/qwen17b-zh-train-$(date -u +%Y%m%dT%H%M%SZ)}" \
      "$@"
    ;;
  *) echo "Unknown stage: $STAGE (use download, run or train)" >&2; exit 2 ;;
esac
