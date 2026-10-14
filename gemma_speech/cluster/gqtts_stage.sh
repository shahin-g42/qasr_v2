#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE="${1:?Stages: download cache speaker-train train synthesize}"
shift
source "${GEMMA_SPEECH_ENV:-$ROOT/.venv}/bin/activate"
mkdir -p "$ROOT/logs"
exec > >(tee "$ROOT/logs/gqtts-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-${NODE_RANK:-local}.log") 2>&1
export TOKENIZERS_PARALLELISM=false
if [[ "$STAGE" == download ]]; then
  export HF_HUB_OFFLINE=0
  gemma-speech prefetch --output "$ROOT/models/embeddinggemma2"
  gemma-speech prefetch --qwen-tts --output "$ROOT/models/qwen-tts"
else
  export HF_HUB_OFFLINE=1
  case "$STAGE" in
    cache) gqtts cache --gemma "$ROOT/models/embeddinggemma2" --qwen "$ROOT/models/qwen-tts" "$@" ;;
    speaker-train|train)
      CONFIG="$ROOT/gqtts/configs/speaker.json"
      if [[ "$STAGE" == train ]]; then CONFIG="$ROOT/gqtts/configs/arabic.json"; fi
      CONFIG="${GQTTS_CONFIG:-$CONFIG}"
      : "${MASTER_ADDR:?Set node 0 address}"
      : "${NODE_RANK:?Set node rank}"
      export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
      torchrun --nnodes="${NNODES:-12}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
        --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
        -m gqtts "$STAGE" --config "$CONFIG" "$@" ;;
    synthesize) gqtts synthesize --gemma "$ROOT/models/embeddinggemma2" --qwen "$ROOT/models/qwen-tts" "$@" ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
  esac
fi
