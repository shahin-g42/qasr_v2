#!/usr/bin/env bash
# Run from the activated AUD environment. See audm/AUDFLASH_CLUSTER.md.
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export AUD_REPO_ROOT
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
AUD_MODEL="${AUDFLASH_MODEL:-$AUD_REPO_ROOT/models/AudFlash-Gemma4-v1}"
AUD_RUN_ROOT="${AUDFLASH_RUN_ROOT:-$AUD_REPO_ROOT/output/audflash-gemma4-v1}"
AUD_PYTHON="${AUD_PYTHON:-python}"
AUD_ADAPTER_CONFIG="${AUDFLASH_ADAPTER_CONFIG:-$AUD_REPO_ROOT/audm/configs/audflash_adapter_1node.yaml}"
AUD_FULL_CONFIG="${AUDFLASH_FULL_CONFIG:-$AUD_REPO_ROOT/audm/configs/audflash_full_1node.yaml}"
export TRITON_CACHE_DIR="${AUDFLASH_TRITON_CACHE_DIR:-/tmp/${USER:-audm}/audflash-triton-${HOSTNAME:-$(hostname)}}"
mkdir -p "$TRITON_CACHE_DIR"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$AUD_REPO_ROOT"

check_model() {
  "$AUD_PYTHON" - "$1/config.json" <<'PYMODEL'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
if (config.get("variant"), config.get("flash_encoder"), config.get("flash_audio_stride")) != ("audflash", "gemma4", 1):
    raise RuntimeError("This workflow requires a fresh Gemma4 AudFlash stride-1 export; do not resume a Moonshine or stride-2 run.")
PYMODEL
}

case "$ACTION" in
  export)
    : "${GQASR_CHECKPOINT:?Set GQASR_CHECKPOINT to the trained GQASR checkpoint to retain}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    exec "$AUD_PYTHON" -m audm export --variant audflash \
      --gqasr-checkpoint "$GQASR_CHECKPOINT" --output "$AUD_MODEL" \
      --encoder-source "${AUDFLASH_ENCODER_SOURCE:-google/gemma-4-E2B-it}" \
      --encoder-revision "${AUDFLASH_ENCODER_REVISION:-main}" \
      --bridge-init "${AUDFLASH_BRIDGE_INIT:-gqasr}" --flash-audio-stride 1 "$@"
    ;;
  verify)
    check_model "$AUD_MODEL"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    exec "$AUD_PYTHON" -m audm verify "$AUD_MODEL" --device "${AUD_VERIFY_DEVICE:-cuda:0}" "$@"
    ;;
  preflight)
    STAGE="${1:-adapter}"
    if [[ $# -gt 0 ]]; then shift; fi
    case "$STAGE" in
      adapter) CONFIG="$AUD_ADAPTER_CONFIG"; MODEL="$AUD_MODEL" ;;
      full) CONFIG="$AUD_FULL_CONFIG"; MODEL="$AUD_RUN_ROOT/adapter" ;;
      *) echo 'preflight stage must be adapter or full' >&2; exit 2 ;;
    esac
    check_model "$MODEL"
    exec "$AUD_PYTHON" -m audm preflight --config "$CONFIG" --model "$MODEL" \
      --output "$AUD_RUN_ROOT/$STAGE" "$@"
    ;;
  nccl|smoke|adapter|full)
    RANK="${1:-0}"
    [[ "$RANK" == 0 ]] || { echo 'Supply node rank 0' >&2; exit 2; }
    if [[ $# -gt 0 ]]; then shift; fi
    export AUD_MASTER_ADDR="${AUDFLASH_MASTER_ADDR:-127.0.0.1}"
    export AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK="$RANK"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export AUD_MASTER_PORT="${AUDFLASH_MASTER_PORT:-29611}"
    # Always launch directly on this node, including inside a Slurm allocation.
    export AUD_USE_SLURM=0
    if [[ "$ACTION" == nccl ]]; then
      export AUD_COMMAND=distributed-check
      ARGS=(--expected-world-size 8)
    else
      export AUD_COMMAND=train
      if [[ "$ACTION" == full ]]; then
        ARGS=(--config "$AUD_FULL_CONFIG" --model "$AUD_RUN_ROOT/adapter" --output "$AUD_RUN_ROOT/full")
      else
        ARGS=(--config "$AUD_ADAPTER_CONFIG" --model "$AUD_MODEL" --output "$AUD_RUN_ROOT/adapter")
        if [[ "$ACTION" == smoke ]]; then ARGS+=(--smoke); fi
      fi
    fi
    if [[ "${AUD_DRY_RUN:-0}" == 1 ]]; then
      printf 'rank=%s nodes=1 GPUs/node=8 master=%s:%s command=%s\n' "$RANK" "$AUD_MASTER_ADDR" "$AUD_MASTER_PORT" "$AUD_COMMAND"
      printf '%q ' bash "$AUD_REPO_ROOT/audm/scripts/train_distributed.sh" "${ARGS[@]}" "$@"
      printf '\n'
      exit 0
    fi
    if [[ "$ACTION" == full ]]; then check_model "$AUD_RUN_ROOT/adapter"
    elif [[ "$ACTION" != nccl ]]; then check_model "$AUD_MODEL"; fi
    "$AUD_PYTHON" - <<'PY'
import torch
import transformers
if not transformers.__version__.startswith('5.14.'):
    raise RuntimeError(f'Expected Transformers 5.14.x, got {transformers.__version__}')
if torch.cuda.device_count() != 8:
    raise RuntimeError(f'Expected 8 allocated visible GPUs, got {torch.cuda.device_count()}')
if not torch.cuda.is_bf16_supported():
    raise RuntimeError('BF16 support is required')
print(f'Torch={torch.__version__}; CUDA={torch.version.cuda}; GPUs={torch.cuda.device_count()}', flush=True)
PY
    export AUD_LOG_DIR="${AUD_LOG_DIR:-$AUD_RUN_ROOT/logs}"
    exec bash "$AUD_REPO_ROOT/audm/scripts/train_distributed.sh" "${ARGS[@]}" "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
AudFlash manual-node workflow (activate the same environment on every node):
  bash audm/scripts/audflash.sh export
  bash audm/scripts/audflash.sh verify
  bash audm/scripts/audflash.sh preflight adapter --scan
  bash audm/scripts/audflash.sh nccl NODE_RANK
  bash audm/scripts/audflash.sh smoke NODE_RANK
  bash audm/scripts/audflash.sh adapter NODE_RANK [--resume auto]
  bash audm/scripts/audflash.sh preflight full --scan
  bash audm/scripts/audflash.sh full NODE_RANK [--resume auto]

NODE_RANK is 0; run on the single AudFlash node. Master defaults to 127.0.0.1:29611.
Required: GQASR_CHECKPOINT for export.
Optional: AUDFLASH_MODEL, AUDFLASH_RUN_ROOT, AUDFLASH_MASTER_PORT,
          AUDFLASH_ADAPTER_CONFIG, AUDFLASH_FULL_CONFIG, AUDFLASH_ENCODER_SOURCE,
          AUDFLASH_ENCODER_REVISION, AUDFLASH_BRIDGE_INIT, AUDFLASH_TRITON_CACHE_DIR, AUD_REPO_ROOT, AUD_PYTHON.
AudFlash-specific defaults do not inherit AudTurbo/AudPro model or run paths.
AUD_DRY_RUN=1 prints a distributed launch without allocating GPUs or writing logs.
HELP
    ;;
  *) echo "Unknown action: $ACTION; use --help" >&2; exit 2 ;;
esac
