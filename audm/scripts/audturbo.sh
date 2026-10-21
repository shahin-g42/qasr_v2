#!/usr/bin/env bash
# Run from the activated AUD environment. See audm/AUDTURBO_CLUSTER.md.
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export AUD_REPO_ROOT
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
AUD_MODEL="${AUD_MODEL:-$AUD_REPO_ROOT/models/AudTurbo-v1}"
AUD_RUN_ROOT="${AUD_RUN_ROOT:-$AUD_REPO_ROOT/output/audturbo-v1}"
AUD_PYTHON="${AUD_PYTHON:-python}"
AUD_ADAPTER_CONFIG="${AUD_ADAPTER_CONFIG:-$AUD_REPO_ROOT/audm/configs/audturbo_adapter_8node.yaml}"
AUD_FULL_CONFIG="${AUD_FULL_CONFIG:-$AUD_REPO_ROOT/audm/configs/audturbo_full_8node.yaml}"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$AUD_REPO_ROOT"

case "$ACTION" in
  export)
    : "${QASR_CHECKPOINT:?Set QASR_CHECKPOINT to the trained QASR checkpoint to retain}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    exec "$AUD_PYTHON" -m audm export --variant audturbo \
      --qasr-checkpoint "$QASR_CHECKPOINT" --output "$AUD_MODEL" \
      --encoder-source "${AUD_CAPTIONER_SOURCE:-Qwen/Qwen3-Omni-30B-A3B-Captioner}" \
      --encoder-revision "${AUD_CAPTIONER_REVISION:-main}" "$@"
    ;;
  verify)
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
    exec "$AUD_PYTHON" -m audm preflight --config "$CONFIG" --model "$MODEL" \
      --output "$AUD_RUN_ROOT/$STAGE" "$@"
    ;;
  nccl|smoke|adapter|full)
    RANK="${1:-}"
    [[ "$RANK" =~ ^[0-7]$ ]] || { echo 'Supply node rank 0..7' >&2; exit 2; }
    shift
    : "${AUD_MASTER_ADDR:?Set AUD_MASTER_ADDR to the reachable rank-0 training host}"
    export AUD_NNODES=8 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK="$RANK"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export AUD_MASTER_PORT="${AUD_MASTER_PORT:-29591}"
    # Always launch directly on this node, including inside a Slurm allocation.
    export AUD_USE_SLURM=0
    if [[ "$ACTION" == nccl ]]; then
      export AUD_COMMAND=distributed-check
      ARGS=(--expected-world-size 64)
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
      printf 'rank=%s nodes=8 GPUs/node=8 master=%s:%s command=%s\n' "$RANK" "$AUD_MASTER_ADDR" "$AUD_MASTER_PORT" "$AUD_COMMAND"
      printf '%q ' bash "$AUD_REPO_ROOT/audm/scripts/train_distributed.sh" "${ARGS[@]}" "$@"
      printf '\n'
      exit 0
    fi
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
    LOG_DIR="$AUD_RUN_ROOT/launcher-logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/${ACTION}-$(date -u +%Y%m%dT%H%M%SZ)-rank${RANK}-$$.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "AudTurbo action=$ACTION rank=$RANK master=$AUD_MASTER_ADDR:$AUD_MASTER_PORT log=$LOG_FILE"
    exec bash "$AUD_REPO_ROOT/audm/scripts/train_distributed.sh" "${ARGS[@]}" "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
AudTurbo manual-node workflow (activate the same environment on every node):
  bash audm/scripts/audturbo.sh export
  bash audm/scripts/audturbo.sh verify
  bash audm/scripts/audturbo.sh preflight adapter --scan
  bash audm/scripts/audturbo.sh nccl NODE_RANK
  bash audm/scripts/audturbo.sh smoke NODE_RANK
  bash audm/scripts/audturbo.sh adapter NODE_RANK [--resume auto]
  bash audm/scripts/audturbo.sh preflight full --scan
  bash audm/scripts/audturbo.sh full NODE_RANK [--resume auto]

NODE_RANK is 0..7; launch the same action once on each of the 8 nodes.
Required: QASR_CHECKPOINT for export; AUD_MASTER_ADDR for distributed actions.
Optional: AUD_REPO_ROOT, AUD_MODEL, AUD_RUN_ROOT, AUD_MASTER_PORT,
          AUD_ADAPTER_CONFIG, AUD_FULL_CONFIG, AUD_CAPTIONER_REVISION.
AUD_DRY_RUN=1 prints a distributed launch without allocating GPUs or writing logs.
HELP
    ;;
  *) echo "Unknown action: $ACTION; use --help" >&2; exit 2 ;;
esac
