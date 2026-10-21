#!/usr/bin/env bash
# Run from the activated AUD environment. See audm/AUDPRO_CLUSTER.md.
set -euo pipefail
AUD_REPO_ROOT="${AUD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export AUD_REPO_ROOT
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
AUD_MODEL="${AUDPRO_MODEL:-$AUD_REPO_ROOT/models/AudPro-v1}"
AUD_RUN_ROOT="${AUDPRO_RUN_ROOT:-$AUD_REPO_ROOT/output/audpro-v1}"
AUD_PYTHON="${AUD_PYTHON:-python}"
AUD_FULL_CONFIG="${AUDPRO_FULL_CONFIG:-$AUD_REPO_ROOT/audm/configs/audpro_full_5node.yaml}"
export TRITON_CACHE_DIR="${AUDPRO_TRITON_CACHE_DIR:-/tmp/${USER:-audm}/audpro-triton-${HOSTNAME:-$(hostname)}}"
mkdir -p "$TRITON_CACHE_DIR"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$AUD_REPO_ROOT"

case "$ACTION" in
  export)
    : "${QASR_CHECKPOINT:?Set QASR_CHECKPOINT to the trained QASR checkpoint to retain}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    exec "$AUD_PYTHON" -m audm export --variant audpro \
      --qasr-checkpoint "$QASR_CHECKPOINT" --output "$AUD_MODEL" "$@"
    ;;
  verify)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    exec "$AUD_PYTHON" -m audm verify "$AUD_MODEL" --device "${AUD_VERIFY_DEVICE:-cuda:0}" "$@"
    ;;
  preflight)
    if [[ "${1:-}" == full ]]; then shift; fi
    exec "$AUD_PYTHON" -m audm preflight --config "$AUD_FULL_CONFIG" --model "$AUD_MODEL" \
      --output "$AUD_RUN_ROOT/full" "$@"
    ;;
  nccl|smoke|full)
    RANK="${1:-}"
    [[ "$RANK" =~ ^[0-4]$ ]] || { echo 'Supply node rank 0..4' >&2; exit 2; }
    shift
    : "${AUDPRO_MASTER_ADDR:?Set AUDPRO_MASTER_ADDR to the reachable AudPro rank-0 host}"
    export AUD_MASTER_ADDR="$AUDPRO_MASTER_ADDR"
    export AUD_NNODES=5 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK="$RANK"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export AUD_MASTER_PORT="${AUDPRO_MASTER_PORT:-29601}"
    # Always launch directly on this node, including inside a Slurm allocation.
    export AUD_USE_SLURM=0
    if [[ "$ACTION" == nccl ]]; then
      export AUD_COMMAND=distributed-check
      ARGS=(--expected-world-size 40)
    else
      export AUD_COMMAND=train
      ARGS=(--config "$AUD_FULL_CONFIG" --model "$AUD_MODEL" --output "$AUD_RUN_ROOT/full")
      if [[ "$ACTION" == smoke ]]; then ARGS+=(--smoke); fi
    fi
    if [[ "${AUD_DRY_RUN:-0}" == 1 ]]; then
      printf 'rank=%s nodes=5 GPUs/node=8 master=%s:%s command=%s\n' "$RANK" "$AUD_MASTER_ADDR" "$AUD_MASTER_PORT" "$AUD_COMMAND"
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
    echo "AudPro action=$ACTION rank=$RANK master=$AUD_MASTER_ADDR:$AUD_MASTER_PORT log=$LOG_FILE"
    exec bash "$AUD_REPO_ROOT/audm/scripts/train_distributed.sh" "${ARGS[@]}" "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
AudPro manual-node workflow (activate the AUD environment on every node):
  bash audm/scripts/audpro.sh export
  bash audm/scripts/audpro.sh verify
  bash audm/scripts/audpro.sh preflight full --scan
  bash audm/scripts/audpro.sh nccl NODE_RANK
  bash audm/scripts/audpro.sh smoke NODE_RANK
  bash audm/scripts/audpro.sh full NODE_RANK [--resume auto]

NODE_RANK is 0..4; launch the same action once on each of the 5 nodes.
Required: QASR_CHECKPOINT for export; AUDPRO_MASTER_ADDR for distributed actions.
Optional: AUDPRO_MODEL, AUDPRO_RUN_ROOT, AUDPRO_MASTER_PORT (default 29601),
          AUDPRO_FULL_CONFIG, AUDPRO_TRITON_CACHE_DIR, AUD_REPO_ROOT, AUD_PYTHON.
AudPro-specific paths and rendezvous settings do not inherit AudTurbo settings.
AUD_DRY_RUN=1 prints a distributed launch without allocating GPUs or writing logs.
HELP
    ;;
  *) echo "Unknown action: $ACTION; use --help" >&2; exit 2 ;;
esac
