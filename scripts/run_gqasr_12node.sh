#!/usr/bin/env bash
# Run once on each node: rank 0 on 051, unique ranks 1..11 elsewhere.
# Uses gemma_speech/.venv; does not activate the original QASR conda environment.
set -euo pipefail
usage() {
  echo "Usage: $0 NODE_RANK [MASTER_ADDR] [--stage bridge|full] [--preflight] [--resume CHECKPOINT]" >&2
}
[[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]] || { usage; exit 2; }
export NODE_RANK=$((10#$1))
shift
export MASTER_ADDR="${MASTER_ADDR:-inception-H100-hpc-051}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MASTER_ADDR="$1"
  shift
fi
STAGE=bridge
PREFLIGHT=0
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      STAGE="$2"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    --resume)
      [[ $# -ge 2 && "$2" != --* ]] || { usage; exit 2; }
      TRAIN_ARGS+=(--resume "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "$STAGE" == bridge || "$STAGE" == full ]] || { echo 'Stage must be bridge or full' >&2; exit 2; }
export NNODES="${NNODES:-12}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
[[ "$NNODES" == 12 && "$NPROC_PER_NODE" == 8 ]] || {
  echo 'This launcher requires 12 nodes x 8 GPUs; use torchrun_node.sh with a matching custom config for other sizes.' >&2
  exit 2
}
(( NODE_RANK < 12 )) || { echo 'NODE_RANK must be 0 through 11' >&2; exit 2; }
[[ -n "$MASTER_ADDR" ]] || { echo 'MASTER_ADDR must not be empty' >&2; exit 2; }
if [[ "$STAGE" == bridge ]]; then
  DEFAULT_PORT=29611
  CONFIG_NAME=01_bridge_12node.yaml
else
  DEFAULT_PORT=29612
  CONFIG_NAME=02_full_12node.yaml
fi
export MASTER_PORT="${MASTER_PORT:-$DEFAULT_PORT}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo 'MASTER_PORT must be an integer' >&2; exit 2; }
(( 10#$MASTER_PORT >= 1 && 10#$MASTER_PORT <= 65535 )) || { echo 'Invalid MASTER_PORT' >&2; exit 2; }
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
export GQASR_CONFIG="${GQASR_CONFIG:-$PROJECT_DIR/gemma_speech/gqasr/configs/$CONFIG_NAME}"
[[ -f "$GQASR_CONFIG" ]] || { echo "Missing config: $GQASR_CONFIG" >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Respect scheduler-assigned visibility; GPU_IDS provides an explicit override.
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}}"
echo "GQASR $STAGE: rank=$NODE_RANK/12 master=$MASTER_ADDR:$MASTER_PORT config=$GQASR_CONFIG"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  if (( PREFLIGHT )); then
    printf 'gqasr preflight --config %q\n' "$GQASR_CONFIG"
  else
    bash "$PROJECT_DIR/gemma_speech/gqasr/scripts/torchrun_node.sh" "$GQASR_CONFIG" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
  fi
  exit 0
fi
if (( PREFLIGHT )); then
  exec bash "$PROJECT_DIR/gemma_speech/cluster/gqasr_stage.sh" preflight
fi
# Check the unified environment and eight BF16-capable GPUs before rendezvous.
bash "$PROJECT_DIR/gemma_speech/cluster/gqasr_stage.sh" doctor
exec bash "$PROJECT_DIR/gemma_speech/cluster/gqasr_stage.sh" "$STAGE" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
