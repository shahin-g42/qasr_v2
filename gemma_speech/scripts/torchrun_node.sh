#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: MASTER_ADDR=host NODE_RANK=0 bash torchrun_node.sh CONFIG [train options]" >&2
  exit 2
fi
: "${MASTER_ADDR:?Set the reachable hostname or IP of node 0}"
: "${NODE_RANK:?Set a unique rank from 0 to NNODES-1 on each node}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

config_path="$1"
shift
exec torchrun \
  --nnodes="${NNODES:-12}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="${MASTER_PORT:-29541}" \
  --max_restarts=0 \
  -m gemma_speech "${SPEECH_TRAIN_COMMAND:-train}" --config "$config_path" "$@"
