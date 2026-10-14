#!/usr/bin/env bash
set -euo pipefail
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0 address reachable by all nodes}"
: "${NODE_RANK:?Set NODE_RANK from 0 through 11}"
CONFIG="${1:?Usage: torchrun_node.sh CONFIG [--resume CHECKPOINT]}"
shift
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec torchrun --nnodes="${NNODES:-12}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
  -m gqasr train --config "$CONFIG" "$@"
