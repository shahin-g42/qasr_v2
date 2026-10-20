#!/usr/bin/env bash
set -euo pipefail
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0 address reachable by all nodes}"
: "${NODE_RANK:?Set NODE_RANK from 0 through NNODES minus 1}"
CONFIG="${1:?Usage: torchrun_node.sh CONFIG [--resume CHECKPOINT]}"
shift
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CMD=(torchrun --nnodes="${NNODES:-12}" --nproc_per_node="${NPROC_PER_NODE:-8}"
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}"
  --rdzv_backend=static --rdzv_endpoint="$MASTER_ADDR:${MASTER_PORT:-29500}"
  --rdzv_conf="timeout=${GQASR_RDZV_TIMEOUT:-7200}"
  -m gqasr train --config "$CONFIG" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
# RankStatus runs inside GPU workers, which do not exist until this rendezvous
# finishes. Use PyTorch's own agent logs to expose the earlier waiting stage.
export LOGLEVEL="${LOGLEVEL:-INFO}"
echo "[GQASR startup] node_rank=$NODE_RANK phase=torchrun_rendezvous nodes=${NNODES:-12} gpus_per_node=${NPROC_PER_NODE:-8} master=$MASTER_ADDR:${MASTER_PORT:-29500} timeout_seconds=${GQASR_RDZV_TIMEOUT:-7200}"
echo "[GQASR startup] GPU rank logs begin after all ${NNODES:-12} node launchers join rendezvous. LOGLEVEL=$LOGLEVEL rank_status=${GQASR_RANK_STATUS:-1}"
exec "${CMD[@]}"
