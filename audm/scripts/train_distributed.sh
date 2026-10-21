#!/usr/bin/env bash
set -euo pipefail
AUD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$AUD_ROOT/audm/src:$AUD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export AUD_GPUS_PER_NODE="${AUD_GPUS_PER_NODE:-8}"
export AUD_MASTER_PORT="${AUD_MASTER_PORT:-29571}"
AUD_COMMAND="${AUD_COMMAND:-train}"

if [[ "${AUD_USE_SLURM:-0}" == 1 && -z "${AUD_INSIDE_SRUN:-}" ]]; then
  export AUD_NNODES="${SLURM_JOB_NUM_NODES}"
  AUD_HOSTS="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
  export AUD_MASTER_ADDR="${AUD_HOSTS%%$'\n'*}"
  export AUD_INSIDE_SRUN=1
  exec srun --nodes="$AUD_NNODES" --ntasks="$AUD_NNODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 bash "$AUD_ROOT/audm/scripts/train_distributed.sh" "$@"
fi

# By default, run directly on every node with AUD_NODE_RANK=0..N-1.
export AUD_NNODES="${AUD_NNODES:-8}"
: "${AUD_MASTER_ADDR:?Set AUD_MASTER_ADDR to the rank-0 host reachable from all nodes}"
if [[ "${AUD_USE_SLURM:-0}" == 1 ]]; then
  AUD_NODE_RANK="${SLURM_PROCID:?Slurm launch requires SLURM_PROCID}"
fi
: "${AUD_NODE_RANK:?Set AUD_NODE_RANK on each node}"
exec "${AUD_PYTHON:-python}" -m torch.distributed.run --nnodes="$AUD_NNODES" --nproc-per-node="$AUD_GPUS_PER_NODE" \
  --node-rank="$AUD_NODE_RANK" --master-addr="$AUD_MASTER_ADDR" --master-port="$AUD_MASTER_PORT" \
  --max-restarts=0 -m audm "$AUD_COMMAND" "$@"
