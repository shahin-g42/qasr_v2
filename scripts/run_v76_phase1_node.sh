#!/bin/bash
# =============================================================================
# QASR v7.6 — Phase 1 (projector warm-up) per-node manual launcher.
#
# For a pre-allocated 8-node x 8-GPU cluster where you start the job yourself
# on each node (no sbatch). Run this ON EVERY NODE, within the same ~2h
# rendezvous window, giving each node its rank:
#
#   Node 0 (master):  bash scripts/run_v76_phase1_node.sh 0 <master-hostname>
#   Node 1:           bash scripts/run_v76_phase1_node.sh 1 <master-hostname>
#   ...
#   Node 7:           bash scripts/run_v76_phase1_node.sh 7 <master-hostname>
#
# <master-hostname> is node 0's hostname (or IB IP), identical on all nodes.
# Extra args pass through to train.py, e.g. the smoke gate:
#
#   bash scripts/run_v76_phase1_node.sh 0 <master> --smoke-test   # on all 8
#
# Distribution stack: torchrun (launcher) -> HF Trainer on Accelerate ->
# DeepSpeed ZeRO-2 (configs/deepspeed_zero2.json, wired in the YAML).
# Global batch = 8 nodes x 8 GPUs x 10/device = 640; one full stratified
# epoch = 903,217 steps. The epoch-coverage guard verifies the live world
# size at startup: if any node fails to join, training REFUSES to start
# rather than silently covering a fraction of the data.
#
# Other phases: CONFIG=configs/v7.6/02_full_8node.yaml bash scripts/run_v76_phase1_node.sh <rank> <master>
# (Phase 4 / EAGLE uses train_eagle.py — see configs/v7.6/README.md.)
# =============================================================================

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 NODE_RANK MASTER_ADDR [extra train.py args, e.g. --smoke-test]" >&2
    exit 1
fi

NODE_RANK="$1"
MASTER_ADDR="$2"
shift 2

if ! [[ "${NODE_RANK}" =~ ^[0-7]$ ]]; then
    echo "NODE_RANK must be 0-7, got: ${NODE_RANK}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration (override via env)
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/v7.6/01_projector_8node.yaml}"
NNODES="${NNODES:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache}"

# ---------------------------------------------------------------------------
# Environment (identical to scripts/qasr_train_8node.slurm)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_CACHE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# Known-good H100 + InfiniBand tuning:
export NCCL_BUFFSIZE=8388608
export CUDA_DEVICE_MAX_CONNECTIONS=1
# JIT build caches MUST be node-local: the home dir is NFS, and 64 ranks
# racing a DeepSpeed/Triton build lock on NFS deadlocks the first step
# (the conversion log's Triton warning was about exactly this).
export TORCH_EXTENSIONS_DIR="/tmp/${USER}_torch_ext"
export TRITON_CACHE_DIR="/tmp/${USER}_triton"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
# do NOT set NCCL_ALGO=Tree — breaks ncclInt8 broadcast on NCCL 2.27.5.

eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

# The Phase A recipe is verified against the pinned 5.14.x wheel only.
python -c "import transformers; assert transformers.__version__.startswith('5.14'), \
    f'transformers {transformers.__version__} != pinned 5.14.x'"

echo "============================================================"
echo " QASR v7.6 Phase 1 — node ${NODE_RANK}/${NNODES}"
echo " Config:   ${CONFIG}"
echo " Master:   ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs:     8 on this node ($((NNODES * 8)) total)"
echo " Env:      ${CONDA_ENV}"
echo "============================================================"

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node=8 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_conf=timeout=7200 \
    train.py --config "${CONFIG}" "$@"
