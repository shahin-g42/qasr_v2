#!/bin/bash
#SBATCH --job-name=qasr-train-8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=144:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

# =============================================================================
# QASR Multi-Node Training via SLURM (8 nodes x 8 GPUs = 64 GPUs)
#
# The body is node-count agnostic: it reads SLURM_NNODES/SLURM_NODEID and
# picks the master from SLURM_JOB_NODELIST, so `sbatch --nodes=1 ...` works
# for the 1-node smoke gate as well.
#
# Usage:
#   # Phase 1 — projector warm-up (deep MLP):
#   sbatch --export=ALL,CONFIG=configs/train_projector_8node.yaml scripts/qasr_train_8node.slurm
#
#   # Phase 2 — smoke gate first (1 node), then full 8-node launch:
#   sbatch --nodes=1 --export=ALL,CONFIG=configs/train_full_8node_filtered.yaml \
#     scripts/qasr_train_8node.slurm --smoke-test
#   sbatch --export=ALL,CONFIG=configs/train_full_8node_filtered.yaml scripts/qasr_train_8node.slurm
#
#   # Phase 4 (optional) — HQ fine-tune:
#   sbatch --export=ALL,CONFIG=configs/train_hq_8node.yaml scripts/qasr_train_8node.slurm
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via --export or sbatch --wrap)
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/train_projector_8node.yaml}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/.cache}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Environment
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
# do NOT set NCCL_ALGO=Tree — breaks ncclInt8 broadcast on NCCL 2.27.5.

# ---------------------------------------------------------------------------
# Activate conda and enter project
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

# Cluster transformers drift guard: the whole Phase A recipe is verified
# against the pinned 5.14.x wheel only.
python -c "import transformers; assert transformers.__version__.startswith('5.14'), \
    f'transformers {transformers.__version__} != pinned 5.14.x'"

# ---------------------------------------------------------------------------
# SLURM multi-node discovery
# ---------------------------------------------------------------------------
NNODES="${SLURM_NNODES:-8}"
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

echo "============================================================"
echo " QASR Training (8-node topology)"
echo "============================================================"
echo " Job ID:        ${SLURM_JOB_ID:-local}"
echo " Config:        ${CONFIG}"
echo " Nodes:         ${NNODES}"
echo " Node rank:     ${NODE_RANK}"
echo " Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs/node:     8"
echo " Total GPUs:    $((NNODES * 8))"
echo " Conda env:     ${CONDA_ENV}"
echo " Project dir:   ${PROJECT_DIR}"
echo " HF cache:      ${HF_CACHE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch training with torchrun (elastic rendezvous via static backend)
# ---------------------------------------------------------------------------
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --rdzv_backend=static \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_conf=timeout=7200 \
  train.py \
    --config "${CONFIG}" \
    "$@"

echo "============================================================"
echo " Training finished on node ${NODE_RANK} at $(date)"
echo "============================================================"
