#!/bin/bash
#SBATCH --job-name=qasr-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

# =============================================================================
# QASR Multi-Node Training via SLURM (4 nodes x 8 GPUs = 32 GPUs)
#
# Usage:
#   sbatch scripts/qasr_train_4node.slurm                    # projector stage
#   sbatch --export=ALL,CONFIG=configs/train_full_4node.yaml scripts/qasr_train_4node.slurm
#   sbatch scripts/qasr_train_4node.slurm --smoke-test       # quick validation
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via --export or sbatch --wrap)
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/train_projector_4node.yaml}"
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

# ---------------------------------------------------------------------------
# Activate conda and enter project
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# SLURM multi-node discovery
# ---------------------------------------------------------------------------
NNODES="${SLURM_NNODES:-4}"
NODE_RANK="${SLURM_NODEID:-0}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

echo "============================================================"
echo " QASR Training"
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
