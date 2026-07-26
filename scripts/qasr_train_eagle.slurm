#!/bin/bash
#SBATCH --job-name=qasr-eagle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

# ============================================================================
# EAGLE-2 Draft Head Training for QASR
# Trains only the ~8M parameter EAGLE head on a frozen QASR checkpoint.
# Single node, 8 GPUs, ~2000 steps (~30-60 minutes).
# ============================================================================

set -euo pipefail

# --- Configuration ---
CONFIG="${CONFIG:-configs/train_eagle_8node_filtered.yaml}"
MASTER_PORT="${MASTER_PORT:-29501}"

# --- Environment ---
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
export HF_HOME="${HF_HOME:-/lustrefs/shared/shahin.konadath/.cache/huggingface}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

echo "============================================"
echo " EAGLE-2 Draft Head Training"
echo " Config: ${CONFIG}"
echo " GPUs: 8 (single node)"
echo " Started: $(date)"
echo "============================================"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="${MASTER_PORT}" \
  --rdzv_backend=static \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  train_eagle.py --config "${CONFIG}" "$@"

echo "============================================"
echo " EAGLE training complete: $(date)"
echo "============================================"
