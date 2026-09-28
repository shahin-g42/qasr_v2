#!/bin/bash
#SBATCH --job-name=qasr-eagle-8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

# =============================================================================
# EAGLE-3 Draft Head Training for QASR (8 nodes x 8 GPUs = 64 GPUs)
#
# Trains only the ~8.4M parameter EAGLE head (fusion layer + layer-mix
# weights) on a frozen QASR checkpoint. Global batch = 64 GPUs x 32 = 2,048,
# 10,000 steps (~1-2 h). Launch AFTER the Phase 2 checkpoint is final:
#
#   sbatch --export=ALL,CONFIG=configs/train_eagle_8node_filtered.yaml \
#     scripts/qasr_train_eagle_8node.slurm
#
# Acceptance gate (afterwards):
#   python evaluate.py --model output/full_v2/best --eagle output/eagle_v3 \
#     --manifest configs/eval_en_inworld.json --language en \
#     --num-samples 500 --seed 42 --modes eagle --output eval_eagle_v3/en.json
#   pass = acceptance >= 0.5, tokens/forward >= 2.5, exact_transcript_match 100%
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via --export)
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/train_eagle_8node_filtered.yaml}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/.cache}"
MASTER_PORT="${MASTER_PORT:-29501}"

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
# Node-local JIT caches: DeepSpeed/Triton build locks on the NFS home
# deadlock multi-rank first steps.
export TORCH_EXTENSIONS_DIR="/tmp/${USER}_torch_ext"
export TRITON_CACHE_DIR="/tmp/${USER}_triton"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
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

# Cluster transformers drift guard: the recipe is verified against the
# pinned 5.14.x wheel only.
python -c "import transformers; assert transformers.__version__.startswith('5.14'), \
    f'transformers {transformers.__version__} != pinned 5.14.x'"

mkdir -p logs

# ---------------------------------------------------------------------------
# SLURM multi-node discovery
# ---------------------------------------------------------------------------
NNODES="${SLURM_NNODES:-8}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

echo "============================================================"
echo " EAGLE-3 Draft Head Training (8-node topology)"
echo "============================================================"
echo " Job ID:        ${SLURM_JOB_ID:-local}"
echo " Config:        ${CONFIG}"
echo " Nodes:         ${NNODES}"
echo " Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo " Total GPUs:    $((NNODES * 8))"
echo " Started:       $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch with torchrun (static rendezvous on master:29501).
# srun fans one torchrun launcher out to every allocated node
# (--ntasks-per-node=1), where SLURM_PROCID is the node index and becomes
# --node_rank. The exported env above (NCCL, HF, conda PATH, PYTHONPATH)
# propagates into the step; extra sbatch args ("$@") are forwarded.
# ---------------------------------------------------------------------------
export NNODES MASTER_ADDR MASTER_PORT CONFIG
srun --kill-on-bad-exit=1 bash -c '
  exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node=8 \
    --node_rank="${SLURM_PROCID}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_conf=timeout=3600 \
    train_eagle.py \
      --config "${CONFIG}" \
      "$@"
' _ "$@"

echo "============================================================"
echo " EAGLE training complete on all ${NNODES} nodes at $(date)"
echo "============================================================"
