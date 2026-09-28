#!/bin/bash
# =============================================================================
# QASR v7.6 — PHASE 4: EAGLE draft head, per-node manual launcher.
#
# Same 8-node pattern as the training phases, but a DIFFERENT entry point:
# torchrun launches train_eagle.py (EagleTrainConfig), not train.py.
#
#   Node 0 (master):  bash scripts/run_v76_phase4_eagle_node.sh 0 <master-hostname>
#   Node N:           bash scripts/run_v76_phase4_eagle_node.sh N <master-hostname>
#
# Train ONLY on the final Phase 3 checkpoint (the config points at
# output/v76/hq): the head is bound to exact target weights, and any later
# target change invalidates it. Global batch 64 x 30 = 1,920; 103,194 steps
# = one full stratified epoch. The epoch-coverage guard and the checkpoint
# completeness guard both apply here too.
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
CONFIG="${CONFIG:-configs/v7.6/04_eagle_8node.yaml}"
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
    train_eagle.py --config "${CONFIG}" "$@"
