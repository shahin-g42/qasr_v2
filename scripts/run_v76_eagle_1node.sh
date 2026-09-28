#!/bin/bash
# =============================================================================
# QASR v7.6 — PHASE 4 (ADHOC): EAGLE draft head, single-node launcher.
#
# 1 node x 8 H100 GPUs. No SLURM, no multi-node rendezvous — ssh onto the
# spare node and run this directly. Derived from run_v76_phase4_eagle_node.sh
# with NNODES pinned to 1 and a pre-flight checkpoint guard added.
#
#   bash scripts/run_v76_eagle_1node.sh
#   CONFIG=configs/v7.6/04_eagle_1node.yaml bash scripts/run_v76_eagle_1node.sh
#
# Extra args are forwarded to train_eagle.py, which accepts ONLY:
#   --model --output-dir --train-manifest --max-steps --batch-size --lr
#   --num-draft-tokens --kl-temperature
# Anything else (e.g. --smoke-test, which belongs to train.py) is rejected by
# argparse. For a smoke run, edit max_steps + allow_partial_epoch in the YAML:
# allow_partial_epoch has no CLI flag, so --max-steps alone trips the
# full-coverage guard and raises.
#
# Default config: configs/v7.6/04_eagle_1node.yaml
#   target:  models/v7.6/full/checkpoint-150000  (Phase-2 MID-checkpoint)
#   output:  output/v76/eagle_v2_1node_ckpt150k/eagle_head.pt
#   epoch:   2,408,582 steps = one full stratified epoch at world_size=8 over
#            the FULL Phase-2 pool (44 manifests, ~176.5M records). That is
#            roughly 2-8 weeks on one node. For a shorter adhoc run, flip the
#            partial-epoch preset in the YAML (max_steps: 301073,
#            allow_partial_epoch: true) = 12.5% coverage, ~2-7 days.
#
# The head is bound to exact target weights. This adhoc head is THROWAWAY —
# retrain on the final Phase-3 output before shipping.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env)
# ---------------------------------------------------------------------------
CONFIG="${CONFIG:-configs/v7.6/04_eagle_1node.yaml}"
NNODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONDA_ENV="${CONDA_ENV:-/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3}"
PROJECT_DIR="${PROJECT_DIR:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr}"
HF_CACHE="${HF_CACHE:-/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache}"

# ---------------------------------------------------------------------------
# Environment (identical tuning to scripts/qasr_train_8node.slurm)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_CACHE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
# Single node: no InfiniBand needed, but leave NCCL_IB_DISABLE=0 so NCCL can
# still use IB for intra-node NVLink fallback paths on some H100 boards.
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
# Known-good H100 tuning:
export NCCL_BUFFSIZE=8388608
export CUDA_DEVICE_MAX_CONNECTIONS=1
# JIT build caches MUST be node-local: the home dir is NFS, and 8 ranks
# racing a DeepSpeed/Triton build lock on NFS deadlocks the first step.
export TORCH_EXTENSIONS_DIR="/tmp/${USER}_torch_ext"
export TRITON_CACHE_DIR="/tmp/${USER}_triton"
# wandb keeps a write-ahead log under WANDB_DIR; same rule as the JIT caches --
# node-local, never Lustre/NFS. WANDB_PROJECT/WANDB_LOG_MODEL/WANDB_WATCH are
# set from the YAML by train_eagle.py itself, so only the directory is pinned
# here. If this spare node has no wandb credentials or no egress, launch with
# WANDB_MODE=offline and `wandb sync` the run dir later -- otherwise wandb.init
# can block waiting for a login prompt and stall all 8 ranks.
export WANDB_DIR="${WANDB_DIR:-/tmp/${USER}_wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${WANDB_DIR}"
# do NOT set NCCL_ALGO=Tree — breaks ncclInt8 broadcast on NCCL 2.27.5.

# ---------------------------------------------------------------------------
# Activate conda and enter project
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src"

# Cluster transformers drift guard: the recipe is verified against the pinned
# 5.14.x wheel only.
python -c "import transformers; assert transformers.__version__.startswith('5.14'), \
    f'transformers {transformers.__version__} != pinned 5.14.x'"

mkdir -p logs

# ---------------------------------------------------------------------------
# Pre-flight: verify the target checkpoint is complete BEFORE the multi-GB
# teacher load. Mirrors validate_local_checkpoint() in src/qasr/config.py but
# fails in the launcher (one error) instead of on all 8 ranks (eight errors).
# ---------------------------------------------------------------------------
CKPT="$(python -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['model_name_or_path'])")"
OUT="$(python -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG}'))['eagle_output_dir'])")"
echo "============================================================"
echo " EAGLE-3 adhoc training — 1 node x ${NPROC_PER_NODE} GPUs"
echo "============================================================"
echo " Config:        ${CONFIG}"
echo " Target ckpt:   ${CKPT}"
echo " Output dir:    ${OUT}"
echo " Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo " World size:    $((NNODES * NPROC_PER_NODE))"
echo " Conda env:     ${CONDA_ENV}"
echo " Started:       $(date)"
echo "============================================================"

if [[ ! -d "${CKPT}" ]]; then
    echo "FATAL: target checkpoint dir does not exist on this node: ${CKPT}" >&2
    echo "       Check the Lustre mount and the path in ${CONFIG}." >&2
    exit 1
fi
for required in config.json; do
    if [[ ! -f "${CKPT}/${required}" ]]; then
        echo "FATAL: ${CKPT}/${required} missing — checkpoint is incomplete." >&2
        exit 1
    fi
done
if ! ls "${CKPT}"/model*.safetensors >/dev/null 2>&1; then
    echo "FATAL: no model*.safetensors in ${CKPT} — checkpoint is incomplete." >&2
    echo "       (Phase-2 used ZeRO-2, so weights should be consolidated here." >&2
    echo "        If this is a ZeRO-3 sharded checkpoint, run convert_weights first.)" >&2
    exit 1
fi
if [[ ! -f "${CKPT}/preprocessor_config.json" && ! -f "${CKPT}/processor_config.json" ]]; then
    echo "FATAL: neither preprocessor_config.json nor processor_config.json in ${CKPT}." >&2
    echo "       QASRProcessor.from_pretrained() will fail on all 8 ranks." >&2
    exit 1
fi
mkdir -p "${OUT}"
echo " Pre-flight OK: checkpoint complete, output dir ready."
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch with torchrun (single-node static rendezvous on localhost).
# Extra args ("$@") are forwarded to train_eagle.py. See the header for the
# nine flags it accepts; unknown flags are rejected by argparse, not ignored.
# ---------------------------------------------------------------------------
exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_conf=timeout=7200 \
    train_eagle.py --config "${CONFIG}" "$@"
