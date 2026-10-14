#!/usr/bin/env bash
#SBATCH --job-name=gqasr
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
set -euo pipefail
CONFIG="${1:?Usage: sbatch slurm_12node.sh CONFIG [--resume CHECKPOINT]}"
shift
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="$SLURM_NNODES"
SCRIPT_DIR="${GEMMA_SPEECH_ROOT:-${SLURM_SUBMIT_DIR:?}/gemma_speech}/gqasr/scripts"
srun --kill-on-bad-exit=1 bash -c 'export NODE_RANK="$SLURM_PROCID"; exec "$@"' \
  bash "$SCRIPT_DIR/torchrun_node.sh" "$CONFIG" "$@"
