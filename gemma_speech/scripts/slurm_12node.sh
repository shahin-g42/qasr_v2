#!/usr/bin/env bash
#SBATCH --job-name=gemma-speech
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --output=gemma-speech-%j.out
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm_12node.sh /shared/experiment.json [train options]" >&2
  exit 2
fi
: "${SLURM_JOB_NODELIST:?Run this script using sbatch}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29541}"
export NNODES="$SLURM_JOB_NUM_NODES"
export NPROC_PER_NODE=8
srun --cpu-bind=none bash -c \
  'export NODE_RANK="$SLURM_PROCID"; exec bash "$1" "${@:2}"' \
  _ "$script_dir/torchrun_node.sh" "$@"
