#!/bin/bash
# SLURM batch launcher for the 64-GPU JEV qualification/production job.
#
# One torchrun agent per node via srun; every agent joins a single c10d
# rendezvous at the first allocated host. The node count comes from the
# allocation itself (SLURM_NNODES) and must match the config's replicate.
# Submit from the JEV root (or a location seeing the same filesystem on all
# nodes):
#
#   sbatch launch/slurm_8x8.sbatch                            # default config
#   sbatch launch/slurm_8x8.sbatch my/config.toml -- --resume latest
#
# Run launch/preflight.sh on a compute node first; the same file paths must be
# visible on all nodes. Adjust --partition, --account, --time, and the output
# location for your cluster before submitting.
#SBATCH --job-name=jev-8x8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=jev-8x8-%j.out

set -euo pipefail

JEV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$JEV_ROOT/configs/train/h100_8x8.toml}"
NODES="${SLURM_NNODES:-8}"
shift "$(( $# > 1 ? 1 : $# ))" || true
if [ "${1:-}" = "--" ]; then shift; fi

HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export JEV_RDZV_ENDPOINT="${JEV_RDZV_ENDPOINT:-$HEAD_NODE:29500}"
export JEV_RDZV_ID="${JEV_RDZV_ID:-jev-${SLURM_JOB_ID:-manual}}"

echo "jev 8x8 job on $SLURM_JOB_NODELIST"
echo "config=$CONFIG nodes=$NODES rendezvous=$JEV_RDZV_ENDPOINT id=$JEV_RDZV_ID"

# One agent per node; torchrun_8x8.sh fails before rendezvous when the local
# GPU visibility or the config mesh does not match this launch.
srun --ntasks-per-node=1 --cpu-bind=v \
  bash "$JEV_ROOT/launch/torchrun_8x8.sh" "$CONFIG" "$NODES" "$@"
