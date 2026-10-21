#!/usr/bin/env bash
#SBATCH --job-name=aud-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=aud-train-%j.log
set -euo pipefail
: "${AUD_REPO_ROOT:?Set AUD_REPO_ROOT to the shared qasr checkout before sbatch}"
cd "$AUD_REPO_ROOT"
export AUD_USE_SLURM=1
bash audm/scripts/train_distributed.sh "$@"
