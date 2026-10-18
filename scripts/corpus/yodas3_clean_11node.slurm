#!/usr/bin/env bash
#SBATCH --job-name=yodas3-clean
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/yodas3-clean-%j.out
# Correctors must already be healthy on these nodes; ASR runs on two other nodes.
# Use the per-node run command directly for your existing interactive allocations.
set -euo pipefail
: "${ASR_URLS:?Export both ASR base URLs before submission}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export FOREGROUND=1 RUN_ID="${RUN_ID:-yodas3_v1}"
srun --ntasks=11 --ntasks-per-node=1 bash scripts/corpus/run_yodas3.sh run
