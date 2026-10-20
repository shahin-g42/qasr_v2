#!/usr/bin/env bash
#SBATCH --job-name=aud-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=aud-export-%j.log
set -euo pipefail
: "${AUD_REPO_ROOT:?Set AUD_REPO_ROOT to the shared qasr checkout before sbatch}"
cd "$AUD_REPO_ROOT"
bash audm/scripts/export_all.sh
