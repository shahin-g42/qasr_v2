#!/bin/bash
#SBATCH --job-name=corpus-stage0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/corpus/stage0_%j.log
#SBATCH --error=logs/corpus/stage0_%j.err

# Stage 0 -- preflight (1 node, GPU-free, no corrector needed).
# Resolves every registry entry: repo access, column names, duration coverage.
# Exit 1 if any source fails, so a bad repo_id or missing token surfaces HERE,
# not three hours into Stage 1. Run this before every 9-node campaign.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

echo "[$(date)] stage0 preflight: root=${INTERNAL_ROOT}"

python3 -m data_processing.build_corpus preflight \
    --probe-fields --json --root "${INTERNAL_ROOT}" \
    > "${LOGS}/preflight_${SLURM_JOB_ID}.json"

echo "[$(date)] stage0 done: ${LOGS}/preflight_${SLURM_JOB_ID}.json"
