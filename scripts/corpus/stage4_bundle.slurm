#!/bin/bash
#SBATCH --job-name=corpus-stage4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/corpus/stage4_%j.log
#SBATCH --error=logs/corpus/stage4_%j.err

# Stage 4 -- bundle + audit (1 node, GPU-free). Groups the five languages' bN
# batches into bundles of exactly 100k x 5 (K = min full-batch count), audits
# every hard gate and writes $OUT_DIR/MANIFEST.json.
#
# THE EXIT CODE IS THE VERDICT: 0 = audited and manifested (shippable), 1 =
# audit failed or nothing is shippable yet (and any existing good MANIFEST is
# left untouched). Wire this job into your pipeline as the ship gate.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

echo "[$(date)] stage4 bundle: out=${OUT_DIR} audio_root=${AUDIO_ROOT}"

python3 -m data_processing.build_corpus bundle \
    --out-dir "${OUT_DIR}" \
    --audio-root "${AUDIO_ROOT}" \
    --root "${INTERNAL_ROOT}" \
    --eval-root "${QASR_SFT_ROOT}" \
    --report "${LOGS}/bundle_${SLURM_JOB_ID}.json"

echo "[$(date)] stage4 done: bundles manifested and audited"
