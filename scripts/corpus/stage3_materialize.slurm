#!/bin/bash
#SBATCH --job-name=corpus-stage3
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/corpus/stage3_%j.log
#SBATCH --error=logs/corpus/stage3_%j.err

# Stage 3 -- verify/backfill one node per language: every external clip the
# written batches reference must exist at 16 kHz mono FLAC (idempotent --
# correct files are skipped, so this also mops up after a crashed stage 2).
# Run it before stage 4; the bundle audit re-checks anyway, but finding a
# missing clip HERE means the fix runs with 32 workers instead of failing the
# audit.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

NODE_SCRIPT="${LOGS}/stage3_${SLURM_JOB_ID}.node.sh"
trap 'rm -f "${NODE_SCRIPT}"' EXIT

cat > "${NODE_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail
RANK=${SLURM_PROCID}
case "${RANK}" in
    0) LG=zh ;; 1) LG=hi ;; 2) LG=ar ;; 3) LG=en ;; 4) LG=ml ;;
    *) echo "unexpected rank ${RANK}"; exit 1 ;;
esac
echo "[$(date)] [${LG}] materialize backfill on $(hostname)"
python3 -m data_processing.build_corpus materialize \
    --out-dir "${OUT_DIR}" \
    --lang "${LG}" \
    --pool-dir "${POOL_DIR}" \
    --audio-root "${AUDIO_ROOT}" \
    --workers 32 \
    --report "${LOGS}/materialize_${LG}_${SLURM_JOB_ID}.json"
echo "[$(date)] [${LG}] materialize done"
EOF

srun bash "${NODE_SCRIPT}"
