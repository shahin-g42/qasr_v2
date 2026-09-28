#!/bin/bash
#SBATCH --job-name=corpus-stage2b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/corpus/stage2b_%j.log
#SBATCH --error=logs/corpus/stage2b_%j.err

# Stage 2b -- OPTIONAL accelerator, run CONCURRENTLY with stage2 on the four
# nodes assemble does not use (5-8). Each prewarm node walks a language's pool
# in composite order and materializes the top external candidates ahead of
# selection (budget ~1.3x one 100k batch); assemble then verify-and-skips
# them. Prewarming changes no placement decision -- it only moves fetch
# latency off the assemble critical path. Allocation by external share: en is
# external-only (heaviest), hi is gated-indic-heavy, so they get their own
# nodes; ar and ml share the fourth (ar first, then ml).
#
# Concurrent writers to the same deterministic path are safe: materialize is
# idempotent (atomic sibling-tmp + rename, same bytes from the same source).

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

NODE_SCRIPT="${LOGS}/stage2b_${SLURM_JOB_ID}.node.sh"
trap 'rm -f "${NODE_SCRIPT}"' EXIT

cat > "${NODE_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail
RANK=${SLURM_PROCID}
case "${RANK}" in
    0) LANGS="en" ;;
    1) LANGS="hi" ;;
    2) LANGS="ar" ;;
    3) LANGS="zh ml" ;;
    *) echo "unexpected rank ${RANK}"; exit 1 ;;
esac
echo "[$(date)] [node ${RANK}] prewarming: ${LANGS} on $(hostname)"
for LG in ${LANGS}; do
    python3 -m data_processing.build_corpus materialize \
        --lang "${LG}" \
        --pool-dir "${POOL_DIR}" \
        --audio-root "${AUDIO_ROOT}" \
        --prewarm --budget 130000 \
        --report "${LOGS}/prewarm_${LG}_${SLURM_JOB_ID}.json"
done
echo "[$(date)] [node ${RANK}] prewarm done"
EOF

srun bash "${NODE_SCRIPT}"
