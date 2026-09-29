#!/bin/bash
#SBATCH --job-name=corpus-stage2
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/corpus/stage2_%j.log
#SBATCH --error=logs/corpus/stage2_%j.err

# Stage 2 -- assemble, ONE NODE PER LANGUAGE (5 of the 9; the corrector is
# co-located on every node at localhost:8010). Language pinning by this job's
# SLURM_PROCID:
#
#   0=zh  1=hi  2=ar  3=en  4=ml
#
# zh/hi ride the Flash-Next nodes (ranks 0-1); each language's ledger is
# owned by exactly one process ($LEDGER_DIR -- workspace scratch by default,
# override for node-local disk). No --gres: the container owns the GPUs.
#
# The LLM knobs (prompt family, batch, concurrency, fetch workers) come from
# the assemble: section of configs/corpus.yaml via the dispatcher; identity,
# paths and the drain mode are explicit here so a YAML accident cannot change
# WHAT is built, only how fast.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

NODE_SCRIPT="${LOGS}/stage2_${SLURM_JOB_ID}.node.sh"
trap 'rm -f "${NODE_SCRIPT}"' EXIT

cat > "${NODE_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail
RANK=${SLURM_PROCID}
case "${RANK}" in
    0) LG=zh ;; 1) LG=hi ;; 2) LG=ar ;; 3) LG=en ;; 4) LG=ml ;;
    *) echo "unexpected rank ${RANK}"; exit 1 ;;
esac
echo "[$(date)] [${LG}] assemble on $(hostname)"

# Gate on the corrector before spending hours: fail in 5 minutes, not mid-run.
READY=0
for _ in $(seq 1 60); do
    if curl -sf http://localhost:8010/health > /dev/null 2>&1; then READY=1; break; fi
    sleep 5
done
if [ "${READY}" != "1" ]; then
    echo "ERROR: corrector not healthy on $(hostname) after 300s" >&2
    exit 1
fi

python3 -m data_processing.build_corpus assemble \
    --lang "${LG}" \
    --pool-dir "${POOL_DIR}" \
    --out-dir "${OUT_DIR}" \
    --ledger "${LEDGER_DIR}/${LG}.sqlite3" \
    --audio-root "${AUDIO_ROOT}" \
    --root "${INTERNAL_ROOT}" \
    --eval-root "${QASR_SFT_ROOT}" \
    --llm-url "${CORRECTOR_URL}" \
    --batches 0 \
    --exclude-eval \
    --report "${LOGS}/assemble_${LG}_${SLURM_JOB_ID}.json"
echo "[$(date)] [${LG}] assemble done"
EOF

# No --kill-on-bad-exit: languages are independent; a ledger makes every rank
# resumable. srun returns the worst task exit code once all finish.
srun bash "${NODE_SCRIPT}"
