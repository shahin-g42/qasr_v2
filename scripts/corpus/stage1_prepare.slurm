#!/bin/bash
#SBATCH --job-name=corpus-stage1
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/corpus/stage1_%j.log
#SBATCH --error=logs/corpus/stage1_%j.err

# Stage 1 -- prepare on ALL 9 nodes (GPU-free). SLURM_PROCID is the node rank;
# the registry's (lang, source) pairs are sharded rank % 9, so the nodes write
# disjoint pool directories and never coordinate. --jobs fans each node's
# sources over spawned worker processes (96-core nodes; 4 workers leaves room
# for the OS and the corrector container).
#
# Re-runnable: a re-run of a source clears that source's stale pool shards
# first, so each source's pool is authoritative. One crashed source is
# reported in that node's JSON, never fatal to the node.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

NODE_SCRIPT="${LOGS}/stage1_${SLURM_JOB_ID}.node.sh"
trap 'rm -f "${NODE_SCRIPT}"' EXIT

cat > "${NODE_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail
RANK=${SLURM_PROCID}
echo "[$(date)] [node ${RANK}] prepare on $(hostname)"
python3 -m data_processing.build_corpus prepare \
    --pool-dir "${POOL_DIR}" \
    --audio-root "${AUDIO_ROOT}" \
    --root "${INTERNAL_ROOT}" \
    --node-rank "${RANK}" \
    --num-nodes "${SLURM_NNODES}" \
    --jobs 4 \
    --report "${LOGS}/prepare_node${RANK}_${SLURM_JOB_ID}.json"
echo "[$(date)] [node ${RANK}] prepare done"
EOF

# No --kill-on-bad-exit: ranks are independent and resumable, so one failing
# node must not discard the others' pools; srun still returns the worst code.
srun bash "${NODE_SCRIPT}"
