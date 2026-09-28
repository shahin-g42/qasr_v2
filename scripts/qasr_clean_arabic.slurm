#!/bin/bash
#SBATCH --job-name=qasr-clean-ar
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/qasr_clean_arabic_%j.log
#SBATCH --error=logs/qasr_clean_arabic_%j.err

# Arabic Transcript Processing Pipeline
# 8 nodes x 8 GPUs (vLLM TP=8) x 96 async workers per node (one per CPU core)
# Processes ~70M Arabic records through Qwen3.6-35B-A3B
#
# Every node runs its own vLLM server plus its own pipeline rank: srun fans
# the per-node script below out to all allocated nodes (--ntasks-per-node=1),
# where SLURM_PROCID is the node index and becomes --node-rank.

set -euo pipefail

# Environment
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache"
export TOKENIZERS_PARALLELISM=false

# Conda environment
CONDA_ENV="/lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Node info
NUM_NODES=${SLURM_NNODES}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "========================================"
echo "Arabic Transcript Processing Pipeline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${NUM_NODES}"
echo "Master: ${MASTER_ADDR}"
echo "CPUs per node: ${SLURM_CPUS_PER_TASK}"
echo "GPUs per node: 8 (for vLLM)"
echo "========================================"

# vLLM server settings (port 8010 to match vllm_port in
# configs/data_processing/arabic_cleaning.yaml)
VLLM_MODEL="Qwen/Qwen3.6-35B-A3B"
VLLM_PORT=8010
VLLM_TP=8
export VLLM_MODEL VLLM_PORT VLLM_TP

# ---------------------------------------------------------------------------
# Per-node script: each node starts its own vLLM server, waits for it to be
# healthy, runs its pipeline rank, and always shuts its server down (via the
# EXIT trap) — even when the pipeline fails. Written to the shared submit dir
# so every node can read it.
# ---------------------------------------------------------------------------
NODE_SCRIPT="${SLURM_SUBMIT_DIR}/logs/qasr_clean_arabic_${SLURM_JOB_ID}.node.sh"
trap 'rm -f "${NODE_SCRIPT}"' EXIT

cat > "${NODE_SCRIPT}" <<'NODE_EOF'
#!/bin/bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
NODE_RANK=${SLURM_PROCID}
NUM_NODES=${SLURM_NNODES}

# Start vLLM server in background
echo "[$(date)] [node ${NODE_RANK}] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --tensor-parallel-size ${VLLM_TP} \
    --max-model-len 16384 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.93 \
    --port ${VLLM_PORT} \
    --disable-log-requests &

VLLM_PID=$!
echo "[$(date)] [node ${NODE_RANK}] vLLM server PID: ${VLLM_PID}"

# Always shut the server down on exit — set -e must not leak it.
trap 'kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# Wait for vLLM to be healthy
echo "[$(date)] [node ${NODE_RANK}] Waiting for vLLM server to be ready..."
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] [node ${NODE_RANK}] vLLM server is healthy!"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "[$(date)] [node ${NODE_RANK}] Waiting... (${ELAPSED}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: [node ${NODE_RANK}] vLLM server failed to start within ${MAX_WAIT}s"
    exit 1
fi

# Run the processing pipeline; || keeps set -e from aborting before we can
# report the failure and let the EXIT trap clean up the server.
echo "[$(date)] [node ${NODE_RANK}] Starting processing pipeline..."
PIPELINE_EXIT=0
python -m data_processing \
    --config configs/data_processing/arabic_cleaning.yaml \
    --node-rank ${NODE_RANK} \
    --num-nodes ${NUM_NODES} || PIPELINE_EXIT=$?

echo "[$(date)] [node ${NODE_RANK}] Pipeline finished with exit code: ${PIPELINE_EXIT}"
echo "[$(date)] [node ${NODE_RANK}] Shutting down vLLM server..."
exit ${PIPELINE_EXIT}
NODE_EOF

# Launch one copy per node. No --kill-on-bad-exit: pipeline ranks are
# independent and resumable, so one failing node must not discard the others'
# progress; srun still returns the worst task exit code once all finish.
SRUN_EXIT=0
srun bash "${NODE_SCRIPT}" || SRUN_EXIT=$?

echo "[$(date)] All nodes finished; worst exit code: ${SRUN_EXIT}"
echo "[$(date)] Done."
exit ${SRUN_EXIT}
