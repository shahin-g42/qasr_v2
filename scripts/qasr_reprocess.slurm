#!/bin/bash
#SBATCH --job-name=qasr-reprocess
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=logs/qasr_reprocess_%j.log
#SBATCH --error=logs/qasr_reprocess_%j.err

# Reprocess Rejected Records — 8 nodes, full language coverage
# 8 nodes x 8 GPUs (vLLM TP=8) re-runs ONLY the rejected records
# (cleaned_manifests/<lang>/*_rejected.jsonl) through the improved
# cleaner+validator. Big files are sliced and dealt round-robin across
# nodes; each slice writes its own *_recovered_pNNNN.jsonl part file.
# Fold results back with scripts/assemble_training_manifests.py.

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
NODE_RANK=${SLURM_NODEID}
NUM_NODES=${SLURM_NNODES}

REJECTED_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/cleaned_manifests"

echo "========================================"
echo "Reprocess Rejected Records (all languages)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node rank: ${NODE_RANK}/${NUM_NODES}"
echo "CPUs per node: ${SLURM_CPUS_PER_TASK}"
echo "GPUs per node: 8 (for vLLM)"
echo "========================================"

# vLLM server settings (port 8010 to match both pipeline configs)
VLLM_MODEL="Qwen/Qwen3.6-35B-A3B"
VLLM_PORT=8010
VLLM_TP=8

# Start vLLM server in background
echo "[$(date)] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --tensor-parallel-size ${VLLM_TP} \
    --max-model-len 16384 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.93 \
    --port ${VLLM_PORT} \
    --disable-log-requests &

VLLM_PID=$!
echo "[$(date)] vLLM server PID: ${VLLM_PID}"

# Wait for vLLM to be healthy
echo "[$(date)] Waiting for vLLM server to be ready..."
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server is healthy!"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "[$(date)] Waiting... (${ELAPSED}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# Run the distributed reprocess
echo "[$(date)] Starting reprocess (node ${NODE_RANK}/${NUM_NODES})..."
python -m data_processing.reprocess \
    --rejected-dir "${REJECTED_DIR}" \
    --node-rank ${NODE_RANK} \
    --num-nodes ${NUM_NODES} \
    --slice-size 50000 \
    --slices-per-node 4 \
    --concurrency 512

PIPELINE_EXIT=$?

# Cleanup
echo "[$(date)] Reprocess finished with exit code: ${PIPELINE_EXIT}"
echo "[$(date)] Shutting down vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "[$(date)] Done. Fold recovered records with:"
echo "  python3 scripts/assemble_training_manifests.py --version vNEXT"
exit ${PIPELINE_EXIT}
