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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "========================================"
echo "Arabic Transcript Processing Pipeline"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node rank: ${NODE_RANK}/${NUM_NODES}"
echo "Master: ${MASTER_ADDR}"
echo "CPUs per node: ${SLURM_CPUS_PER_TASK}"
echo "GPUs per node: 8 (for vLLM)"
echo "========================================"

# vLLM server settings
VLLM_MODEL="Qwen/Qwen3.6-35B-A3B"
VLLM_PORT=8000
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

# Run the processing pipeline
echo "[$(date)] Starting processing pipeline..."
python -m data_processing \
    --config configs/data_processing/arabic_cleaning.yaml \
    --node-rank ${NODE_RANK} \
    --num-nodes ${NUM_NODES}

PIPELINE_EXIT=$?

# Cleanup
echo "[$(date)] Pipeline finished with exit code: ${PIPELINE_EXIT}"
echo "[$(date)] Shutting down vLLM server..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "[$(date)] Done."
exit ${PIPELINE_EXIT}
