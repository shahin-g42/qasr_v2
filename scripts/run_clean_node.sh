#!/bin/bash
# Run transcript processing on a single node.
# Usage: bash scripts/run_clean_node.sh <NODE_RANK> [arabic|multilingual] [--verbose]
# Examples:
#   bash scripts/run_clean_node.sh 0                        # Arabic, INFO logging
#   bash scripts/run_clean_node.sh 0 multilingual           # en/zh/hi/ml, INFO logging
#   bash scripts/run_clean_node.sh 0 arabic --verbose       # DEBUG logging
#
# Prerequisites:
#   - vLLM Docker container running on port 8010 on this node
#   - Code synced to /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr
#
# Output locations (namespaced by language):
#   Cleaned manifests: .../cleaned_manifests/<lang>/
#   Checkpoints:       .../cleaning_checkpoints/<lang>/
#   Temp shards:       .../cleaning_shards/<lang>/

set -euo pipefail

NODE_RANK=${1:?"Usage: bash scripts/run_clean_node.sh <NODE_RANK 0-7> [arabic|multilingual] [--verbose]"}
PIPELINE=${2:-arabic}
VERBOSE_FLAG=${3:-}
NUM_NODES=8

PROJECT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr"
case "${PIPELINE}" in
    arabic)       CONFIG="${PROJECT_DIR}/configs/data_processing/arabic_cleaning.yaml" ;;
    multilingual) CONFIG="${PROJECT_DIR}/configs/data_processing/multilingual_cleaning.yaml" ;;
    *) echo "ERROR: unknown pipeline '${PIPELINE}' (use: arabic | multilingual)"; exit 1 ;;
esac
VLLM_PORT=8010

echo "========================================"
echo "Transcript Processing (${PIPELINE})"
echo "========================================"
echo "Node rank:  ${NODE_RANK}/${NUM_NODES}"
echo "Project:    ${PROJECT_DIR}"
echo "Config:     ${CONFIG}"
echo "vLLM port:  ${VLLM_PORT}"
echo "========================================"

# Verify vLLM is reachable
echo "[$(date)] Checking vLLM health on port ${VLLM_PORT}..."
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at http://localhost:${VLLM_PORT}/health"
    echo "Make sure the Docker container is running:"
    echo "  docker run -d --restart unless-stopped --gpus all --shm-size=128g -p ${VLLM_PORT}:8000 \\"
    echo "    -v /lustrefs/shared/shahin.konadath/workspace/train/qasr/.cache:/root/.cache/huggingface \\"
    echo "    --name vllm-server vllm/vllm-openai:v0.26.0 \\"
    echo "    --model Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 8 \\"
    echo "    --max-model-len 16384 --max-num-seqs 512 --gpu-memory-utilization 0.93"
    exit 1
fi
echo "[$(date)] vLLM is healthy!"

# Set up Python path
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

# Run the pipeline (add --verbose as 3rd script arg for DEBUG logging)
echo "[$(date)] Starting processing pipeline (node ${NODE_RANK}/${NUM_NODES})..."
python3 -m data_processing \
    --config "${CONFIG}" \
    --node-rank "${NODE_RANK}" \
    --num-nodes "${NUM_NODES}" \
    --resume \
    ${VERBOSE_FLAG}

echo "[$(date)] Node ${NODE_RANK} processing complete."
