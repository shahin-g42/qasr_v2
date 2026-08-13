#!/bin/bash
# Reprocess REJECTED records on a single node (distributed mode).
# Usage: bash scripts/run_reprocess_node.sh <NODE_RANK> [REJECTED_DIR] [--verbose]
# Examples:
#   bash scripts/run_reprocess_node.sh 0
#   bash scripts/run_reprocess_node.sh 3 /path/to/cleaned_manifests --verbose
#
# Full language coverage: language is auto-detected per
# <REJECTED_DIR>/<lang>/*_rejected.jsonl subdir; Arabic uses the Arabic
# config, every other language the multilingual config. Big files are
# sliced across nodes; outputs are per-slice part files
# (<name>_recovered_pNNNN.jsonl) that the assembly step folds back in.
#
# Prerequisites:
#   - vLLM Docker container running on port 8010 on this node
#   - Code synced to /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr

set -euo pipefail

NODE_RANK=${1:?"Usage: bash scripts/run_reprocess_node.sh <NODE_RANK 0-7> [REJECTED_DIR] [--verbose]"}
REJECTED_DIR=${2:-/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/cleaned_manifests}
VERBOSE_FLAG=${3:-}
NUM_NODES=8

PROJECT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr"
VLLM_PORT=8010

echo "========================================"
echo "Reprocess Rejected Records"
echo "========================================"
echo "Node rank:    ${NODE_RANK}/${NUM_NODES}"
echo "Project:      ${PROJECT_DIR}"
echo "Rejected dir: ${REJECTED_DIR}"
echo "vLLM port:    ${VLLM_PORT}"
echo "========================================"

# Verify vLLM is reachable
echo "[$(date)] Checking vLLM health on port ${VLLM_PORT}..."
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at http://localhost:${VLLM_PORT}/health"
    echo "Start the same container as the cleaning pipeline (port ${VLLM_PORT})."
    exit 1
fi
echo "[$(date)] vLLM is healthy!"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

echo "[$(date)] Starting reprocess (node ${NODE_RANK}/${NUM_NODES})..."
python3 -m data_processing.reprocess \
    --rejected-dir "${REJECTED_DIR}" \
    --node-rank "${NODE_RANK}" \
    --num-nodes "${NUM_NODES}" \
    --slice-size 50000 \
    --slices-per-node 4 \
    --concurrency 512 \
    ${VERBOSE_FLAG}

echo "[$(date)] Node ${NODE_RANK} reprocess complete."
