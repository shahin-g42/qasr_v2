#!/bin/bash
# Prepare the q3asr SFT manifests on a single node (one shard of the work).
# Usage: bash scripts/run_q3asr_prepare_node.sh <NODE_RANK> [NUM_NODES] [--verbose]
# Examples:
#   bash scripts/run_q3asr_prepare_node.sh 0          # rank 0 of 8
#   bash scripts/run_q3asr_prepare_node.sh 3 8 --verbose
#
# Run this on ALL nodes (rank 0..NUM_NODES-1), then merge once from any node:
#   PYTHONPATH=src python3 scripts/merge_q3asr_shards.py \
#       --output-dir <OUTPUT_DIR> --num-nodes 8
#
# Prerequisites (same as scripts/run_clean_node.sh):
#   - vLLM container running on port 8010 on THIS node. Each node needs its own
#     server: the work is LLM-bound, so pointing 8 nodes at one server just
#     moves the queue rather than shortening it.
#   - Code synced to /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr
#
# Each node writes <split>_<lang>_q3asr.rank<N>of<M>.jsonl — never a shared
# path, because the writer truncates and nodes would overwrite each other.

set -euo pipefail

NODE_RANK=${1:?"Usage: bash scripts/run_q3asr_prepare_node.sh <NODE_RANK> [NUM_NODES] [--verbose]"}
NUM_NODES=${2:-8}
VERBOSE_FLAG=${3:-}

PROJECT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr"
SFT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/tts/vllm/qasr/data/asr/.dset/dpo/enriched/q3asr/json/sft"
OUTPUT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/q3asr_sft_manifests"
CONFIG="${PROJECT_DIR}/configs/data_processing/multilingual_cleaning.yaml"
# Tokenizer used to measure transcript length against max_target_length (512).
# Any QASR checkpoint works — fine-tuning does not change the vocabulary — but
# it must be a QASR model dir, not the upstream Qwen repo. Override if your
# training config loads a different model_name_or_path.
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROJECT_DIR}/initial}"
VLLM_PORT=8010

echo "========================================"
echo "q3asr SFT manifest preparation"
echo "========================================"
echo "Node rank:  ${NODE_RANK}/${NUM_NODES}"
echo "Project:    ${PROJECT_DIR}"
echo "Inputs:     ${SFT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Tokenizer:  ${TOKENIZER_DIR}"
echo "vLLM port:  ${VLLM_PORT}"
echo "========================================"

# Verify vLLM is reachable on THIS node.
echo "[$(date)] Checking vLLM health on port ${VLLM_PORT}..."
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at http://localhost:${VLLM_PORT}/health"
    echo "Start it as in scripts/run_clean_node.sh:"
    echo "  docker run -d --restart unless-stopped --gpus all --shm-size=128g -p ${VLLM_PORT}:8000 \\"
    echo "    -v ${PROJECT_DIR%/stt/qasr}/qasr/.cache:/root/.cache/huggingface \\"
    echo "    --name vllm-server vllm/vllm-openai:v0.26.0 \\"
    echo "    --model Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 8 \\"
    echo "    --max-model-len 16384 --max-num-seqs 512 --gpu-memory-utilization 0.93"
    exit 1
fi
echo "[$(date)] vLLM is healthy!"

if [ ! -d "${TOKENIZER_DIR}" ]; then
    echo "ERROR: tokenizer dir not found: ${TOKENIZER_DIR}"
    echo "Transcripts over max_target_length are not truncated by training — each"
    echo "is replaced by a duplicate of another sample — so the real tokenizer is"
    echo "required to find them. Set TOKENIZER_DIR=<model dir> and re-run, or add"
    echo "--no-length-check below to accept that loss."
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_DIR}"

echo "[$(date)] Starting node ${NODE_RANK}/${NUM_NODES}..."
python3 scripts/prepare_q3asr_filter.py \
    --config "${CONFIG}" \
    --inputs "${SFT_DIR}/train_filter.jsonl" "${SFT_DIR}/eval_filter.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --node-rank "${NODE_RANK}" \
    --num-nodes "${NUM_NODES}" \
    ${VERBOSE_FLAG}

echo "[$(date)] Node ${NODE_RANK} complete."
echo ""
echo "When ALL ${NUM_NODES} nodes have finished, merge from any one node:"
echo "  PYTHONPATH=src python3 scripts/merge_q3asr_shards.py \\"
echo "      --output-dir ${OUTPUT_DIR} --num-nodes ${NUM_NODES}"
