#!/bin/bash
# Spread ONE huge manifest across multiple nodes by record-index range.
#
# Usage: bash scripts/run_clean_range.sh <SLICE_IDX> <NUM_SLICES> <MANIFEST_PATH> [arabic|multilingual] [--verbose] [--workers N]
# Example (train_ar_q3asr.jsonl across 3 nodes):
#   node A: bash scripts/run_clean_range.sh 0 3 /path/to/train_ar_q3asr.jsonl
#   node B: bash scripts/run_clean_range.sh 1 3 /path/to/train_ar_q3asr.jsonl
#   node C: bash scripts/run_clean_range.sh 2 3 /path/to/train_ar_q3asr.jsonl
# Workers: default 256 per node (override with --workers, e.g. --workers 96)
#
# Each slice gets its own outputs/checkpoints (…_r<START>_<END>_cleaned.jsonl)
# so nodes never collide, each slice is independently resumable (--resume),
# and --skip-processed makes every slice skip records an earlier partial
# full-manifest run already finished — no duplicated LLM work.
#
# Prerequisites: vLLM on port 8010 on this node; code synced to PROJECT_DIR.

set -euo pipefail

# Parse out the optional --workers flag anywhere in argv, then restore the
# positional arguments (SLICE_IDX, NUM_SLICES, MANIFEST_PATH, ...).
WORKERS=256
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="${2:?--workers requires a value}"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

SLICE_IDX=${1:?"Usage: bash scripts/run_clean_range.sh <SLICE_IDX> <NUM_SLICES> <MANIFEST_PATH> [arabic|multilingual] [--verbose] [--workers N]"}
NUM_SLICES=${2:?"NUM_SLICES required"}
MANIFEST_PATH=${3:?"MANIFEST_PATH required"}
PIPELINE=${4:-arabic}
VERBOSE_FLAG=${5:-}

PROJECT_DIR="/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr"
case "${PIPELINE}" in
    arabic)       CONFIG="${PROJECT_DIR}/configs/data_processing/arabic_cleaning.yaml" ;;
    multilingual) CONFIG="${PROJECT_DIR}/configs/data_processing/multilingual_cleaning.yaml" ;;
    *) echo "ERROR: unknown pipeline '${PIPELINE}' (use: arabic | multilingual)"; exit 1 ;;
esac
VLLM_PORT=8010

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "ERROR: manifest not found: ${MANIFEST_PATH}"; exit 1
fi
if [ "${SLICE_IDX}" -ge "${NUM_SLICES}" ]; then
    echo "ERROR: SLICE_IDX must be < NUM_SLICES"; exit 1
fi

# Manifest stem (strip .jsonl/.json) for --manifests filtering
BASENAME=$(basename "${MANIFEST_PATH}")
STEM="${BASENAME%.jsonl}"; STEM="${STEM%.json}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# Equal slices over the VALID-RECORD count, NOT the raw line count:
# --record-range indexes the pipeline's valid-record space (read_manifest
# skips blank/malformed lines), so wc -l boundaries would misalign slices
# across nodes and create overlapping coverage. Every node recomputes the
# same count with the same enumerator, so the partition is exact.
TOTAL_RECORDS=$(python3 -c "
from data_processing.manifest_io import count_manifest_records
import sys
print(count_manifest_records(sys.argv[1]))
" "${MANIFEST_PATH}")
START=$(( SLICE_IDX * TOTAL_RECORDS / NUM_SLICES ))
if [ "${SLICE_IDX}" -eq $(( NUM_SLICES - 1 )) ]; then
    END=""   # last slice: to EOF
else
    END=$(( (SLICE_IDX + 1) * TOTAL_RECORDS / NUM_SLICES ))
fi
RANGE="${START}:${END}"

echo "========================================"
echo "Range Processing (${PIPELINE})"
echo "========================================"
echo "Manifest:   ${BASENAME} (${TOTAL_RECORDS} valid records)"
echo "Slice:      ${SLICE_IDX}/${NUM_SLICES}  ->  records [${RANGE}]"
echo "Config:     ${CONFIG}"
echo "Workers:    ${WORKERS}"
echo "vLLM port:  ${VLLM_PORT}"
echo "========================================"

echo "[$(date)] Checking vLLM health on port ${VLLM_PORT}..."
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at http://localhost:${VLLM_PORT}/health"
    exit 1
fi
echo "[$(date)] vLLM is healthy!"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT_DIR}"

# node-rank 0 / num-nodes 1: distribution here is by --record-range,
# not by the round-robin manifest assignment.
echo "[$(date)] Starting slice ${SLICE_IDX}/${NUM_SLICES} of ${STEM}..."
python3 -m data_processing \
    --config "${CONFIG}" \
    --node-rank 0 \
    --num-nodes 1 \
    --manifests "${STEM}" \
    --record-range "${RANGE}" \
    --workers "${WORKERS}" \
    --skip-processed \
    --resume \
    ${VERBOSE_FLAG}

echo "[$(date)] Slice ${SLICE_IDX}/${NUM_SLICES} of ${STEM} complete."
