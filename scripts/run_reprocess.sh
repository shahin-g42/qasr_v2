#!/usr/bin/env bash
# Reprocess REJECTED records only (no full re-run) through the improved
# pipeline (v0.2.0 prompts + homoglyph hard check + fixed audit classifier).
#
# Run on a node with the vLLM server up (port 8010). Resumable: safe to
# re-run after interruption — already-processed records are skipped.
#
# Usage:
#   bash scripts/run_reprocess.sh                    # all languages
#   bash scripts/run_reprocess.sh ar hi              # selected languages
#
# Outputs (next to each *_rejected.jsonl):
#   <name>_recovered.jsonl       — now pass validation -> merge into training
#   <name>_still_rejected.jsonl  — still fail after rework
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST_DIR="${MANIFEST_DIR:-cleaned_manifests}"
CONCURRENCY="${CONCURRENCY:-512}"     # matches vllm --max-num-seqs
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
    LANGS=(ar en hi zh ml)
fi

config_for() {
    case "$1" in
        ar) echo "configs/data_processing/arabic_cleaning.yaml" ;;
        *)  echo "configs/data_processing/multilingual_cleaning.yaml" ;;
    esac
}

for lang in "${LANGS[@]}"; do
    dir="${MANIFEST_DIR}/${lang}"
    if [ ! -d "${dir}" ]; then
        echo "[skip] ${dir} not found"
        continue
    fi
    if ! ls "${dir}"/*_rejected.jsonl >/dev/null 2>&1; then
        echo "[skip] ${dir}: no *_rejected.jsonl files"
        continue
    fi
    log="${LOG_DIR}/reprocess_${lang}_$(date +%Y%m%d_%H%M%S).log"
    echo "[${lang}] reprocessing rejected files in ${dir} (log: ${log})"
    PYTHONPATH=src python -m data_processing.reprocess \
        --config "$(config_for "${lang}")" \
        --rejected-dir "${dir}" \
        --language "${lang}" \
        --concurrency "${CONCURRENCY}" \
        2>&1 | tee "${log}"
done

# The ml rejected file is ~75% actual Hindi audio (upstream contamination).
# Recover those as HINDI data — route through the hi agents. Copy under a
# distinct stem first: reprocess.py names outputs after the input stem, and
# sharing outputs with the normal ml pass above would make resume-dedupe
# silently skip every record here.
if [[ " ${LANGS[*]} " == *" ml "* ]] && [ -f "${MANIFEST_DIR}/ml/eval_ml_inworld_rejected.jsonl" ]; then
    echo "[ml->hi] recovering Hindi-contaminated ml rejects as Hindi"
    cp "${MANIFEST_DIR}/ml/eval_ml_inworld_rejected.jsonl" \
       "${MANIFEST_DIR}/ml/eval_ml_inworld_as_hi_rejected.jsonl"
    PYTHONPATH=src python -m data_processing.reprocess \
        --config configs/data_processing/multilingual_cleaning.yaml \
        --rejected-dir "${MANIFEST_DIR}/ml" \
        --files "eval_ml_inworld_as_hi_rejected.jsonl" \
        --language hi \
        --concurrency "${CONCURRENCY}" \
        2>&1 | tee "${LOG_DIR}/reprocess_ml_as_hi_$(date +%Y%m%d_%H%M%S).log"
    echo "NOTE: ml/eval_ml_inworld_as_hi_recovered.jsonl is HINDI data —"
    echo "      move it into the hi training set, not ml."
fi

echo "Done. Merge per corpus with:"
echo "  cat <name>_cleaned.jsonl <name>_recovered.jsonl > <name>_final.jsonl"
