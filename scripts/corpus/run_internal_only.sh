#!/bin/bash
# Internal-only corpus build: prepare + assemble/correct the TWO internal trees
# (v7.6 cleaned manifests + raw q3asr SFT envelopes) with no Hub source in the
# loop -- no `datasets` library, no HF_TOKEN, no downloads, no docker twin.
# Run it in a manual session on any ONE node whose corrector is healthy
# (localhost:8010); the LLM stage is the only thing that needs the node.
#
# How it differs from the full campaign, and why:
#   * Separate pool/out/ledger by default ($QASR/corpus/pool_internal,
#     $QASR/training_manifests/v8.0_internal, /scratch/corpus/ledgers_internal)
#     so this build cannot touch the campaign's pools, claims or batches.
#     The ledger is node-local: resume on the SAME node, or copy the
#     <lang>.sqlite3 files first (SQLite-WAL must not live on Lustre).
#   * --max-per-source-fraction 0.5 (env MAX_PER_SOURCE_FRACTION): every
#     language has exactly TWO internal sources (internal_v76_<lang> +
#     internal_sft_<lang>); at the campaign's 0.40 cap two sources can fill at
#     most 80% of a batch, so a full batch could never close. 0.5 is the
#     maximum-diversity setting for a two-source build (50/50).
#   * --no-materialize: internal audio is read in place, there is nothing
#     external to fetch, and stage 3 is not needed at all.
#   * --batches 0 (drain): each language gets one partial batch of whatever
#     the gates admit -- an internal-only partial batch is NOT a campaign
#     bundle. zh is EXPECTED to come out tiny or refused: 153k paths over
#     7.8k transcripts is the known failure mode the diversity floor exists
#     for; the campaign counts on externals to fill zh.
#
# Knobs (environment): LANGS, JOBS, BATCH_SIZE, MAX_PER_SOURCE_FRACTION,
# LIMIT (prepare-only row cap, e.g. LIMIT=500 for a smoke run), BUNDLE=1
# (also run the stage-4 audit -- expect exit 1 while any language lacks a
# FULL batch; the report JSON is the point), and POOL_DIR_INT / OUT_DIR_INT /
# LEDGER_DIR_INT to relocate the build.
#
# Multi-machine: run ONE node per language (LANGS="<lang>"), five nodes for
# five languages -- that is the parallelism ceiling (one ledger, one writer
# per language; stage 1 has ten atomic source units, >=2 per node either way).
# NEVER point two nodes at the same language: the ledgers are node-local and
# cannot see each other, so both would claim the same paths and overwrite
# each other's batch files. Report names carry the language filter, so
# per-node reports never collide in the shared $LOGS.
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/env.sh"

LANGS="${LANGS:-ar en zh hi ml}"
JOBS="${JOBS:-4}"
BATCH_SIZE="${BATCH_SIZE:-100000}"
MAX_PER_SOURCE_FRACTION="${MAX_PER_SOURCE_FRACTION:-0.5}"
LIMIT="${LIMIT:-}"
BUNDLE="${BUNDLE:-0}"
POOL_DIR_INT="${POOL_DIR_INT:-$QASR/corpus/pool_internal}"
OUT_DIR_INT="${OUT_DIR_INT:-$QASR/training_manifests/v8.0_internal}"
LEDGER_DIR_INT="${LEDGER_DIR_INT:-/scratch/corpus/ledgers_internal}"
LANGS_CSV="${LANGS// /,}"
# Report names carry the language filter so per-language nodes sharing one
# $LOGS never overwrite each other's reports.
LANGS_TAG="${LANGS// /_}"

[ -n "$LANGS" ] || { echo "ERROR: LANGS is empty" >&2; exit 1; }
mkdir -p "$POOL_DIR_INT" "$OUT_DIR_INT" "$LEDGER_DIR_INT"

for D in "$INTERNAL_ROOT" "$QASR_SFT_ROOT"; do
    [ -d "$D" ] || { echo "ERROR: internal tree not found: $D" >&2; exit 1; }
done
for L in $LANGS; do
    [ -d "$INTERNAL_ROOT/$L" ] || echo "WARNING: no v7.6 dir $INTERNAL_ROOT/$L" >&2
    [ -d "$QASR_SFT_ROOT/$L" ] || echo "WARNING: no SFT dir $QASR_SFT_ROOT/$L" >&2
done

# Gate on the co-located corrector before hours of work (mirrors stage 2).
READY=0
for _ in $(seq 1 60); do
    if curl -sf http://localhost:8010/health > /dev/null 2>&1; then READY=1; break; fi
    sleep 5
done
[ "$READY" = "1" ] || { echo "ERROR: corrector not healthy on $(hostname) after 300s" >&2; exit 1; }

# --- stage 1: internal sources only (no Hub access of any kind) --------------
ONLY_ARGS=()
for L in $LANGS; do
    ONLY_ARGS+=(--only "internal_v76_$L" --only "internal_sft_$L")
done
LIMIT_ARGS=()
[ -n "$LIMIT" ] && LIMIT_ARGS=(--limit "$LIMIT")
python3 -m data_processing.build_corpus prepare \
    --pool-dir "$POOL_DIR_INT" --audio-root "$AUDIO_ROOT" \
    --root "$INTERNAL_ROOT" --langs "$LANGS_CSV" \
    "${ONLY_ARGS[@]}" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
    --jobs "$JOBS" \
    --report "$LOGS/prepare_internal_${LANGS_TAG}.json"

# --- stage 2: assemble + correct each language, drain -------------------------
for L in $LANGS; do
    echo "[$(date)] [${L}] internal assemble on $(hostname)"
    python3 -m data_processing.build_corpus assemble \
        --lang "$L" \
        --pool-dir "$POOL_DIR_INT" --out-dir "$OUT_DIR_INT" \
        --ledger "${LEDGER_DIR_INT}/${L}.sqlite3" \
        --audio-root "$AUDIO_ROOT" \
        --root "$INTERNAL_ROOT" --exclude-eval --eval-root "$QASR_SFT_ROOT" \
        --llm-url "$CORRECTOR_URL" \
        --batches 0 --no-materialize \
        --batch-size "$BATCH_SIZE" \
        --max-per-source-fraction "$MAX_PER_SOURCE_FRACTION" \
        --report "$LOGS/assemble_internal_${L}.json"
done

# --- optional stage 4: audit. Exit 1 while any language has no full batch is
# EXPECTED here (that is the K = min contract doing its job); the JSON report
# is what carries the numbers, so a non-zero bundle exit does not fail this
# script.
if [ "$BUNDLE" = "1" ]; then
    python3 -m data_processing.build_corpus bundle \
        --out-dir "$OUT_DIR_INT" --audio-root "$AUDIO_ROOT" \
        --root "$INTERNAL_ROOT" --eval-root "$QASR_SFT_ROOT" \
        --langs "$LANGS_CSV" \
        --batch-size "$BATCH_SIZE" \
        --max-per-source-fraction "$MAX_PER_SOURCE_FRACTION" \
        --report "$LOGS/bundle_internal_${LANGS_TAG}.json" \
        || echo "NOTE: internal-only audit not green (expected while any language lacks a full batch); see $LOGS/bundle_internal_${LANGS_TAG}.json" >&2
fi

echo "[$(date)] internal-only build done: $OUT_DIR_INT"
echo "reports: $LOGS/prepare_internal_${LANGS_TAG}.json, $LOGS/assemble_internal_<lang>.json"
