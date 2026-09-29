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
# Multi-machine, two shapes (stage 2 is the ceiling either way: one ledger,
# one writer per language):
#   * one node per language -- five nodes, no barrier:
#       LANGS="zh" scripts/corpus/run_internal_only.sh
#   * all nine nodes -- shard stage 1 across nine ranks, assemble on five:
#       phase A, on each node (R = that node's rank 0..8, keep the default
#       LANGS so every rank slices the same list):
#         PREPARE_ONLY=1 NODE_RANK=$R NUM_NODES=9 scripts/corpus/run_internal_only.sh
#       barrier: every rank's prepare report must exist before phase B
#       phase B, on the five assemble nodes (rank map 0=zh 1=hi 2=ar 3=en 4=ml):
#         SKIP_PREPARE=1 LANGS="zh" scripts/corpus/run_internal_only.sh
# NEVER point two nodes at the same language in phase B (or reuse a rank in
# phase A): the ledgers are node-local and cannot see each other, so both
# would claim the same paths and overwrite each other's batches/shards.
# Report names carry language and node tags, so shared-$LOGS runs never
# collide.
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
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
POOL_DIR_INT="${POOL_DIR_INT:-$QASR/corpus/pool_internal}"
OUT_DIR_INT="${OUT_DIR_INT:-$QASR/training_manifests/v8.0_internal}"
LEDGER_DIR_INT="${LEDGER_DIR_INT:-/scratch/corpus/ledgers_internal}"
LANGS_CSV="${LANGS// /,}"
# Report names carry the language filter so per-language nodes sharing one
# $LOGS never overwrite each other's reports.
LANGS_TAG="${LANGS// /_}"
# Stage-1 report: multi-node ranks each write one; single-node keeps the
# plain language-tagged name.
PREPARE_REPORT="$LOGS/prepare_internal_${LANGS_TAG}.json"
if [ "$NUM_NODES" -gt 1 ]; then
    PREPARE_REPORT="$LOGS/prepare_internal_${LANGS_TAG}_node${NODE_RANK}.json"
fi

[ -n "$LANGS" ] || { echo "ERROR: LANGS is empty" >&2; exit 1; }
if [ "$NUM_NODES" -lt 1 ] || [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NUM_NODES" ]; then
    echo "ERROR: bad sharding: NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES (need 0 <= rank < nodes)" >&2
    exit 1
fi
if [ "$NUM_NODES" -gt 1 ] && [ "$PREPARE_ONLY" != "1" ] && [ "$SKIP_PREPARE" != "1" ]; then
    echo "ERROR: NUM_NODES>1 shards stage 1; running it with assemble in the same command" >&2
    echo "       would let every node assemble every language into its own ledger. Use the" >&2
    echo "       two phases: PREPARE_ONLY=1 on all ranks, then SKIP_PREPARE=1 LANGS=<lang>." >&2
    exit 1
fi
mkdir -p "$POOL_DIR_INT" "$OUT_DIR_INT" "$LEDGER_DIR_INT"

for D in "$INTERNAL_ROOT" "$QASR_SFT_ROOT"; do
    [ -d "$D" ] || { echo "ERROR: internal tree not found: $D" >&2; exit 1; }
done
for L in $LANGS; do
    [ -d "$INTERNAL_ROOT/$L" ] || echo "WARNING: no v7.6 dir $INTERNAL_ROOT/$L" >&2
    [ -d "$QASR_SFT_ROOT/$L" ] || echo "WARNING: no SFT dir $QASR_SFT_ROOT/$L" >&2
done

# Gate on the co-located corrector before hours of work (mirrors stage 2).
# Skipped for phase A (PREPARE_ONLY=1): stage 1 needs no GPU, and the ranks
# that only shard stage 1 may not run a corrector at all.
if [ "$PREPARE_ONLY" != "1" ]; then
    READY=0
    for _ in $(seq 1 60); do
        if curl -sf http://localhost:8010/health > /dev/null 2>&1; then READY=1; break; fi
        sleep 5
    done
    [ "$READY" = "1" ] || { echo "ERROR: corrector not healthy on $(hostname) after 300s" >&2; exit 1; }
fi

# --- stage 1: internal sources only (no Hub access of any kind) --------------
if [ "$SKIP_PREPARE" != "1" ]; then
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
        --jobs "$JOBS" --node-rank "$NODE_RANK" --num-nodes "$NUM_NODES" \
        --report "$PREPARE_REPORT"
fi

# Phase A (nine-node split) stops after stage 1; assemble happens in phase B.
if [ "$PREPARE_ONLY" = "1" ]; then
    echo "[$(date)] phase A (prepare) done on rank ${NODE_RANK}/${NUM_NODES}: $POOL_DIR_INT"
    echo "next: wait until all ranks wrote $LOGS/prepare_internal_${LANGS_TAG}_node*.json,"
    echo "then assemble on the five nodes: SKIP_PREPARE=1 LANGS=<lang> $0"
    exit 0
fi

# Phase B sanity (SKIP_PREPARE=1): warn per language whose pool is missing a
# source's shards -- the usual cause is phase A not finished on every rank yet.
if [ "$SKIP_PREPARE" = "1" ]; then
    for L in $LANGS; do
        for SRC in "internal_v76_$L" "internal_sft_$L"; do
            compgen -G "$POOL_DIR_INT/$L/$SRC/part-*.jsonl*" > /dev/null \
                || echo "WARNING: no $L/$SRC pool shards under $POOL_DIR_INT -- is phase A finished on all ranks?" >&2
        done
    done
fi

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
if [ "$SKIP_PREPARE" = "1" ]; then
    echo "reports: $LOGS/assemble_internal_<lang>.json"
else
    echo "reports: $PREPARE_REPORT, $LOGS/assemble_internal_<lang>.json"
fi
