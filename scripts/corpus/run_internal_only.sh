#!/bin/bash
# Internal-only corpus build: prepare + assemble/correct the TWO internal trees
# (v7.6 cleaned manifests + raw q3asr SFT envelopes) with no Hub source in the
# loop -- no `datasets` library, no HF_TOKEN, no downloads, no docker twin.
# Run it on EVERY node of a manual session, with that node's rank; every stage
# then runs on all nodes (see "Multi-node" below). The LLM assembly stage is
# the only one that needs a healthy co-located corrector (localhost:8010).
#
# How it differs from the full campaign, and why:
#   * Separate pool/out/ledger by default ($QASR/corpus/pool_internal,
#     $QASR/training_manifests/v8.0_internal,
#     $QASR/scratch/corpus/ledgers_internal) so this build cannot touch the
#     campaign's pools, claims or batches. Each ledger is SQLite-WAL with
#     exactly one writer and lives under the workspace (any node can resume);
#     point LEDGER_DIR_INT at node-local disk when the cluster offers one.
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
# (also run the stage-4 audit at the end -- expect exit 1 while any language
# lacks a FULL batch; the report JSON is the point), BUNDLE_ONLY=1 (skip
# stages 1-2 and only audit), COVERAGE_CHECK=0 (skip the tree-vs-config
# coverage gate), BARRIER_TIMEOUT (seconds to wait for stage 1 on the other
# ranks; 0 = wait forever), and POOL_DIR_INT / OUT_DIR_INT / LEDGER_DIR_INT
# to relocate the build.
#
# Multi-node (the default shape): run the SAME command on every node of a
# NUM_NODES-node job, with NODE_RANK = 0..NUM_NODES-1:
#
#   NODE_RANK=$R NUM_NODES=9 scripts/corpus/run_internal_only.sh
#
# Each rank then: prepares its slice of every language (stage 1); waits at a
# barrier until all NUM_NODES ranks finished theirs; runs
# plan_distribution.py, which ANALYSES the finished pools, decides how many
# hash slices each language gets and which rank runs which slice; and finally
# assembles its own (language, slice) assignments. Slices are keyed by
# blake2b(audio_filepath), so a clip's duplicate rows (the v7.6 cleaned and
# q3asr SFT copies share the path) always land in ONE slice; each slice keeps
# its own ledger (<lang>_p<k>.sqlite3) and writes part k of every
# batch at batch_size / slices rows per part, so the parts of a label compose
# one whole batch. Single-node (NUM_NODES=1, the default) runs the same flow
# with one whole-language writer per language -- the legacy shape.
#
# Resuming: re-run the same command on the same nodes, starting every rank
# close together. A full re-run drops each rank's marker and redoes its
# stage 1 (prepare is authoritative per source, not incremental), so a rank
# that has not been restarted yet still shows its PREVIOUS run's marker and
# the barrier may pass early -- start all ranks within a few minutes of each
# other. For a stage-2-only resume, re-run with SKIP_PREPARE=1: the markers
# stay valid, the barrier passes at once, the plan is a pure function of the
# finished pools, and each ledger continues its label sequence. The slice
# count per language is pinned in $LOGS/slices_internal_<lang>.txt at first
# assembly: a run that would change it (different NUM_NODES, or pools so
# different that the plan changes) is refused -- start a fresh
# OUT_DIR_INT/LEDGER_DIR_INT for a new layout.
#
# NOT the planner: NEVER point two processes at the same language AND the same
# ledger. A ledger file is owned by exactly one process -- two writers would
# claim the same paths and overwrite each other's batches/shards. One language
# per node still works (five nodes, no slicing):
#   NUM_NODES=1 LANGS="zh" scripts/corpus/run_internal_only.sh
# Report names carry language, slice and node tags, so shared-$LOGS runs never
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
BUNDLE_ONLY="${BUNDLE_ONLY:-0}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-0}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
POOL_DIR_INT="${POOL_DIR_INT:-$QASR/corpus/pool_internal}"
OUT_DIR_INT="${OUT_DIR_INT:-$QASR/training_manifests/v8.0_internal}"
LEDGER_DIR_INT="${LEDGER_DIR_INT:-$QASR/scratch/corpus/ledgers_internal}"
LANGS_CSV="${LANGS// /,}"
# Report/marker names carry the language filter so per-language nodes sharing
# one $LOGS never overwrite each other's reports.
LANGS_TAG="${LANGS// /_}"
PLAN_FILE="$LOGS/plan_internal_${LANGS_TAG}.json"
DONE_MARKER="$LOGS/prepare_done_internal_${LANGS_TAG}_node${NODE_RANK}"
# Stage-1 report: multi-node ranks each write one; single-node keeps the
# plain language-tagged name.
PREPARE_REPORT="$LOGS/prepare_internal_${LANGS_TAG}.json"
if [ "$NUM_NODES" -gt 1 ]; then
    PREPARE_REPORT="$LOGS/prepare_internal_${LANGS_TAG}_node${NODE_RANK}.json"
fi
if [ "$BUNDLE_ONLY" = "1" ]; then
    BUNDLE=1
fi

[ -n "$LANGS" ] || { echo "ERROR: LANGS is empty" >&2; exit 1; }
if [ "$NUM_NODES" -lt 1 ] || [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NUM_NODES" ]; then
    echo "ERROR: bad sharding: NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES (need 0 <= rank < nodes)" >&2
    exit 1
fi
mkdir -p "$POOL_DIR_INT" "$OUT_DIR_INT" "$LEDGER_DIR_INT"

# A run that will redo stage 1 invalidates this rank's previous barrier marker
# up front -- before any gate. If this rank then exits early (coverage or
# corrector failure), the barrier waits for it loudly instead of passing on a
# stale marker; a fresh start drops the stale marker within milliseconds.
# SKIP_PREPARE (phase B / stage-2 resume) keeps the marker: it represents
# stage-1 completion.
if [ "$SKIP_PREPARE" != "1" ] && [ "$BUNDLE_ONLY" != "1" ]; then
    rm -f "$DONE_MARKER"
fi

for D in "$INTERNAL_ROOT" "$QASR_SFT_ROOT"; do
    [ -d "$D" ] || { echo "ERROR: internal tree not found: $D" >&2; exit 1; }
done
for L in $LANGS; do
    [ -d "$INTERNAL_ROOT/$L" ] || echo "WARNING: no v7.6 dir $INTERNAL_ROOT/$L" >&2
    [ -d "$QASR_SFT_ROOT/$L" ] || echo "WARNING: no SFT dir $QASR_SFT_ROOT/$L" >&2
done

# Coverage gate: the trees on disk must match configs/corpus/internal_ingest.yaml
# file-for-file -- every listed train file ingestible, no unlisted train file
# that would sneak in, every listed eval file excluded AND gate-visible. Runs in
# every mode. COVERAGE_CHECK=0 bypasses it for a deliberate partial tree.
COVERAGE_CHECK="${COVERAGE_CHECK:-1}"
if [ "$COVERAGE_CHECK" = "1" ]; then
    if ! python3 "${SCRIPTS_DIR}/check_internal_sources.py" \
        --internal-root "$INTERNAL_ROOT" --sft-root "$QASR_SFT_ROOT"; then
        echo "ERROR: internal trees do not match configs/corpus/internal_ingest.yaml (report above)." >&2
        echo "       Fix the tree or the config, or re-run with COVERAGE_CHECK=0 to bypass." >&2
        exit 1
    fi
fi

# Gate on the co-located corrector before hours of work (mirrors stage 2).
# Skipped for phase A (PREPARE_ONLY=1: stage 1 needs no GPU) and for an
# audit-only run (BUNDLE_ONLY=1: no GPU either).
if [ "$PREPARE_ONLY" != "1" ] && [ "$BUNDLE_ONLY" != "1" ]; then
    READY=0
    for _ in $(seq 1 60); do
        if curl -sf http://localhost:8010/health > /dev/null 2>&1; then READY=1; break; fi
        sleep 5
    done
    [ "$READY" = "1" ] || { echo "ERROR: corrector not healthy on $(hostname) after 300s" >&2; exit 1; }
fi

if [ "$BUNDLE_ONLY" != "1" ]; then
    # --- stage 1: internal sources only (no Hub access of any kind) ----------
    if [ "$SKIP_PREPARE" != "1" ]; then
        # (The barrier marker was already dropped before the gates -- see the
        # comment above the tree checks.)
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
        touch "$DONE_MARKER"
        echo "[$(date)] stage 1 done on rank ${NODE_RANK}/${NUM_NODES} (marker: $DONE_MARKER)"
    fi

    # Phase A (prepare-only) stops after stage 1; assemble is a later run.
    if [ "$PREPARE_ONLY" = "1" ]; then
        echo "[$(date)] phase A (prepare) done on rank ${NODE_RANK}/${NUM_NODES}: $POOL_DIR_INT"
        echo "next: once $LOGS/prepare_done_internal_${LANGS_TAG}_node* exist for every rank, run on"
        echo "each rank: NUM_NODES=$NUM_NODES NODE_RANK=<that rank> SKIP_PREPARE=1 $0"
        exit 0
    fi

    # Stage-2 sanity: warn per language whose pool is missing a source's
    # shards -- the usual cause is stage 1 not finished on every rank yet.
    for L in $LANGS; do
        for SRC in "internal_v76_$L" "internal_sft_$L"; do
            compgen -G "$POOL_DIR_INT/$L/$SRC/part-*.jsonl*" > /dev/null \
                || echo "WARNING: no $L/$SRC pool shards under $POOL_DIR_INT -- is stage 1 finished on all ranks?" >&2
        done
    done

    # --- barrier: every rank finished its stage 1 (multi-node only) ----------
    if [ "$NUM_NODES" -gt 1 ]; then
        echo "[$(date)] rank ${NODE_RANK}/${NUM_NODES}: barrier -- waiting for stage 1 on all ranks"
        WAITED=0
        while :; do
            MISSING=0
            for R in $(seq 0 $((NUM_NODES - 1))); do
                if [ ! -f "$LOGS/prepare_done_internal_${LANGS_TAG}_node${R}" ]; then
                    MISSING=$((MISSING + 1))
                fi
            done
            if [ "$MISSING" -eq 0 ]; then
                break
            fi
            if [ "$BARRIER_TIMEOUT" -gt 0 ] && [ "$WAITED" -ge "$BARRIER_TIMEOUT" ]; then
                echo "ERROR: barrier timed out after ${BARRIER_TIMEOUT}s: ${MISSING} of ${NUM_NODES} stage-1 marker(s) missing" >&2
                exit 1
            fi
            if [ $((WAITED % 60)) -eq 0 ]; then
                echo "[$(date)] waiting: ${MISSING} of ${NUM_NODES} stage-1 marker(s) missing (waited ${WAITED}s)"
            fi
            sleep 10
            WAITED=$((WAITED + 10))
        done
        echo "[$(date)] barrier passed: stage 1 is complete on all ${NUM_NODES} ranks"
    fi

    # --- stage 1.5: analyse the pools, plan the slice layout, take my rows ---
    # Every rank computes the same plan (pure function of the finished pools)
    # and keeps only its own TSV rows; the human table goes to stderr.
    if ! ASSIGNMENTS="$(python3 "${SCRIPTS_DIR}/plan_distribution.py" \
            --pool-dir "$POOL_DIR_INT" --langs "$LANGS_CSV" \
            --nodes "$NUM_NODES" --batch-size "$BATCH_SIZE" \
            --json "$PLAN_FILE" --assign-rank "$NODE_RANK")"; then
        echo "ERROR: stage-2 planning failed (see above); no pool shards for $LANGS_CSV under $POOL_DIR_INT?" >&2
        exit 1
    fi

    # --- stage 2: assemble each assigned (language, slice) pair, drain -------
    COUNT=0
    while IFS=$'\t' read -r L PART NSLICES; do
        if [ -z "$L" ]; then
            continue
        fi
        if [ "$((BATCH_SIZE % NSLICES))" -ne 0 ]; then
            echo "ERROR: BATCH_SIZE=$BATCH_SIZE is not divisible by the $NSLICES slices planned for $L" >&2
            exit 1
        fi
        SLICE_ARGS=()
        if [ "$NSLICES" -gt 1 ]; then
            SLICE_ARGS=(--pool-part "$PART/$NSLICES" --batch-part "$PART")
            LEDGER_FILE="${LEDGER_DIR_INT}/${L}_p${PART}.sqlite3"
            REPORT_FILE="$LOGS/assemble_internal_${L}_p${PART}.json"
        else
            LEDGER_FILE="${LEDGER_DIR_INT}/${L}.sqlite3"
            REPORT_FILE="$LOGS/assemble_internal_${L}.json"
        fi

        # The slice count for a language is pinned once assembled: a changed
        # layout would put new parts under old labels and break the batch
        # contract (and differently-laid-out ledgers claim the same paths).
        MARKER="$LOGS/slices_internal_${L}.txt"
        if [ -f "$MARKER" ]; then
            RECORDED="$(tr -d '[:space:]' < "$MARKER")"
            if [ "$RECORDED" != "$NSLICES" ]; then
                echo "ERROR: $L was assembled with $RECORDED slice(s) (pinned in $MARKER)," >&2
                echo "       but this run plans $NSLICES. Keep the same NUM_NODES and pools, or" >&2
                echo "       start a fresh OUT_DIR_INT + LEDGER_DIR_INT for a new layout." >&2
                exit 1
            fi
        else
            printf '%s\n' "$NSLICES" > "$MARKER"
        fi
        # Existing parts must fit this layout: a p0004 where only 0..3 belong
        # means a wider earlier run wrote into this out dir.
        for F in "$OUT_DIR_INT/$L"/train_"$L"_*_p*.jsonl*; do
            if [ -e "$F" ]; then
                P="${F##*_p}"
                P="${P%%.*}"
                if [ "$((10#$P))" -ge "$NSLICES" ]; then
                    echo "ERROR: $OUT_DIR_INT/$L already holds part p$P, outside the $NSLICES-slice" >&2
                    echo "       layout pinned for $L; use a fresh OUT_DIR_INT or restore the layout." >&2
                    exit 1
                fi
            fi
        done

        echo "[$(date)] [${L} part ${PART}/${NSLICES}] internal assemble on $(hostname)"
        python3 -m data_processing.build_corpus assemble \
            --lang "$L" \
            --pool-dir "$POOL_DIR_INT" --out-dir "$OUT_DIR_INT" \
            --ledger "$LEDGER_FILE" \
            --audio-root "$AUDIO_ROOT" \
            --root "$INTERNAL_ROOT" --exclude-eval --eval-root "$QASR_SFT_ROOT" \
            --llm-url "$CORRECTOR_URL" \
            --batches 0 --no-materialize \
            --batch-size "$((BATCH_SIZE / NSLICES))" \
            ${SLICE_ARGS[@]+"${SLICE_ARGS[@]}"} \
            --max-per-source-fraction "$MAX_PER_SOURCE_FRACTION" \
            --report "$REPORT_FILE"
        COUNT=$((COUNT + 1))
    done <<< "$ASSIGNMENTS"

    if [ "$COUNT" -eq 0 ]; then
        echo "[$(date)] no stage-2 work for rank ${NODE_RANK}: every slice is assigned to another rank"
    fi
fi

# --- optional stage 4: audit. Exit 1 while any language has no full batch is
# EXPECTED here (that is the K = min contract doing its job); the JSON report
# is what carries the numbers, so a non-zero bundle exit does not fail this
# script. Multi-node runs audit on rank 0 only (same input, no stampede).
RUN_AUDIT=0
if [ "$BUNDLE" = "1" ]; then
    if [ "$NUM_NODES" -eq 1 ] || [ "$NODE_RANK" -eq 0 ]; then
        RUN_AUDIT=1
    else
        echo "[$(date)] rank ${NODE_RANK}: the stage-4 audit is rank 0's job; skipping here"
    fi
fi
if [ "$RUN_AUDIT" = "1" ]; then
    python3 -m data_processing.build_corpus bundle \
        --out-dir "$OUT_DIR_INT" --audio-root "$AUDIO_ROOT" \
        --root "$INTERNAL_ROOT" --eval-root "$QASR_SFT_ROOT" \
        --langs "$LANGS_CSV" \
        --batch-size "$BATCH_SIZE" \
        --max-per-source-fraction "$MAX_PER_SOURCE_FRACTION" \
        --report "$LOGS/bundle_internal_${LANGS_TAG}.json" \
        || echo "NOTE: internal-only audit not green (expected while any language lacks a full batch); see $LOGS/bundle_internal_${LANGS_TAG}.json" >&2
fi

echo "[$(date)] internal-only build done on rank ${NODE_RANK}/${NUM_NODES}: $OUT_DIR_INT"
if [ "$BUNDLE_ONLY" = "1" ]; then
    echo "reports: $LOGS/bundle_internal_${LANGS_TAG}.json"
elif [ "$PREPARE_ONLY" = "1" ]; then
    echo "reports: $PREPARE_REPORT (phase A only)"
elif [ "$SKIP_PREPARE" = "1" ]; then
    echo "reports: $LOGS/assemble_internal_<lang>.json (+ _p<part> for sliced languages); plan: $PLAN_FILE"
else
    echo "reports: $PREPARE_REPORT, $LOGS/assemble_internal_<lang>[_p<part>].json; plan: $PLAN_FILE"
fi
