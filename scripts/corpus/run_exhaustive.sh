#!/usr/bin/env bash
# Exhaustive internal-only corpus build across the nine-node cluster.
#
#   NODE_RANK=R NUM_NODES=9 RUN_ID=v8int scripts/corpus/run_exhaustive.sh
#
# Rank 0 prepares the exact internal_ingest.yaml contract, builds the unique
# worksets, and freezes the hash partitions. Then EVERY rank processes its
# assigned (lang, part) slices with 80 concurrent corrector requests. All
# stages are idempotent: rerunning a node after a crash resumes its slices
# from the durable cursors and never duplicates accepted rows.
set -euo pipefail
cd "$(dirname "$0")/../.."

NODE_RANK="${NODE_RANK:?NODE_RANK must be set (0..NUM_NODES-1)}"
NUM_NODES="${NUM_NODES:-9}"
RUN_ID="${RUN_ID:?RUN_ID must identify this run on all nodes}"
JOBS="${JOBS:-80}"
CONCURRENCY="${LLM_CONCURRENCY:-80}"
BATCH="${LLM_BATCH:-16}"
MAX_TOKENS="${LLM_MAX_TOKENS:-12288}"
URL="${CORRECTOR_URL:-http://localhost:8010/v1}"
LANGS="${LANGS:-ar en hi ml zh}"

# shellcheck source=env.sh
source scripts/corpus/env.sh

RUN_ROOT="${RUN_ROOT:-$QASR/scratch/corpus/exhaustive_$RUN_ID}"
PY="${PYTHON:-python}"
mkdir -p "$RUN_ROOT" "$LOGS"
echo "run_root : $RUN_ROOT"
echo "run_id   : $RUN_ID  (rank $NODE_RANK/$NUM_NODES)"

CONTRACT="$QASR/configs/corpus/internal_ingest.yaml"
V76_ROOT="$INTERNAL_ROOT"
SFT_ROOT="$QASR_SFT_ROOT"

log() { echo "[$(date '+%a %b %d %H:%M:%S %Z')] $*"; }

_read_status() {
    # Status field of a stage report, or empty when absent/invalid.
    $PY - "$1" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", ""))
except Exception:
    pass
PYEOF
}

# ---- rank-local preparation (every rank does its own byte ranges) ----------
if [ ! -f "$RUN_ROOT/prepare/rank$NODE_RANK.json" ] \
    || [ "$(_read_status "$RUN_ROOT/prepare/rank$NODE_RANK.json" 2>/dev/null || echo)" != "complete" ]; then
    log "stage 1: prepare rank $NODE_RANK ($JOBS task buckets)"
    $PY - "$RUN_ROOT" "$CONTRACT" "$V76_ROOT" "$SFT_ROOT" "$NODE_RANK" "$NUM_NODES" "$JOBS" "$RUN_ID" <<'PYEOF'
import sys
from data_processing import workset
root, contract, v76, sft, rank, nodes, jobs, run_id = sys.argv[1:9]
report = workset.prepare_rank(root, contract_path=contract, v76_root=v76,
                              sft_root=sft, rank=int(rank), nodes=int(nodes),
                              jobs=int(jobs), run_id=run_id)
print(f"rank {rank}: {report['train_rows']} train rows, status={report['status']}")
PYEOF
else
    log "stage 1: prepare rank $NODE_RANK already complete (resume)"
fi

# ---- barrier: all ranks must finish preparation before workset build ------
for r in $(seq 0 $((NUM_NODES - 1))); do
    while [ "$(_read_status "$RUN_ROOT/prepare/rank$r.json" 2>/dev/null || echo)" != "complete" ]; do
        sleep 30
    done
done
log "barrier passed: preparation complete on all $NUM_NODES ranks"

# ---- worksets + partitions (all ranks call; owners + markers make it safe) -
for lang in $LANGS; do
    $PY - "$RUN_ROOT" "$CONTRACT" "$V76_ROOT" "$SFT_ROOT" "$lang" "$NUM_NODES" "$JOBS" "$RUN_ID" <<'PYEOF'
import sys
from data_processing import workset
root, contract, v76, sft, lang, nodes, jobs, run_id = sys.argv[1:9]
finalized = workset.finalize_worksets(root)
if finalized.get("status") != "complete":
    raise SystemExit(f"workset finalization incomplete: {finalized.get('status')}")
inv = finalized["inventories"][lang]
workset.build_language(root, lang, run_id=run_id)
report = workset.partition_language(root, lang, max(1, inv["est_rows"] // 2_000_000) if inv["est_rows"] else 1)
print(f"{lang}: {inv['eligible']} unique clips -> {report['nparts']} part(s)")
PYEOF
done

# ---- barrier: wait for every partition marker ------------------------------
for lang in $LANGS; do
    while [ "$(_read_status "$RUN_ROOT/partitions/$lang.json" 2>/dev/null || echo)" != "complete" ]; do
        sleep 30
    done
done
log "barrier passed: partitions frozen"

# ---- stage 2: this rank's (lang, part) slices -------------------------------
# Slices are assigned round-robin over ranks for each language's part list.
SLICES=$($PY - "$RUN_ROOT" "$LANGS" "$NODE_RANK" "$NUM_NODES" <<'PYEOF'
import json, sys
from pathlib import Path
root, langs, rank, nodes = Path(sys.argv[1]), sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4])
mine = []
for lang in langs:
    plan = json.loads((root / "partitions" / f"{lang}.json").read_text())
    for part in range(plan["nparts"]):
        if part % nodes == rank:
            mine.append(f"{lang} {part}")
print(";".join(mine))
PYEOF
)

for slice in ${SLICES//;/ }; do
    lang="${slice% *}"; part="${slice#* }"
    log "stage 2: slice $lang p$part (concurrency=$CONCURRENCY batch=$BATCH)"
    $PY -m data_processing.exhaustive slice --root "$RUN_ROOT" --lang "$lang" \
        --part "$part" --run-id "$RUN_ID" --url "$URL" \
        --concurrency "$CONCURRENCY" --batch-size "$BATCH" --max-tokens "$MAX_TOKENS" \
        2>&1 | tee "$LOGS/exhaustive_${lang}_p${part}_rank${NODE_RANK}.log"
done

log "rank $NODE_RANK done; run the audit to reconcile coverage:"
log "  $PY -m data_processing.exhaustive audit --root $RUN_ROOT"
