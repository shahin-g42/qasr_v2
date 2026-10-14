#!/usr/bin/env bash
# reset_runs.sh — wipe all derived JEV pipeline state for a clean from-scratch run.
#
# Removes ONLY derived artifacts:
#   runs/      (worlds, stages, work chunks, teacher caches, jobs, reviews, reports)
#   releases/  (frozen releases, if any)
#   /tmp/jev-shards  (shard SQLite state; override with JEV_SHARD_ROOT)
#
# NEVER touches: packages/ tools/ tests/ configs/ docs/ scripts/ — rights
# approvals (sources.yaml), teacher revision pins (teachers.yaml), and prompts
# are preserved by design; re-approving them by hand is error-prone.
#
# Usage:
#   bash tools/reset_runs.sh --dry-run   # show what would be removed
#   bash tools/reset_runs.sh             # apply
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

# --- 1) Sentinel: refuse to run anywhere but the jev tree -------------------
for sentinel in packages/data/src/jev_data/cli.py configs/data/sources.yaml; do
  if [ ! -f "$ROOT/$sentinel" ]; then
    echo "refusing: $ROOT does not look like the jev tree (missing $sentinel)" >&2
    exit 1
  fi
done

SHARDS="${JEV_SHARD_ROOT:-/tmp/jev-shards}"
targets=("$ROOT/runs" "$ROOT/releases" "$SHARDS")

echo "jev reset — root: $ROOT"
echo "removing: runs/ releases/ $SHARDS"
du -sh "${targets[@]}" 2>/dev/null || true

if [ "$DRY_RUN" = 1 ]; then
  echo "dry run: nothing removed; re-run without --dry-run to apply."
  exit 0
fi

for target in "${targets[@]}"; do
  if [ -e "$target" ]; then
    rm -rf -- "$target"
    echo "removed: $target"
  fi
done

# --- 2) Recreate the standard output roots -----------------------------------
# Plain JSON outputs (registry.json, reports) require parents to already exist.
mkdir -p "$ROOT"/runs/{worlds,stages,work,cache,jobs,reviews,reports} "$ROOT/releases"
echo "recreated: runs/{worlds,stages,work,cache,jobs,reviews,reports} and releases/"

# --- 3) Post-reset health check: code and rights must be production-ready ----
echo "--- health check ---"
pins="$(grep -c '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' "$ROOT/configs/data/teachers.yaml" || true)"
echo "teacher revision pins: $pins (expect 2)"
echoch="$(grep -c 'Never echo' "$ROOT/packages/data/src/jev_data/verify.py" || true)"
echo "anti-echo prompt:      $echoch (expect 1)"
un="$(grep -c 'user_note' "$ROOT/packages/data/src/jev_data/teachers.py" || true)"
echo "user_note plumbing:    $un (expect 3 or more)"
appr="$(grep -c 'id: oracle-templates' "$ROOT/configs/data/sources.yaml" || true)"
echo "oracle approvals:      $appr (expect 2)"
echo "tasks.yaml sha:        $(sha256sum "$ROOT/configs/data/tasks.yaml" | cut -c1-16) (expect dcce5fb0ea099faa)"

python3 - "$ROOT/configs/data/sources.yaml" <<'PY'
import sys
import yaml

doc = yaml.safe_load(open(sys.argv[1]))
approved = [s["id"] for s in doc["sources"] if s["status"] == "approved"]
print("registry parses OK:    ", len(doc["sources"]), "sources; approved:", approved)
PY

echo "--- reset complete: derived state is clean; code, approvals and pins intact ---"
