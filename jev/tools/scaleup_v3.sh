#!/usr/bin/env bash
# JEV v3 certification scale-up driver: 430 new 1-row families closing every audit
# cell at 120 families/split, in the same proven stage/tool chain as the pilot.
# Self-contained: every command carries its own sealed-test flags; only the final
# audit/approve/freeze steps use tools/pilot.sh.
#
# Steps (run in order; each step refuses to run before its input exists):
#   author          430 worlds -> 4 oracle stages + runs/reports/reservations-v3.json
#   plan            seeded plan with reservations -> plan-v3 + train/panels subsets
#   verify          (superseded by verify2; kept for history) 4 verify workers
#   topup           16 same-unit replacement train rows -> plan-v3-topup
#   verify2         re-verify ALL train rows under the evidence-first prompt
#                   (label-first templates let the t=0.0 judge answer before it
#                   reasoned, flipping cross-unit rows its own evidence solved
#                   correctly) -> verified-v3r-w0..w3
#   panels-export   annotate export -> jobs + 3 blind answer sheets (prints fill commands)
#   panels-needed   (after the 3 sheets are filled) -> adjudication-needed list + command
#   panels-import   (after adjudications are filled) -> panels-reviewed-v3-<ts>
#   accepted        merges old accepted + new verified + new panels -> accepted-v3-<ts>
#
# Env: PANEL_A PANEL_B PANEL_C ADJUDICATOR must be reviewer ids that exist in
# runs/reviews/reviewers.json (cat it; the pilot already has three panel ids and
# an adjudicator id). PYTHON/JEVDATA override the interpreters if needed.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REG=configs/data/sources.yaml
PYTHON="${PYTHON:-python3}"
JEVDATA="${JEVDATA:-jev-data}"
export PYTHONPATH="$PWD/packages/contracts/src:$PWD/packages/data/src${PYTHONPATH:+:$PYTHONPATH}"
RUBRIC="panel-v3-2026-10-05"
TS="$(date +%Y%m%d-%H%M%S)"
STEP="${1:-}"
die() { echo "scaleup: $*" >&2; exit 2; }
require_reviewers() {
  for name in PANEL_A PANEL_B PANEL_C ADJUDICATOR; do
    [[ -n "${!name:-}" ]] || die "set $name (see: cat runs/reviews/reviewers.json)"
  done
}

step_author() {
  $PYTHON tools/worlds_v3.py
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-v3-train.jsonl \
    --source oracle-templates --registry "$REG" --output runs/stages/oracle-v3-train --prefix v3tr-
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-v3-dev.jsonl \
    --source oracle-templates --registry "$REG" --output runs/stages/oracle-v3-dev --prefix v3dv- \
    --split development
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-v3-cal.jsonl \
    --source oracle-templates --registry "$REG" --output runs/stages/oracle-v3-cal --prefix v3cl- \
    --split calibration --calibration-subset temperature_fit
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-v3-test.jsonl \
    --source oracle-templates --registry "$REG" --output runs/stages/oracle-v3-test --prefix v3ts- \
    --split test --allow-test --purpose evaluation
  echo "== authored 430 oracle rows (train 89 / development 112 / calibration 118 / test 111)"
  echo "== next: bash tools/scaleup_v3.sh plan"
}

step_plan() {
  [[ -d runs/stages/oracle-v3-test ]] || die "run 'scaleup_v3.sh author' first"
  $JEVDATA plan \
    --input runs/stages/oracle-v3-train runs/stages/oracle-v3-dev \
            runs/stages/oracle-v3-cal runs/stages/oracle-v3-test \
    --config configs/data/tasks.yaml --seed scaleup-v3-2026-10 \
    --reservations runs/reports/reservations-v3.json \
    --allow-test --purpose evaluation --output runs/stages/plan-v3
  $PYTHON - <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-v3")
counts = Counter(row["split"] for row in rows)
expected = {"train": 89, "development": 112, "calibration": 118, "test": 111}
print("split counts:", {key: counts[key] for key in sorted(counts)})
if dict(counts) != expected:
    raise SystemExit(f"ABORT: expected {expected}; reservations were not honored")
families = {row["family_id"] for row in rows}
if len(families) != len(rows):
    raise SystemExit("ABORT: new families must be one row each")
print(f"{len(rows)} rows in {len(families)} new families, reservations honored")
PY
  $PYTHON tools/collect_rows.py --input runs/stages/plan-v3 --split train \
    --output runs/stages/plan-v3-train
  $PYTHON tools/collect_rows.py --input runs/stages/plan-v3 \
    --split development calibration test --output runs/stages/plan-v3-panels
  echo "== plan done. verify: 89 train rows | panels: 341 rows"
  echo "== next: bash tools/scaleup_v3.sh verify"
}

step_verify() {
  [[ -d runs/stages/plan-v3-train ]] || die "run 'scaleup_v3.sh plan' first"
  mkdir -p runs/work/v3-verify
  $PYTHON - <<'PY'
import json
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-v3-train")
for worker in range(4):
    chunk = rows[worker::4]
    with open(f"runs/work/v3-verify/input-{worker}.jsonl", "w", encoding="utf-8") as handle:
        for row in chunk:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"runs/work/v3-verify/input-{worker}.jsonl: {len(chunk)} rows")
PY
  echo "== launching 4 verify workers (single_family, t=0.0) in parallel"
  for worker in 0 1 2 3; do
    $JEVDATA verify --input "runs/work/v3-verify/input-$worker.jsonl" --registry "$REG" \
      --teacher-config configs/data/teachers.yaml --teacher verifier_a \
      --verification-mode single_family --cache runs/cache --allow-network \
      --shard "runs/work/v3-verify/shard-$worker" --owner "v3-w$worker" --local-storage \
      --output "runs/stages/verified-v3-w$worker" \
      > "runs/work/v3-verify/w$worker.log" 2>&1 &
  done
  wait || true
  $PYTHON - <<'PY'
import sys
from jev_data.io import read_rows
silver = total = 0
for worker in range(4):
    rows = read_rows(f"runs/stages/verified-v3-w{worker}")
    tiers = [row["quality"]["tier"] for row in rows]
    silver += tiers.count("silver")
    total += len(rows)
    print(f"w{worker}: {len(rows)} rows ({ {tier: tiers.count(tier) for tier in sorted(set(tiers))} })")
if silver != total:
    print("ABORT: rows were quarantined; inspect runs/work/v3-verify/w*.log, "
          "retry them with jev-data retry, and re-verify into fresh output dirs", file=sys.stderr)
    raise SystemExit(2)
print(f"verify done: {silver}/{total} silver")
PY
  echo "== superseded by verify2 after the evidence-first prompt fix"
}

step_topup() {
  [[ -f runs/reports/reservations-topup.json ]] || \
    $PYTHON tools/worlds_v3.py --append "${TOPUP_N:-16}"
  [[ -f runs/worlds/worlds-v3-topup.jsonl ]] || die "top-up worlds missing"
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-v3-topup.jsonl \
    --source oracle-templates --registry "$REG" --output runs/stages/oracle-v3-topup \
    --prefix v3tx-
  $JEVDATA plan --input runs/stages/oracle-v3-topup \
    --config configs/data/tasks.yaml --seed scaleup-v3-2026-10 \
    --reservations runs/reports/reservations-topup.json \
    --output runs/stages/plan-v3-topup
  $PYTHON - <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-v3-topup")
counts = Counter(row["split"] for row in rows)
if not rows or dict(counts) != {"train": len(rows)}:
    raise SystemExit(f"ABORT: top-up must be non-empty and all-train, got {dict(counts)}")
families = {row["family_id"] for row in rows}
if len(families) != len(rows):
    raise SystemExit("ABORT: top-up families must be one row each")
print(f"{len(rows)} top-up rows in {len(families)} families, all reserved to train")
PY
  echo "== next: bash tools/scaleup_v3.sh verify2"
}

step_verify2() {
  [[ -d runs/stages/plan-v3-train ]] || die "run 'scaleup_v3.sh plan' first"
  mkdir -p runs/work/v3-verify2
  $PYTHON - <<'PY'
import json
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-v3-train")
try:
    rows = rows + read_rows("runs/stages/plan-v3-topup")
except FileNotFoundError:
    pass
print(f"re-verifying {len(rows)} train rows under the evidence-first prompt")
for worker in range(4):
    chunk = rows[worker::4]
    with open(f"runs/work/v3-verify2/input-{worker}.jsonl", "w", encoding="utf-8") as handle:
        for row in chunk:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"runs/work/v3-verify2/input-{worker}.jsonl: {len(chunk)} rows")
PY
  echo "== launching 4 verify workers (single_family, t=0.0, evidence-first)"
  for worker in 0 1 2 3; do
    $JEVDATA verify --input "runs/work/v3-verify2/input-$worker.jsonl" --registry "$REG" \
      --teacher-config configs/data/teachers.yaml --teacher verifier_a \
      --verification-mode single_family --cache runs/cache --allow-network \
      --shard "runs/work/v3-verify2/shard-$worker" --owner "v3r-w$worker" --local-storage \
      --output "runs/stages/verified-v3r-w$worker" \
      > "runs/work/v3-verify2/w$worker.log" 2>&1 &
  done
  wait || true
  $PYTHON - <<'PY'
import sys
from jev_data.io import read_rows
silver = total = 0
for worker in range(4):
    try:
        rows = read_rows(f"runs/stages/verified-v3r-w{worker}")
    except FileNotFoundError:
        continue
    tiers = [row["quality"]["tier"] for row in rows]
    silver += tiers.count("silver")
    total += len(rows)
    print(f"w{worker}: {len(rows)} rows ({ {tier: tiers.count(tier) for tier in sorted(set(tiers))} })")
if total == 0:
    raise SystemExit("ABORT: no verified-v3r-w* output found")
if silver != total:
    print(f"NOTE: {total - silver} rows still quarantined; accepted applies the "
          "115-family floor", file=sys.stderr)
    raise SystemExit(2)
print(f"verify2 done: {silver}/{total} silver")
PY
  echo "== next: bash tools/scaleup_v3.sh panels-needed once the three sheets are done"
}

step_panels_export() {
  require_reviewers
  [[ -d runs/stages/plan-v3-panels ]] || die "run 'scaleup_v3.sh plan' first"
  JOBS="runs/jobs/panels-v3-$TS"
  $JEVDATA annotate export --input runs/stages/plan-v3-panels --registry "$REG" \
    --rubric-version "$RUBRIC" --allow-test --purpose evaluation --output "$JOBS"
  $PYTHON tools/review_sheets.py panels --jobs "$JOBS/jobs.json" \
    --reviewer "$PANEL_A" --reviewer "$PANEL_B" --reviewer "$PANEL_C" \
    --out runs/reviews/sheets-panels-v3
  echo "== fill all three sheets IN PARALLEL (each writes its own file), e.g.:"
  for pair in "$PANEL_A:0.2" "$PANEL_B:0.5" "$PANEL_C:0.8"; do
    who="${pair%%:*}"; temp="${pair##*:}"
    echo "  nohup $PYTHON tools/llm_panel.py --jobs $JOBS/jobs.json \\"
    echo "    --sheet runs/reviews/sheets-panels-v3/answer-$who.tsv \\"
    echo "    --base-url http://localhost:8010/v1 --model datagen \\"
    echo "    --reviewer-id $who --temperature $temp \\"
    echo "    > runs/reviews/sheets-panels-v3/$who.log 2>&1 &"
  done
  echo "== when all three log 'done' with '0 left', run: bash tools/scaleup_v3.sh panels-needed"
}

step_panels_needed() {
  require_reviewers
  JOBS="$(ls -1dt runs/jobs/panels-v3-* | head -1)"
  [[ -n "$JOBS" ]] || die "run 'scaleup_v3.sh panels-export' first"
  $PYTHON - "$JOBS" "$ADJUDICATOR" <<'PY'
import json, sys
from pathlib import Path
from jev_data.io import read_rows
jobs_dir, adjudicator = Path(sys.argv[1]), sys.argv[2]
public = json.loads((jobs_dir / "jobs.json").read_text())["jobs"]
private = json.loads((jobs_dir / "private.json").read_text())["private_map"]
rows = {row["id"]: row for row in read_rows("runs/stages/plan-v3-panels")}
job_row = {job["job_id"]: private[job["job_id"]]["row_id"] for job in public}
columns = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
           "surface_faithful", "evidence", "elapsed_seconds", "flags"]
votes: dict[str, list[dict]] = {}
sheets = sorted(Path("runs/reviews/sheets-panels-v3").glob("answer-*.tsv"))
if len(sheets) != 3:
    raise SystemExit(f"expected 3 answer sheets, found {len(sheets)}")
for sheet in sheets:
    lines = [line for line in sheet.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines[0].split("\t") != columns:
        raise SystemExit(f"{sheet}: unexpected header")
    for line in lines[1:]:
        entry = dict(zip(columns, line.split("\t")))
        votes.setdefault(entry["job_id"], []).append(entry)
needed, short = [], []
for job_id in sorted(job_row):
    row = rows[job_row[job_id]]
    filled = [vote for vote in votes.get(job_id, []) if vote["label"].strip()]
    if len(filled) < 3 and row["split"] != "test":
        short.append(job_id)
    flagged = (len({vote["label"] for vote in filled}) > 1
               or any(vote["flags"].strip() or vote["unambiguous"].strip().lower() != "true"
                      or vote["surface_faithful"].strip().lower() != "true" for vote in filled))
    if row["split"] == "test" or flagged:
        needed.append(job_id)
Path("runs/reviews/adjudication-needed-v3.txt").write_text("\n".join(needed) + "\n",
                                                            encoding="utf-8")
print(f"{len(needed)} jobs need adjudication (all test-split jobs + disagreements/flags)")
if short:
    print(f"WARNING: {len(short)} non-test jobs still have unfilled votes; rerun the "
          f"llm_panel commands before adjudicating: {short[:5]}"
          f"{'...' if len(short) > 5 else ''}")
print("== next, fill the adjudication sheet:")
print(f"  $PYTHON tools/llm_panel.py --jobs {jobs_dir}/jobs.json \\")
print("    --sheet runs/reviews/adjudications-v3.tsv --adjudication \\")
print("    --only runs/reviews/adjudication-needed-v3.txt \\")
print("    --base-url http://localhost:8010/v1 --model datagen \\")
print(f"    --reviewer-id {adjudicator} --temperature 0.0")
print("== then: bash tools/scaleup_v3.sh panels-import")
PY
}

step_panels_import() {
  JOBS="$(ls -1dt runs/jobs/panels-v3-* | head -1)"
  [[ -n "$JOBS" ]] || die "run 'scaleup_v3.sh panels-export' first"
  [[ -f runs/reviews/adjudications-v3.tsv ]] || die "fill runs/reviews/adjudications-v3.tsv first"
  $PYTHON - <<'PY'
from pathlib import Path
sheets = sorted(Path("runs/reviews/sheets-panels-v3").glob("answer-*.tsv"))
if len(sheets) != 3:
    raise SystemExit(f"expected 3 answer sheets, found {len(sheets)}")
header, lines = None, []
for sheet in sheets:
    content = [line for line in sheet.read_text(encoding="utf-8").splitlines() if line.strip()]
    if header is None:
        header = content[0]
    elif content[0] != header:
        raise SystemExit(f"{sheet}: header mismatch")
    lines.extend(content[1:])
Path("runs/reviews/votes-v3.tsv").write_text(header + "\n" + "\n".join(lines) + "\n",
                                             encoding="utf-8")
print(f"votes-v3.tsv: {len(lines)} vote rows from {len(sheets)} sheets")
PY
  $PYTHON tools/assemble_reviews.py --jobs "$JOBS/jobs.json" \
    --votes runs/reviews/votes-v3.tsv --roster runs/reviews/reviewers.json \
    --adjudications runs/reviews/adjudications-v3.tsv \
    --output "runs/reviews/submissions-v3-$TS.json"
  $JEVDATA annotate import --input runs/stages/plan-v3-panels --registry "$REG" \
    --jobs "$JOBS" --reviews "runs/reviews/submissions-v3-$TS.json" \
    --allow-test --purpose evaluation --output "runs/stages/panels-reviewed-v3-$TS"
  $PYTHON - "runs/stages/panels-reviewed-v3-$TS" <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
tiers, splits, reasons = Counter(), Counter(), Counter()
for row in read_rows(sys.argv[1]):
    tiers[row["quality"]["tier"]] += 1
    splits[row["split"]] += 1
    if row["quality"]["tier"] == "bronze":
        reasons.update(row["quality"]["metadata"]["disposition"]["reasons"])
print("panel tiers:", dict(tiers), "| splits:", dict(splits), "| reasons:", dict(reasons))
PY
  echo "== next: bash tools/scaleup_v3.sh accepted"
}

step_accepted() {
  PANELS="$(ls -1dt runs/stages/panels-reviewed-v3-* | head -1)"
  [[ -n "$PANELS" ]] || die "run 'scaleup_v3.sh panels-import' first"
  OLD="$(ls -1dt runs/stages/accepted-* 2>/dev/null | { grep -v "accepted-v3" || true; } | head -1)"
  [[ -n "$OLD" ]] || die "original pilot accepted stage not found"
  [[ -d runs/stages/verified-v3r-w0 ]] || die "run 'scaleup_v3.sh verify2' first (evidence-first round)"
  EXCLUDE="runs/reports/excluded-v3.txt"
  local exc=()
  if [[ -f "$EXCLUDE" ]]; then
    echo "== excluding $(awk 'NF && $1 !~ /^#/' "$EXCLUDE" | wc -l | tr -d ' ') planned rows per $EXCLUDE"
    exc=(--exclude-file "$EXCLUDE")
  fi
  OUT="runs/stages/accepted-v3-$TS"
  $PYTHON tools/collect_rows.py --input "$OLD" \
    runs/stages/verified-v3r-w0 runs/stages/verified-v3r-w1 \
    runs/stages/verified-v3r-w2 runs/stages/verified-v3r-w3 "$PANELS" \
    --tier silver gold "${exc[@]}" --output "$OUT"
  $PYTHON - "$OUT" <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
rows = read_rows(sys.argv[1])
tiers = Counter(row["quality"]["tier"] for row in rows)
splits = Counter(row["split"] for row in rows)
families = {key: len({row["family_id"] for row in rows if row["split"] == key})
            for key in sorted(splits)}
print(f"accepted: {len(rows)} rows | tiers {dict(tiers)} | splits {dict(splits)}")
print(f"families per split: {families} (floor 115 each)")
short = {key: value for key, value in families.items() if value < 115}
if short:
    raise SystemExit(f"ABORT: split families below the 115 floor: {short}")
print("== accepted set ready. next: bash tools/curate_finish.sh")
PY
}

case "$STEP" in
  author) step_author ;;
  plan) step_plan ;;
  verify) step_verify ;;
  topup) step_topup ;;
  verify2) step_verify2 ;;
  panels-export) step_panels_export ;;
  panels-needed) step_panels_needed ;;
  panels-import) step_panels_import ;;
  accepted) step_accepted ;;
  *) die "unknown step '$STEP' (author|plan|verify|topup|verify2|panels-export|panels-needed|panels-import|accepted)" ;;
esac
