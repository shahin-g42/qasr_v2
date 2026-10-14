#!/usr/bin/env bash
# JEV voice-coverage v1 campaign driver: 1188 oracle rows spanning nine rule
# families (comparison, temporal_window, policy_logic, budget_check,
# intent_routing, safety_gate, triage_rubric, readback_match, slot_assertion)
# x six rendering styles (MSA, Egyptian, Levantine, Gulf, code-switch, English)
# x twelve scenarios each, authored under the oracle-templates-v2 approval and
# planned with full reservations. Same stage/tool chain as the v3 scale-up.
#
# Steps (run in order; each step refuses to run before its input exists):
#   author          1188 worlds -> 4 oracle stages + reservations-voice-v1.json
#   plan            seeded plan with reservations -> plan-voice-v1 + train/panels subsets
#   verify          4 verify workers over the 648 train rows (evidence-first prompt)
#   panels-export   annotate export -> jobs + 3 blind answer sheets (prints fill commands)
#   panels-needed   (after the 3 sheets are filled) -> adjudication-needed list + command
#   panels-import   (after adjudications are filled) -> panels-reviewed-voice-v1-<ts>
#   accepted        merges previous accepted + verified train + panels -> accepted-voice-v1-<ts>
#
# Env: PANEL_A PANEL_B PANEL_C ADJUDICATOR must be reviewer ids that exist in
# runs/reviews/reviewers.json (see tools/assemble_reviews.py for the roster
# format). PYTHON/JEVDATA override the interpreters if needed. VOICE_FLOOR=n
# scales the per-split family floors in the accepted step (percent, default 85).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REG=configs/data/sources.yaml
PYTHON="${PYTHON:-python3}"
JEVDATA="${JEVDATA:-jev-data}"
export PYTHONPATH="$PWD/packages/contracts/src:$PWD/packages/data/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE=oracle-templates-v2
RUBRIC="panel-voice-v1-2026-10-05"
SEED="voice-v1-2026-10"
RESERVATIONS="runs/reports/reservations-voice-v1.json"
TS="$(date +%Y%m%d-%H%M%S)"
STEP="${1:-}"
die() { echo "voiceup: $*" >&2; exit 2; }
require_reviewers() {
  for name in PANEL_A PANEL_B PANEL_C ADJUDICATOR; do
    [[ -n "${!name:-}" ]] || die "set $name (see: cat runs/reviews/reviewers.json)"
  done
}

step_author() {
  $PYTHON tools/worlds_voice_v1.py
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-voice-v1-train.jsonl \
    --source "$SOURCE" --registry "$REG" --output runs/stages/oracle-voice-v1-train \
    --prefix vv1tr-
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-voice-v1-dev.jsonl \
    --source "$SOURCE" --registry "$REG" --output runs/stages/oracle-voice-v1-dev \
    --prefix vv1dv- --split development
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-voice-v1-cal.jsonl \
    --source "$SOURCE" --registry "$REG" --output runs/stages/oracle-voice-v1-cal \
    --prefix vv1cl- --split calibration --calibration-subset temperature_fit
  $PYTHON tools/oracle_batch.py --worlds runs/worlds/worlds-voice-v1-test.jsonl \
    --source "$SOURCE" --registry "$REG" --output runs/stages/oracle-voice-v1-test \
    --prefix vv1ts- --split test --allow-test --purpose evaluation
  $PYTHON - <<'PY'
import sys
from jev_data.io import read_rows
expected = {("train", 648), ("development", 216), ("calibration", 108), ("test", 216)}
seen = set()
for name, stage in (("train", "runs/stages/oracle-voice-v1-train"),
                    ("dev", "runs/stages/oracle-voice-v1-dev"),
                    ("cal", "runs/stages/oracle-voice-v1-cal"),
                    ("test", "runs/stages/oracle-voice-v1-test")):
    rows = read_rows(stage)
    splits = {row["split"] for row in rows}
    seen.add((next(iter(splits)), len(rows)))
    families = {row["family_id"] for row in rows}
    if len(families) != len(rows):
        raise SystemExit(f"ABORT: {stage} families must be one row each")
    print(f"{stage}: {len(rows)} rows, split {splits}")
if seen != expected:
    raise SystemExit(f"ABORT: authored counts {sorted(seen)} != expected {sorted(expected)}")
print("authored 1188 oracle rows (train 648 / development 216 / calibration 108 / test 216)")
PY
  echo "== next: bash tools/voiceup_v1.sh plan"
}

step_plan() {
  [[ -d runs/stages/oracle-voice-v1-test ]] || die "run 'voiceup_v1.sh author' first"
  $JEVDATA plan \
    --input runs/stages/oracle-voice-v1-train runs/stages/oracle-voice-v1-dev \
            runs/stages/oracle-voice-v1-cal runs/stages/oracle-voice-v1-test \
    --config configs/data/tasks.yaml --seed "$SEED" \
    --reservations "$RESERVATIONS" \
    --allow-test --purpose evaluation --output runs/stages/plan-voice-v1
  $PYTHON - <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-voice-v1")
counts = Counter(row["split"] for row in rows)
expected = {"train": 648, "development": 216, "calibration": 108, "test": 216}
print("split counts:", {key: counts[key] for key in sorted(counts)})
if dict(counts) != expected:
    raise SystemExit(f"ABORT: expected {expected}; reservations were not honored")
families = {row["family_id"] for row in rows}
if len(families) != len(rows):
    raise SystemExit("ABORT: new families must be one row each")
print(f"{len(rows)} rows in {len(families)} new families, reservations honored")
PY
  $PYTHON tools/collect_rows.py --input runs/stages/plan-voice-v1 --split train \
    --output runs/stages/plan-voice-v1-train
  $PYTHON tools/collect_rows.py --input runs/stages/plan-voice-v1 \
    --split development calibration test --output runs/stages/plan-voice-v1-panels
  echo "== plan done. verify: 648 train rows | panels: 540 rows"
  echo "== next: bash tools/voiceup_v1.sh verify"
}

step_verify() {
  [[ -d runs/stages/plan-voice-v1-train ]] || die "run 'voiceup_v1.sh plan' first"
  mkdir -p runs/work/voice-v1-verify
  $PYTHON - <<'PY'
import json
from jev_data.io import read_rows
rows = read_rows("runs/stages/plan-voice-v1-train")
for worker in range(4):
    chunk = rows[worker::4]
    with open(f"runs/work/voice-v1-verify/input-{worker}.jsonl", "w", encoding="utf-8") as handle:
        for row in chunk:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"runs/work/voice-v1-verify/input-{worker}.jsonl: {len(chunk)} rows")
PY
  echo "== launching 4 verify workers (single_family, t=0.0, evidence-first) in parallel"
  for worker in 0 1 2 3; do
    $JEVDATA verify --input "runs/work/voice-v1-verify/input-$worker.jsonl" --registry "$REG" \
      --teacher-config configs/data/teachers.yaml --teacher verifier_a \
      --verification-mode single_family --cache runs/cache --allow-network \
      --shard "runs/work/voice-v1-verify/shard-$worker" --owner "vv1-w$worker" --local-storage \
      --output "runs/stages/verified-voice-v1-w$worker" \
      > "runs/work/voice-v1-verify/w$worker.log" 2>&1 &
  done
  wait || true
  $PYTHON - <<'PY'
import sys
from jev_data.io import read_rows
silver = total = 0
for worker in range(4):
    try:
        rows = read_rows(f"runs/stages/verified-voice-v1-w{worker}")
    except FileNotFoundError:
        continue
    tiers = [row["quality"]["tier"] for row in rows]
    silver += tiers.count("silver")
    total += len(rows)
    print(f"w{worker}: {len(rows)} rows ({ {tier: tiers.count(tier) for tier in sorted(set(tiers))} })")
if total == 0:
    raise SystemExit("ABORT: no verified-voice-v1-w* output found")
if silver != total:
    print(f"NOTE: {total - silver} rows quarantined; inspect "
          "runs/work/voice-v1-verify/w*.log, retry with jev-data retry, then "
          "re-verify into fresh output dirs", file=sys.stderr)
    raise SystemExit(2)
print(f"verify done: {silver}/{total} silver")
PY
  echo "== next: bash tools/voiceup_v1.sh panels-export"
}

step_panels_export() {
  require_reviewers
  [[ -d runs/stages/plan-voice-v1-panels ]] || die "run 'voiceup_v1.sh plan' first"
  JOBS="runs/jobs/panels-voice-v1-$TS"
  $JEVDATA annotate export --input runs/stages/plan-voice-v1-panels --registry "$REG" \
    --rubric-version "$RUBRIC" --allow-test --purpose evaluation --output "$JOBS"
  $PYTHON tools/review_sheets.py panels --jobs "$JOBS/jobs.json" \
    --reviewer "$PANEL_A" --reviewer "$PANEL_B" --reviewer "$PANEL_C" \
    --out runs/reviews/sheets-panels-voice-v1
  echo "== fill all three sheets IN PARALLEL (each writes its own file), e.g.:"
  for pair in "$PANEL_A:0.2" "$PANEL_B:0.5" "$PANEL_C:0.8"; do
    who="${pair%%:*}"; temp="${pair##*:}"
    echo "  nohup $PYTHON tools/llm_panel.py --jobs $JOBS/jobs.json \\"
    echo "    --sheet runs/reviews/sheets-panels-voice-v1/answer-$who.tsv \\"
    echo "    --base-url http://localhost:8010/v1 --model datagen \\"
    echo "    --reviewer-id $who --temperature $temp \\"
    echo "    > runs/reviews/sheets-panels-voice-v1/$who.log 2>&1 &"
  done
  echo "== when all three log 'done' with '0 left', run: bash tools/voiceup_v1.sh panels-needed"
}

step_panels_needed() {
  require_reviewers
  JOBS="$(ls -1dt runs/jobs/panels-voice-v1-* | head -1)"
  [[ -n "$JOBS" ]] || die "run 'voiceup_v1.sh panels-export' first"
  $PYTHON - "$JOBS" "$ADJUDICATOR" <<'PY'
import json, sys
from pathlib import Path
from jev_data.io import read_rows
jobs_dir, adjudicator = Path(sys.argv[1]), sys.argv[2]
public = json.loads((jobs_dir / "jobs.json").read_text())["jobs"]
private = json.loads((jobs_dir / "private.json").read_text())["private_map"]
rows = {row["id"]: row for row in read_rows("runs/stages/plan-voice-v1-panels")}
job_row = {job["job_id"]: private[job["job_id"]]["row_id"] for job in public}
columns = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
           "surface_faithful", "evidence", "elapsed_seconds", "flags"]
votes: dict[str, list[dict]] = {}
sheets = sorted(Path("runs/reviews/sheets-panels-voice-v1").glob("answer-*.tsv"))
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
Path("runs/reviews/adjudication-needed-voice-v1.txt").write_text(
    "\n".join(needed) + "\n", encoding="utf-8")
print(f"{len(needed)} jobs need adjudication (all test-split jobs + disagreements/flags)")
if short:
    print(f"WARNING: {len(short)} non-test jobs still have unfilled votes; rerun the "
          f"llm_panel commands before adjudicating: {short[:5]}"
          f"{'...' if len(short) > 5 else ''}")
print("== next, fill the adjudication sheet:")
print(f"  $PYTHON tools/llm_panel.py --jobs {jobs_dir}/jobs.json \\")
print("    --sheet runs/reviews/adjudications-voice-v1.tsv --adjudication \\")
print("    --only runs/reviews/adjudication-needed-voice-v1.txt \\")
print("    --base-url http://localhost:8010/v1 --model datagen \\")
print(f"    --reviewer-id {adjudicator} --temperature 0.0")
print("== then: bash tools/voiceup_v1.sh panels-import")
PY
}

step_panels_import() {
  JOBS="$(ls -1dt runs/jobs/panels-voice-v1-* | head -1)"
  [[ -n "$JOBS" ]] || die "run 'voiceup_v1.sh panels-export' first"
  [[ -f runs/reviews/adjudications-voice-v1.tsv ]] || \
    die "fill runs/reviews/adjudications-voice-v1.tsv first"
  $PYTHON - <<'PY'
from pathlib import Path
sheets = sorted(Path("runs/reviews/sheets-panels-voice-v1").glob("answer-*.tsv"))
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
Path("runs/reviews/votes-voice-v1.tsv").write_text(header + "\n" + "\n".join(lines) + "\n",
                                                    encoding="utf-8")
print(f"votes-voice-v1.tsv: {len(lines)} vote rows from {len(sheets)} sheets")
PY
  $PYTHON tools/assemble_reviews.py --jobs "$JOBS/jobs.json" \
    --votes runs/reviews/votes-voice-v1.tsv --roster runs/reviews/reviewers.json \
    --adjudications runs/reviews/adjudications-voice-v1.tsv \
    --output "runs/reviews/submissions-voice-v1-$TS.json"
  $JEVDATA annotate import --input runs/stages/plan-voice-v1-panels --registry "$REG" \
    --jobs "$JOBS" --reviews "runs/reviews/submissions-voice-v1-$TS.json" \
    --allow-test --purpose evaluation --output "runs/stages/panels-reviewed-voice-v1-$TS"
  $PYTHON - "runs/stages/panels-reviewed-voice-v1-$TS" <<'PY'
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
  echo "== next: bash tools/voiceup_v1.sh accepted"
}

step_accepted() {
  PANELS="$(ls -1dt runs/stages/panels-reviewed-voice-v1-* | head -1)"
  [[ -n "$PANELS" ]] || die "run 'voiceup_v1.sh panels-import' first"
  [[ -d runs/stages/verified-voice-v1-w0 ]] || die "run 'voiceup_v1.sh verify' first"
  EXCLUDE="runs/reports/excluded-voice-v1.txt"
  local exc=()
  if [[ -f "$EXCLUDE" ]]; then
    echo "== excluding $(awk 'NF && $1 !~ /^#/' "$EXCLUDE" | wc -l | tr -d ' ') planned rows per $EXCLUDE"
    exc=(--exclude-file "$EXCLUDE")
  fi
  local inputs=()
  OLD="$(ls -1dt runs/stages/accepted-* 2>/dev/null | { grep -v "accepted-voice-v1" || true; } | head -1)"
  if [[ -n "$OLD" ]]; then
    echo "== carrying forward existing accepted set: $OLD"
    inputs+=("$OLD")
  fi
  inputs+=(runs/stages/verified-voice-v1-w0 runs/stages/verified-voice-v1-w1
           runs/stages/verified-voice-v1-w2 runs/stages/verified-voice-v1-w3 "$PANELS")
  OUT="runs/stages/accepted-voice-v1-$TS"
  $PYTHON tools/collect_rows.py --input "${inputs[@]}" \
    --tier silver gold "${exc[@]}" --output "$OUT"
  $PYTHON - "$OUT" "${VOICE_FLOOR:-85}" <<'PY'
import sys
from collections import Counter
from jev_data.io import read_rows
rows = read_rows(sys.argv[1])
floor_percent = int(sys.argv[2])
capacity = {"train": 648, "development": 216, "calibration": 108, "test": 216}
tiers = Counter(row["quality"]["tier"] for row in rows)
splits = Counter(row["split"] for row in rows)
families = {key: len({row["family_id"] for row in rows if row["split"] == key})
            for key in sorted(splits)}
print(f"accepted: {len(rows)} rows | tiers {dict(tiers)} | splits {dict(splits)}")
print(f"families per split: {families}")
floors = {key: capacity[key] * floor_percent // 100 for key in capacity}
print(f"floors ({floor_percent}% of reserved capacity {capacity}): {floors}")
short = {key: value for key, value in families.items() if value < floors[key]}
missing = [key for key in capacity if key not in families]
if short or missing:
    raise SystemExit(f"ABORT: split families below floor {short}; missing splits {missing}")
print("== accepted set ready. next: bash tools/curate_finish.sh")
PY
}

case "$STEP" in
  author) step_author ;;
  plan) step_plan ;;
  verify) step_verify ;;
  panels-export) step_panels_export ;;
  panels-needed) step_panels_needed ;;
  panels-import) step_panels_import ;;
  accepted) step_accepted ;;
  *) die "unknown step '$STEP' (author|plan|verify|panels-export|panels-needed|panels-import|accepted)" ;;
esac
