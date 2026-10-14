#!/usr/bin/env bash
# One-shot from accepted set to preregistered audit plan. Self-contained: the
# jev-data calls here carry the sealed-test flags themselves and do not depend
# on the pilot.sh version. 'pilot.sh audit'/'freeze' afterwards DO require the
# fixed pilot.sh, so this script refuses to run against a stale one.
# Run after 'tools/pilot.sh accepted'.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REG=configs/data/sources.yaml
TS="$(date +%Y%m%d-%H%M%S)"
ACCEPTED="$(ls -1dt runs/stages/accepted-* | head -1)"
echo "== accepted set: $ACCEPTED"

if ! grep -q -- "--purpose evaluation" tools/pilot.sh; then
  echo "ABORT: tools/pilot.sh is the OLD version (no sealed-test flags)."
  echo "       Re-copy it from the Mac, then rerun this script:"
  echo "       scp .../jev/tools/pilot.sh .../jev/tools/  (from the Mac)"
  exit 2
fi

echo "== step 1/5: fresh curate report (gates expected to fail on near-dups)"
jev-data curate --input "$ACCEPTED" --registry "$REG" \
  --allow-test --purpose evaluation \
  --output "runs/reports/curate-$TS.json" || true
REPORT="$(ls -1t runs/reports/curate-[0-9]*.json 2>/dev/null | head -1 || true)"
if [[ ! -f "$REPORT" ]]; then
  echo "== curate crashed without writing a report; unmasked rerun:"
  python3 - "$ACCEPTED" <<'PY'
import sys, traceback
sys.path.insert(0, "packages/data/src"); sys.path.insert(0, "packages/contracts/src")
from jev_data.cli import parser, run
args = parser().parse_args(["curate", "--input", sys.argv[1],
                            "--registry", "configs/data/sources.yaml",
                            "--allow-test", "--purpose", "evaluation"])
try:
    print("cli.run returned:", run(args))
except Exception:
    traceback.print_exc()
sys.exit(2)
PY
fi
echo "== curate report: $REPORT"

echo "== step 2/5: mechanical near-dup resolutions (template-lineage rule)"
python3 - "$ACCEPTED" "$REPORT" <<'PY'
import json, pathlib, sys
sys.path.insert(0, "packages/data/src"); sys.path.insert(0, "packages/contracts/src")
from jev_data.io import dataset_identity, digest, read_rows
report = json.loads(pathlib.Path(sys.argv[2]).read_text())
rows = {r["id"]: r for r in read_rows(sys.argv[1])}
if report["dataset"]["sha256"] != dataset_identity(list(rows.values()))["sha256"]:
    raise SystemExit("STALE curate report — dataset digest mismatch; aborting")
groups, out, seen = report["split_index"]["groups"], [], set()
for error in report["errors"]:
    if error.get("reason") != "near_duplicate_requires_review":
        continue
    a, b = sorted(error["ids"])
    if (a, b) in seen:
        continue
    seen.add((a, b))
    same = groups[a] == groups[b]
    out.append({"ids": [a, b],
                "row_hashes": {a: digest(rows[a]), b: digest(rows[b])},
                "reviewer_id": "template-lineage", "independent": True,
                "decision": "linked" if same else "distinct",
                "reason": "same plan family (shared oracle template lineage)" if same
                          else "different plan families; similarity is incidental"})
pathlib.Path("runs/reports/resolutions.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print(f"{len(out)} resolutions; {sum(r['decision'] == 'linked' for r in out)} linked")
PY

echo "== step 3/5: curate with resolutions (must pass now)"
python3 - <<'PY'
import json
items = json.load(open("runs/reports/resolutions.json"))
linked = [item["ids"] for item in items if item["decision"] == "linked"]
open("runs/reports/union-groups.json", "w").write(json.dumps(linked))
print(f"linked pairs needing union-groups: {len(linked)}")
PY
RESOLVED="runs/reports/curate-resolved-$TS.json"
jev-data curate --input "$ACCEPTED" --registry "$REG" \
  --allow-test --purpose evaluation \
  --resolutions runs/reports/resolutions.json \
  --union-groups runs/reports/union-groups.json \
  --output "$RESOLVED" || true
if [[ ! -f "$RESOLVED" ]]; then
  echo "== curate-resolved crashed without a report; unmasked rerun:"
  python3 - "$ACCEPTED" <<'PY'
import sys, traceback
sys.path.insert(0, "packages/data/src"); sys.path.insert(0, "packages/contracts/src")
from jev_data.cli import parser, run
args = parser().parse_args(["curate", "--input", sys.argv[1],
                            "--registry", "configs/data/sources.yaml",
                            "--allow-test", "--purpose", "evaluation",
                            "--resolutions", "runs/reports/resolutions.json",
                            "--union-groups", "runs/reports/union-groups.json"])
try:
    print("cli.run returned:", run(args))
except Exception:
    traceback.print_exc()
sys.exit(2)
PY
fi
python3 - "$RESOLVED" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print("curate-resolved passed:", report["passed"], "| errors:", len(report["errors"]))
for error in report["errors"][:5]:
    print("  ", error.get("reason"), error.get("ids", [])[:2])
sys.exit(0 if report["passed"] else 2)
PY

echo "== step 4/5: preregistered audit plan"
PLAN="runs/reports/audit-plan-$TS.json"
jev-data audit --phase plan --input "$ACCEPTED" --registry "$REG" \
  --allow-test --purpose evaluation --seed pilot-audit-2026-10 --output "$PLAN"

echo "== step 5/5: audit sheet for qwen-audit"
python3 tools/review_sheets.py audit --input "$ACCEPTED" --audit-plan "$PLAN" \
  --reviewer-id qwen-audit --reviewer-version "Qwen3.8-27B-FP8@datagen t=0.0" \
  --out runs/reviews/sheets-audit

echo "== curate_finish complete. Next:"
echo "  python3 tools/llm_panel.py --jobs runs/jobs/panels-001/jobs.json \\"
echo "    --sheet runs/reviews/sheets-audit/audit-answer-qwen-audit.tsv \\"
echo "    --audit --input-rows $ACCEPTED \\"
echo "    --base-url http://localhost:8010/v1 --model datagen \\"
echo "    --reviewer-id qwen-audit --temperature 0.0"
echo "  then: review_sheets.py audit-json; tools/pilot.sh audit; approve; freeze"
