#!/usr/bin/env bash
# JEV pilot driver: everything from filled panel votes to a frozen, release-audited bundle.
# Human gates (panel votes, near-dup resolutions, audit reviews) pause the run with
# exact instructions; re-run the same command after filling the sheet.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REG=configs/data/sources.yaml
TS="$(date +%Y%m%d-%H%M%S)"
STEP="${1:-all}"

die() { echo "pilot: $*" >&2; exit 2; }
latest() { ls -1dt "$@" 2>/dev/null | head -1 || true; }
data_rows() { [[ -f "$1" ]] && (( "$(wc -l < "$1" | tr -d ' ')" > 1 )); }

step_panels() {
  local jobs votes roster adj=() out
  jobs="$(latest runs/jobs/panels-*)";  [[ -n "$jobs" ]] || die "no runs/jobs/panels-* export"
  votes="$(latest runs/reviews/votes*.tsv)"; data_rows "$votes" || die "fill runs/reviews/votes.tsv first"
  roster=runs/reviews/reviewers.json; [[ -f "$roster" ]] || die "missing $roster roster"
  if data_rows runs/reviews/adjudications.tsv; then adj=(--adjudications runs/reviews/adjudications.tsv)
  else echo "no adjudications yet (fine for pass 1)"
  fi
  python3 tools/assemble_reviews.py --jobs "$jobs/jobs.json" --votes "$votes" \
    --roster "$roster" "${adj[@]}" --output "runs/reviews/submissions-$TS.json"
  out="runs/stages/panels-reviewed-$TS"
  jev-data annotate import --input runs/stages/plan-panels --registry "$REG" \
    --jobs "$jobs" --reviews "runs/reviews/submissions-$TS.json" \
    --allow-test --purpose evaluation --output "$out" \
    || die "import failed; see assembler warnings above"
  echo "== panels reviewed -> $out"
  python3 - "$out" <<'PY'
import json, sys, pathlib
from collections import Counter
tiers, reasons = Counter(), Counter()
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tiers[row["quality"]["tier"]] += 1
            if row["quality"]["tier"] == "bronze":
                reasons.update(row["quality"]["metadata"]["disposition"]["reasons"])
print("panel tiers:", dict(tiers), "| reasons:", dict(reasons))
PY
}

step_accepted() {
  local panels out
  panels="$(latest runs/stages/panels-reviewed-*)"
  [[ -n "$panels" ]] || die "run 'pilot.sh panels' first"
  out="runs/stages/accepted-$TS"
  python3 tools/collect_rows.py \
    --input runs/stages/verified-w0 runs/stages/verified-w1 runs/stages/verified-w2 \
            runs/stages/verified-w3 "$panels" \
    --tier silver gold --output "$out"
  echo "== accepted (silver train + gold panels) -> $out"
}

step_curate() {
  local accepted out status=0
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  out="runs/reports/curate-$TS.json"
  jev-data curate --input "$accepted" --registry "$REG" \
    --allow-test --purpose evaluation --output "$out" || status=$?
  echo "== curate report -> $out (exit $status: 0 passed, 2 gate failed)"
  python3 - "$out" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print("passed:", report["passed"], "| errors:", len(report["errors"]),
      "| families:", report["unique_families"], "| rows:", report["unique_decisions"])
for error in report["errors"][:10]:
    print("  ", error.get("reason"), error.get("ids", [])[:2])
PY
  if python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if any(e['reason']=='near_duplicate_requires_review' for e in r['errors']) else 1)" "$out"; then
    python3 tools/review_sheets.py resolutions --curation "$out" --input "$accepted" \
      --reviewer-id "${RESOLUTIONS_REVIEWER:-resolution-reviewer-1}" \
      --out runs/reviews/resolutions-pending.tsv || true
    echo ">> HUMAN GATE: fill runs/reviews/resolutions-pending.tsv (distinct|linked + reason),"
    echo ">> then: python3 tools/review_sheets.py resolutions-json --sheet runs/reviews/resolutions-pending.tsv --input $accepted --reviewer-id ${RESOLUTIONS_REVIEWER:-resolution-reviewer-1} --out runs/reports/resolutions.json"
    echo ">> then re-run: tools/pilot.sh curate-resolved"
  fi
  return "$status"
}

step_curate_resolved() {
  local accepted out status=0
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  [[ -f runs/reports/resolutions.json ]] || die "missing runs/reports/resolutions.json"
  python3 - runs/reports/resolutions.json <<'PY'
import json, sys
items = json.load(open(sys.argv[1]))
linked = [item["ids"] for item in items if item["decision"] == "linked"]
open("runs/reports/union-groups.json", "w").write(json.dumps(linked))
print(f"linked pairs needing union-groups: {len(linked)}")
PY
  out="runs/reports/curate-resolved-$TS.json"
  jev-data curate --input "$accepted" --registry "$REG" \
    --allow-test --purpose evaluation \
    --resolutions runs/reports/resolutions.json \
    --union-groups runs/reports/union-groups.json --output "$out" || status=$?
  echo "== curate (with resolutions) -> $out (exit $status: 0 passed, 2 gate failed)"
  [[ $status -eq 0 ]] || echo "inspect $out for the remaining errors"
  return "$status"
}

step_audit_plan() {
  local accepted out
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  out="runs/reports/audit-plan-$TS.json"
  jev-data audit --phase plan --input "$accepted" --registry "$REG" \
    --allow-test --purpose evaluation --seed pilot-audit-2026-10 --output "$out"
  echo "== preregistered audit plan -> $out"
  python3 tools/review_sheets.py audit --input "$accepted" --audit-plan "$out" \
    --reviewer-id "${AUDIT_REVIEWER:-quality-auditor-1}" --reviewer-version 2026-10-02 \
    --out runs/reviews/sheets-audit
  echo ">> HUMAN GATE: audit the rows in runs/reviews/sheets-audit/audit-questions.txt,"
  echo ">> fill runs/reviews/sheets-audit/audit-answer-*.tsv, then:"
  echo ">> python3 tools/review_sheets.py audit-json --sheet <sheet> --audit-plan $out --input $accepted --reviewer-id ${AUDIT_REVIEWER:-quality-auditor-1} --out runs/reports/audit-reviews.json"
  echo ">> then: tools/pilot.sh audit"
}

step_audit() {
  local accepted out status=0
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  [[ -f runs/reports/audit-reviews.json ]] || die "missing runs/reports/audit-reviews.json (see audit-plan instructions)"
  out="runs/reports/audit-run-$TS.json"
  local res=()
  if [[ -f runs/reports/resolutions.json ]]; then res=(--resolutions runs/reports/resolutions.json); fi
  jev-data audit --phase run --input "$accepted" --registry "$REG" \
    --allow-test --purpose evaluation \
    --audit-plan "$(latest runs/reports/audit-plan-*)" --reviews runs/reports/audit-reviews.json \
    "${res[@]}" --output "$out" || status=$?
  echo "== audit -> $out (exit $status: 0 passed, 2 failed)"
  python3 - "$out" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print("passed:", report["passed"], "| missing reviews:", len(report["missing_review_ids"]))
for cell, result in sorted(report["results"].items()):
    if cell == "overall":
        print(f"  overall: wilson95={result['wilson_upper_95']:.4f} <= {result['threshold']} "
              f"({result['critical_error_families']}/{result['reviewed_families']} families) "
              f"passed={result['passed']}")
PY
  return "$status"
}

step_approve() {
  local accepted audit_json out
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  audit_json="$(latest runs/reports/audit-run-*)"; [[ -n "$audit_json" ]] || die "run 'pilot.sh audit' first"
  out="runs/reports/approval.json"
  python3 - "$accepted" "$audit_json" "$out" "${APPROVER:-shahin.konadath}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "packages/data/src"); sys.path.insert(0, "packages/contracts/src")
from jev_data.io import dataset_identity, digest, read_rows
from jev_data.registry import Registry

accepted_dir, audit_path, out, approver = sys.argv[1:5]
accepted = read_rows(accepted_dir)
audit = json.loads(Path(audit_path).read_text())
planned = {row["id"] for path in sorted(Path("runs/stages").glob("plan*/*.jsonl"))
           for row in ([json.loads(l) for l in path.read_text().splitlines() if l.strip()])}
accepted_ids = {row["id"] for row in accepted}
excluded_reasons = {}
excluded_file = Path("runs/reports/excluded-v3.txt")
if excluded_file.exists():
    for line in excluded_file.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)
        key = entry[0].strip()
        if key:
            excluded_reasons[key] = entry[1].strip() if len(entry) > 1 and entry[1].strip() \
                else "operator_exclusion_recorded"
quarantined, excluded = [], []
for key in sorted(planned - accepted_ids):
    if key in excluded_reasons:
        excluded.append({"id": key, "reason": excluded_reasons[key]})
        continue
    evidence = None
    for root in ("runs/stages",):
        for path in sorted(Path(root).glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            for line in path.read_text().splitlines():
                if line.strip() and json.loads(line).get("id") == key:
                    evidence = json.loads(line); break
            if evidence: break
        if evidence: break
    reasons = ";".join(evidence["quality"]["metadata"]["disposition"]["reasons"]) if evidence else "not_processed"
    item = {"id": key, "reason": reasons}
    (quarantined if evidence else excluded).append(item)
approval = {"schema_version": 1, "approved": True, "approver": approver,
            "approval_ref": "approval-pilot-2026-10-02-SK",
            "dataset_hash": dataset_identity(accepted)["sha256"],
            "audit_hash": digest(audit),
            "registry_hash": digest(Registry.load("configs/data/sources.yaml").to_dict()),
            "coverage": {"planned_ids": sorted(planned), "accepted_ids": sorted(accepted_ids),
                         "quarantined": quarantined, "excluded": excluded,
                         "pending_ids": [], "accepted_subset_approved": True}}
Path(out).write_text(json.dumps(approval, ensure_ascii=False, indent=1) + "\n")
print(f"wrote {out}: planned {len(planned)} = accepted {len(accepted_ids)} + "
      f"quarantined {len(quarantined)} + excluded {len(excluded)}")
PY
}

step_freeze() {
  local accepted audit_json out status=0
  accepted="$(latest runs/stages/accepted-*)"; [[ -n "$accepted" ]] || die "run 'pilot.sh accepted' first"
  audit_json="$(latest runs/reports/audit-run-*)"; [[ -n "$audit_json" ]] || die "audit first"
  [[ -f runs/reports/approval.json ]] || die "run 'pilot.sh approve' first"
  out="releases/pilot-$TS"
  jev-data freeze --input "$accepted" --registry "$REG" --audit "$audit_json" \
    --approval runs/reports/approval.json --allow-test --purpose evaluation --output "$out" \
    || die "freeze rejected the release; see message above"
  echo "== frozen release -> $out"
  jev-data audit --phase release --release "$out" --registry "$REG" \
    --allow-test --purpose evaluation \
    --output "runs/reports/release-check-$TS.json" || status=$?
  echo "== release check exit $status (0 = clean)"
  ls -la "$out"
}

case "$STEP" in
  panels) step_panels ;;
  accepted) step_accepted ;;
  curate) step_curate ;;
  curate-resolved) step_curate_resolved ;;
  audit-plan) step_audit_plan ;;
  audit) step_audit ;;
  approve) step_approve ;;
  freeze) step_freeze ;;
  all)
    step_panels
    step_accepted
    step_curate && step_audit_plan || true
    ;;
  *) die "unknown step '$STEP' (panels|accepted|curate|curate-resolved|audit-plan|audit|approve|freeze|all)" ;;
esac
