#!/usr/bin/env python3
"""Human-workflow scaffolding for the JEV pilot: sheets in, validated JSON out.

Subcommands
  panels      Render blind annotation jobs into readable questions + answer sheets.
  audit       Render the preregistered audit sample into an auditor sheet.
  audit-json  Convert a filled audit sheet into the reviews JSON, fully re-verified
              against the rows and the plan digests.
  resolutions Render curate near-duplicate pairs into a resolution sheet.
  resolutions-json
              Convert a filled resolution sheet into the resolutions JSON.

Sheets are tab-separated with one header row. Reviewers fill only the empty
cells; everything hash-bearing is prefilled and re-verified on conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/data/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/contracts/src"))

from jev_data.io import digest, read_rows  # noqa: E402

VOTE_COLUMNS = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
                "surface_faithful", "evidence", "elapsed_seconds", "flags"]
AUDIT_COLUMNS = ["id", "row_hash", "plan_hash", "reviewer_id", "reviewer_version",
                 "native_variety", "independent", "critical_error", "evidence",
                 "elapsed_seconds"]
RESOLUTION_COLUMNS = ["id_a", "id_b", "decision", "reason"]
BOOLEAN = {"true": True, "false": False}


def fail(message: str) -> None:
    print(f"review_sheets: {message}", file=sys.stderr)
    raise SystemExit(2)


def write_sheet(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    print(f"wrote {path} ({len(rows)} data rows)")


def read_sheet(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing sheet: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [line.split("\t") for line in lines]
    if rows[0] != columns:
        fail(f"{path}: header must be exactly: {chr(9).join(columns)}")
    parsed = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(columns):
            fail(f"{path} line {number}: expected {len(columns)} columns, got {len(row)}")
        parsed.append(dict(zip(columns, row, strict=False)))
    return parsed


def boolean(value: str, where: str) -> bool:
    key = value.strip().lower()
    if key not in BOOLEAN:
        fail(f"{where}: expected true/false, got {value!r}")
    return BOOLEAN[key]


def render_question(index: int, job: dict) -> str:
    payload = job["input"]
    options = payload.get("question", {}).get("options", [])
    lines = [f"[{index}] job_id: {job['job_id']}", "STATE:", str(payload.get("state", "")),
             "QUESTION:", str(payload.get("question", {}).get("text", payload.get("question")))]
    for option in options:
        lines.append(f"  - {option['id']}: {option.get('description', '')}")
    lines.append("ANSWER (option id) + the three booleans + Arabic evidence + elapsed "
                 "seconds; leave flags empty unless flagging.")
    return "\n".join(lines)


def cmd_panels(args: argparse.Namespace) -> int:
    jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.txt").write_text(
        "\n\n".join(render_question(i, job) for i, job in enumerate(jobs, start=1)) + "\n",
        encoding="utf-8")
    print(f"wrote {out / 'questions.txt'} ({len(jobs)} jobs)")
    for reviewer in args.reviewer:
        write_sheet(out / f"answer-{reviewer}.tsv", VOTE_COLUMNS,
                    [[job["job_id"], reviewer, "", "", "", "", "", "", ""] for job in jobs])
    write_sheet(out / "adjudications-template.tsv",
                ["job_id", "adjudicator_id", *VOTE_COLUMNS[2:], "resolved", "reason"], [])
    print("fill the sheets, then: python3 tools/assemble_reviews.py --jobs "
          f"{args.jobs} --votes <sheet> --roster <roster> "
          "[--adjudications <filled adjudications>] --output <submissions.json>")
    return 0


def needed_ids(plan: dict) -> set[str]:
    return {key for sample in plan["samples"].values()
            for ids in sample["families"].values() for key in ids}


def cmd_audit(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.audit_plan).read_text())
    rows = {row["id"]: row for row in read_rows(args.input)}
    sample = needed_ids(plan)
    missing = sorted(set(rows) - sample)
    if missing:
        print(f"note: {len(missing)} rows are outside the audit sample (first: {missing[:3]})")
    plan_hash = digest(plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit-questions.txt").write_text(
        "\n\n".join(
            f"[{i}] id: {key}\nSTATE:\n{rows[key]['state']}\nQUESTION:\n"
            f"{rows[key]['question'].get('text', rows[key]['question'])}\nOPTIONS: "
            + "; ".join(f"{o['id']}: {o.get('description', '')}"
                        for o in rows[key]["question"]["options"])
            + f"\nSTORED TARGET: {rows[key]['target']['option_id']}"
            for i, key in enumerate(sorted(sample), start=1)) + "\n",
        encoding="utf-8")
    print(f"wrote {out / 'audit-questions.txt'} ({len(sample)} rows to audit)")
    write_sheet(out / f"audit-answer-{args.reviewer_id}.tsv", AUDIT_COLUMNS,
                [[key, digest(rows[key]), plan_hash, args.reviewer_id, args.reviewer_version,
                  rows[key]["variety"], "true", "", "", ""] for key in sorted(sample)])
    print("audit each row (critical_error true/false, Arabic evidence, elapsed seconds), "
          "then: python3 tools/review_sheets.py audit-json --sheet <sheet> "
          f"--audit-plan {args.audit_plan} --input {args.input} --out <reviews.json>")
    return 0


def cmd_audit_json(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.audit_plan).read_text())
    rows = {row["id"]: row for row in read_rows(args.input)}
    plan_hash, sample = digest(plan), needed_ids(plan)
    reviews = []
    for entry in read_sheet(Path(args.sheet), AUDIT_COLUMNS):
        key = entry["id"]
        where = f"audit sheet row id {key}"
        if key not in sample:
            fail(f"{where}: not part of the preregistered sample")
        if entry["row_hash"] != digest(rows[key]):
            fail(f"{where}: row_hash mismatch")
        if entry["plan_hash"] != plan_hash:
            fail(f"{where}: plan_hash mismatch; sheet was generated from another plan")
        if entry["reviewer_id"] != args.reviewer_id:
            fail(f"{where}: reviewer_id must match --reviewer-id ({args.reviewer_id})")
        if entry["native_variety"] != rows[key]["variety"]:
            fail(f"{where}: native_variety must stay prefilled")
        evidence = entry["evidence"].strip()
        if not evidence:
            fail(f"{where}: evidence is required for every audited row")
        try:
            elapsed = float(entry["elapsed_seconds"])
        except ValueError:
            fail(f"{where}: elapsed_seconds must be a number")
        if not 0.001 <= elapsed <= 86400:
            fail(f"{where}: elapsed_seconds outside the rubric range")
        reviews.append({"id": key, "row_hash": entry["row_hash"], "plan_hash": plan_hash,
                        "reviewer_id": entry["reviewer_id"],
                        "reviewer_version": entry["reviewer_version"],
                        "native_variety": entry["native_variety"],
                        "independent": boolean(entry["independent"], where),
                        "critical_error": boolean(entry["critical_error"], where),
                        "evidence": evidence, "elapsed_seconds": elapsed})
    seen = {review["id"] for review in reviews}
    still_missing = sorted(sample - seen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reviews, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} ({len(reviews)} reviews)")
    if still_missing:
        print(f"WARNING: {len(still_missing)} sampled rows still lack reviews: "
              f"{still_missing[:5]}{'...' if len(still_missing) > 5 else ''}")
        return 2
    print("all sampled rows reviewed — run the audit: "
          f"jev-data audit --phase run --input {args.input} --registry configs/data/sources.yaml "
          f"--audit-plan {args.audit_plan} --reviews {out} "
          "--output runs/reports/audit-run.json")
    return 0


def cmd_resolutions(args: argparse.Namespace) -> int:
    curation = json.loads(Path(args.curation).read_text())
    pairs = [error["ids"] for error in curation.get("errors", [])
             if error.get("reason") == "near_duplicate_requires_review"]
    if not pairs:
        print("no near-duplicate resolutions required")
        return 0
    out = Path(args.out)
    write_sheet(out, RESOLUTION_COLUMNS,
                [[pair[0], pair[1], "", ""] for pair in pairs])
    print("decide each pair: distinct (independent questions) or linked (same family, "
          "needs a matching union-groups entry); add an Arabic reason, then: "
          "python3 tools/review_sheets.py resolutions-json --sheet "
          f"{out} --input {args.input} --reviewer-id {args.reviewer_id} "
          "--out runs/reports/resolutions.json")
    return 0


def cmd_resolutions_json(args: argparse.Namespace) -> int:
    rows = {row["id"]: row for row in read_rows(args.input)}
    resolutions = []
    for entry in read_sheet(Path(args.sheet), RESOLUTION_COLUMNS):
        where = f"resolution pair {entry['id_a']}+{entry['id_b']}"
        if entry["id_a"] not in rows or entry["id_b"] not in rows:
            fail(f"{where}: unknown row id")
        decision = entry["decision"].strip().lower()
        if decision not in {"distinct", "linked"}:
            fail(f"{where}: decision must be distinct or linked")
        reason = entry["reason"].strip()
        if not reason:
            fail(f"{where}: reason is required")
        resolutions.append({"ids": [entry["id_a"], entry["id_b"]],
                            "row_hashes": {entry["id_a"]: digest(rows[entry["id_a"]]),
                                           entry["id_b"]: digest(rows[entry["id_b"]])},
                            "reviewer_id": args.reviewer_id, "independent": True,
                            "decision": decision, "reason": reason})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resolutions, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} ({len(resolutions)} resolutions)")
    linked = [item["ids"] for item in resolutions if item["decision"] == "linked"]
    if linked:
        print(f"note: {len(linked)} linked pairs; pass the same pairs as --union-groups "
              "to curate/audit (see tools/pilot.sh curate)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    panels = sub.add_parser("panels", help="render blind jobs into answer sheets")
    panels.add_argument("--jobs", required=True)
    panels.add_argument("--reviewer", action="append", required=True,
                        help="repeat once per reviewer")
    panels.add_argument("--out", required=True)

    audit = sub.add_parser("audit", help="render the audit sample into an auditor sheet")
    audit.add_argument("--input", required=True, help="accepted stage dir")
    audit.add_argument("--audit-plan", required=True)
    audit.add_argument("--reviewer-id", required=True)
    audit.add_argument("--reviewer-version", required=True)
    audit.add_argument("--out", required=True)

    audit_json = sub.add_parser("audit-json", help="filled audit sheet -> reviews JSON")
    audit_json.add_argument("--sheet", required=True)
    audit_json.add_argument("--audit-plan", required=True)
    audit_json.add_argument("--input", required=True)
    audit_json.add_argument("--reviewer-id", required=True)
    audit_json.add_argument("--out", required=True)

    res = sub.add_parser("resolutions", help="near-duplicate pairs -> decision sheet")
    res.add_argument("--curation", required=True, help="curate report JSON")
    res.add_argument("--input", required=True)
    res.add_argument("--reviewer-id", required=True)
    res.add_argument("--out", required=True)

    res_json = sub.add_parser("resolutions-json", help="filled sheet -> resolutions JSON")
    res_json.add_argument("--sheet", required=True)
    res_json.add_argument("--input", required=True)
    res_json.add_argument("--reviewer-id", required=True)
    res_json.add_argument("--out", required=True)

    handlers = {"panels": cmd_panels, "audit": cmd_audit, "audit-json": cmd_audit_json,
                "resolutions": cmd_resolutions, "resolutions-json": cmd_resolutions_json}
    args = parser.parse_args(argv)
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
