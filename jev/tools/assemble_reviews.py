#!/usr/bin/env python3
"""Assemble reviewer votes into jev-data annotate import submissions.

Reviewers never touch hashes or JSON. They fill a tab-separated sheet; this tool
validates it against the exported blind jobs, computes input hashes and the
adjudication votes hash with the exact digest the importer uses, and emits the
submissions file for `jev-data annotate import`.

Sheets (UTF-8, tab-separated, one header row):

votes.tsv:
  job_id  reviewer_id  label  independent  unambiguous  surface_faithful  \\
  evidence  elapsed_seconds  flags
  - booleans: true/false (case-insensitive)
  - flags: empty or |-separated tokens
  - evidence: free text; tabs and newlines are rejected

adjudications.tsv (only for jobs that need it: label disagreement, flagged or
ambiguous votes, and every test-split job):
  job_id  adjudicator_id  label  independent  unambiguous  surface_faithful  \\
  evidence  elapsed_seconds  flags  resolved  reason

reviewers.json roster:
  [{"id": "reviewer-1", "version": "2026-10-02",
    "native_varieties": ["MSA"],
    "qualifications": ["native_arabic", "quantitative_rules", "rubric_scoring",
                       "grounded_inference"]}, ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/data/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/contracts/src"))

from jev_data.io import digest  # noqa: E402

BOOLEAN = {"true": True, "false": False}
VOTE_COLUMNS = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
                "surface_faithful", "evidence", "elapsed_seconds", "flags"]
ADJ_COLUMNS = (["job_id", "adjudicator_id", *VOTE_COLUMNS[2:], "resolved", "reason"])


def fail(message: str) -> None:
    print(f"assemble_reviews: {message}", file=sys.stderr)
    raise SystemExit(2)


def boolean(value: str, where: str) -> bool:
    key = value.strip().lower()
    if key not in BOOLEAN:
        fail(f"{where}: expected true/false, got {value!r}")
    return BOOLEAN[key]


def read_sheet(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing sheet: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        fail(f"{path}: sheet is empty")
    if rows[0] != columns:
        fail(f"{path}: header must be exactly: {chr(9).join(columns)}")
    parsed = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(columns):
            fail(f"{path} line {number}: expected {len(columns)} columns, got {len(row)}")
        if any("\n" in cell for cell in row):
            fail(f"{path} line {number}: newlines are not allowed inside cells")
        parsed.append(dict(zip(columns, row, strict=False)))
    return parsed


def build_vote(entry: dict[str, str], options: set[str], rubric: str, roster: dict,
               where: str, reviewer_column: str = "reviewer_id") -> dict:
    reviewer_id = entry[reviewer_column].strip()
    if reviewer_id not in roster:
        fail(f"{where}: unknown reviewer {reviewer_id!r}; add them to the roster")
    label = entry["label"].strip()
    if label not in options:
        fail(f"{where}: label {label!r} is not one of the job's options {sorted(options)}")
    try:
        elapsed = float(entry["elapsed_seconds"])
    except ValueError:
        fail(f"{where}: elapsed_seconds must be a number, got {entry['elapsed_seconds']!r}")
    if not 0.001 <= elapsed <= 86400:
        fail(f"{where}: elapsed_seconds {elapsed} outside the rubric range 0.001..86400")
    evidence = entry["evidence"].strip()
    if not evidence:
        fail(f"{where}: evidence must be nonempty")
    flags = [flag.strip() for flag in entry["flags"].split("|") if flag.strip()]
    return {"job_id": entry["job_id"], "reviewer": roster[reviewer_id],
            "rubric_version": rubric, "input_hash": None, "label": label,
            "independent": boolean(entry["independent"], where),
            "unambiguous": boolean(entry["unambiguous"], where),
            "surface_faithful": boolean(entry["surface_faithful"], where),
            "evidence": evidence, "elapsed_seconds": elapsed, "flags": flags}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True, help="exported jobs.json (public bundle)")
    parser.add_argument("--votes", required=True, help="votes.tsv sheet")
    parser.add_argument("--roster", required=True, help="reviewers.json roster")
    parser.add_argument("--adjudications", help="optional adjudications.tsv sheet")
    parser.add_argument("--output", required=True, help="submissions JSON path")
    args = parser.parse_args(argv)

    bundle = json.loads(Path(args.jobs).read_text())
    jobs = {job["job_id"]: job for job in bundle["jobs"]}
    if len(jobs) != len(bundle["jobs"]):
        fail("jobs.json contains duplicate job ids")
    rubrics = {job.get("rubric_version") for job in bundle["jobs"]}
    if None in rubrics or len(rubrics) != 1:
        fail("jobs.json jobs do not carry a single rubric_version")
    rubric = next(iter(rubrics))
    if "rubric_version" in bundle and bundle["rubric_version"] != rubric:
        fail("jobs.json rubric_version disagrees with the jobs")
    roster_list = json.loads(Path(args.roster).read_text())
    roster = {entry["id"]: entry for entry in roster_list}
    if len(roster) != len(roster_list):
        fail("roster contains duplicate reviewer ids")

    options_by_job = {}
    for job_id, job in jobs.items():
        options = job["input"].get("question", {}).get("options", [])
        options_by_job[job_id] = {option["id"] for option in options}

    votes: dict[str, list[dict]] = {job_id: [] for job_id in jobs}
    voters: dict[str, set[str]] = {job_id: set() for job_id in jobs}
    skipped_votes, skipped_adj = [], []
    for entry in read_sheet(Path(args.votes), VOTE_COLUMNS):
        job_id = entry["job_id"].strip()
        if job_id not in jobs:
            fail(f"votes: unknown job_id {job_id!r} (not part of this export)")
        if not entry["label"].strip():
            skipped_votes.append(job_id)  # unfilled row: stale prefill, not a vote
            continue
        reviewer_id = entry["reviewer_id"].strip()
        if reviewer_id in voters[job_id]:
            fail(f"votes: duplicate vote for job {job_id} by {reviewer_id!r}")
        vote = build_vote(entry, options_by_job[job_id], rubric, roster,
                          f"votes job {job_id} reviewer {reviewer_id}")
        vote["input_hash"] = digest(jobs[job_id]["input"])
        votes[job_id].append(vote)
        voters[job_id].add(reviewer_id)

    adjudications: dict[str, dict] = {}
    if args.adjudications:
        for entry in read_sheet(Path(args.adjudications), ADJ_COLUMNS):
            job_id = entry["job_id"].strip()
            if job_id not in jobs:
                fail(f"adjudications: unknown job_id {job_id!r}")
            if not entry["label"].strip():
                skipped_adj.append(job_id)
                continue
            adjudicator = entry["adjudicator_id"].strip()
            if adjudicator in voters[job_id]:
                fail(f"adjudications: adjudicator {adjudicator!r} already voted on {job_id}")
            vote = build_vote(entry, options_by_job[job_id], rubric, roster,
                              f"adjudication job {job_id} by {adjudicator}",
                              reviewer_column="adjudicator_id")
            vote["input_hash"] = digest(jobs[job_id]["input"])
            resolved = boolean(entry["resolved"], f"adjudication job {job_id}")
            reason = entry["reason"].strip()
            if not reason:
                fail(f"adjudication job {job_id}: reason must be nonempty")
            adjudications[job_id] = {"vote": vote, "resolved": resolved,
                                     "reason": reason, "votes_hash": digest(votes[job_id])}

    if skipped_votes:
        print(f"NOTE: skipped {len(skipped_votes)} unfilled vote rows (no label yet); "
              f"those jobs keep fewer votes: {sorted(set(skipped_votes))[:5]}")
    if skipped_adj:
        print(f"NOTE: skipped {len(skipped_adj)} unfilled adjudication rows")

    submissions = []
    empty, thin = [], []
    for job_id in sorted(jobs):
        job_votes = votes[job_id]
        adjudication = adjudications.get(job_id)
        if not job_votes:
            empty.append(job_id)
        elif len(job_votes) < 2 and adjudication is None:
            thin.append(job_id)
        submissions.append({"job_id": job_id, "votes": job_votes,
                            "adjudication": adjudication})

    flagged = [job_id for job_id in sorted(jobs)
               if votes[job_id] and (len({v["label"] for v in votes[job_id]}) > 1
                                     or any(v["flags"] or not v["unambiguous"]
                                            or not v["surface_faithful"]
                                            for v in votes[job_id]))]
    missing_adjudication = [job_id for job_id in flagged if job_id not in adjudications]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(submissions, ensure_ascii=False, indent=1) + "\n")

    print(f"jobs: {len(jobs)} | votes: {sum(len(v) for v in votes.values())} | "
          f"adjudications: {len(adjudications)}")
    unanimous = sum(1 for v in votes.values()
                    if v and len({x["label"] for x in v}) == 1)
    print(f"unanimous jobs: {unanimous}")
    if empty:
        print(f"WARNING: {len(empty)} jobs have no votes at all; "
              f"they will quarantine on import: {empty[:5]}{'...' if len(empty) > 5 else ''}")
    if thin:
        print(f"WARNING: {len(thin)} jobs have a single vote and no adjudication")
    if missing_adjudication:
        print(f"WARNING: {len(missing_adjudication)} flagged/disagreeing jobs lack an "
              f"adjudication and will quarantine: {missing_adjudication[:5]}"
              f"{'...' if len(missing_adjudication) > 5 else ''}")
    print(f"wrote {output} — import with: jev-data annotate import "
          f"--input <panel stage> --registry configs/data/sources.yaml "
          f"--jobs {Path(args.jobs).parent} --reviews {output} "
          f"--allow-test --purpose evaluation --output <reviewed stage>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
