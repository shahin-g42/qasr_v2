"""Blind native-human jobs, literal judgments and preserved adjudication evidence."""

from __future__ import annotations

import secrets

from jev_contracts import renderable_input, validate_example

from .io import (
    boolean,
    dataset_identity,
    digest,
    fields,
    guard_test,
    number,
    require,
    strings,
    text,
    version,
)


def review_binding(row: dict) -> str:
    row = validate_example(row)
    return digest({key: value for key, value in row.items() if key not in {"quality", "target"}})


def export_jobs(rows: list[dict], *, rubric_version: str, allow_test: bool = False,
                purpose: str = "training") -> dict:
    dataset_identity(rows)
    guard_test(rows, allow_test=allow_test, purpose=purpose)
    text(rubric_version)
    jobs, mapping = [], {}
    for row in rows:
        opaque = secrets.token_hex(16)
        payload = renderable_input(row)
        jobs.append({"job_id": opaque, "rubric_version": rubric_version, "input": payload})
        mapping[opaque] = {"row_id": row["id"], "binding": review_binding(row),
                           "row_hash": digest(row), "input_hash": digest(payload)}
    return {"schema_version": 1, "rubric_version": rubric_version, "jobs": jobs,
            "private_map": mapping, "dataset": dataset_identity(rows)}


def _reviewer(value: dict, row: dict) -> str:
    fields(value, {"id", "version", "native_varieties", "qualifications"})
    text(value["version"])
    strings(value["native_varieties"])
    strings(value["qualifications"])
    require(row["variety"] in value["native_varieties"], "reviewer not qualified for variety")
    required = row["provenance"].get("metadata", {}).get("task", {}).get("qualifications", [])
    require(set(required) <= set(value["qualifications"]), "reviewer missing domain qualification")
    return text(value["id"])


def _vote(value: dict, row: dict, job_id: str, rubric: str) -> str:
    fields(value, {"job_id", "reviewer", "rubric_version", "input_hash", "label", "independent",
                   "unambiguous", "surface_faithful", "evidence", "elapsed_seconds", "flags"})
    require(value["job_id"] == job_id, "review job ID mismatch")
    require(value["rubric_version"] == rubric, "review rubric mismatch")
    require(value["input_hash"] == digest(renderable_input(row)), "review payload hash mismatch")
    require(type(value["label"]) is str and
            value["label"] in {item["id"] for item in row["question"]["options"]},
            "review label not among supplied options")
    for key in ("independent", "unambiguous", "surface_faithful"):
        boolean(value[key])
    text(value["evidence"])
    number(value["elapsed_seconds"], 0.001, 86400)
    strings(value["flags"])
    return _reviewer(value["reviewer"], row)


def resolve(row: dict, evidence: dict) -> tuple[str | None, list[str]]:
    fields(evidence, {"schema_version", "job_id", "binding", "rubric_version", "original_target",
                      "votes", "adjudication"})
    version(evidence["schema_version"])
    require(evidence["binding"] == review_binding(row), "annotation binding mismatch")
    job_id, rubric = text(evidence["job_id"]), text(evidence["rubric_version"])
    votes = evidence["votes"]
    require(type(votes) is list, "votes must be an array")
    reviewers = [_vote(vote, row, job_id, rubric) for vote in votes]
    require(len(set(reviewers)) == len(reviewers), "duplicate human reviewer")
    author = row["provenance"].get("metadata", {}).get("author", {}).get("reviewer_id")
    require(author not in reviewers, "author cannot be own blind reviewer")
    minimum = 3 if row["split"] == "test" else 2
    reasons = []
    if len(votes) < minimum:
        reasons.append("insufficient_independent_human_votes")
    if not all(vote["independent"] is True for vote in votes):
        reasons.append("nonindependent_review")
    flags = any(vote["flags"] or not vote["unambiguous"] or not vote["surface_faithful"]
                for vote in votes)
    labels = {vote["label"] for vote in votes}
    task = row["provenance"].get("metadata", {}).get("task", {})
    needs_adjudication = (len(labels) != 1 or flags or row["split"] == "test"
                          or task.get("risk") == "high")
    adjudication = evidence["adjudication"]
    label = next(iter(labels)) if len(labels) == 1 else None
    if adjudication is not None:
        fields(adjudication, {"vote", "resolved", "reason", "votes_hash"})
        reviewer = _vote(adjudication["vote"], row, job_id, rubric)
        require(reviewer not in reviewers and reviewer != author, "adjudicator must be distinct")
        boolean(adjudication["resolved"])
        text(adjudication["reason"])
        require(adjudication["votes_hash"] == digest(votes), "adjudication votes hash mismatch")
        vote = adjudication["vote"]
        if not (adjudication["resolved"] and vote["unambiguous"] and vote["surface_faithful"]
                and vote["independent"] and not vote["flags"]):
            reasons.append("unresolved_ambiguity")
        label = vote["label"]
    elif needs_adjudication:
        reasons.append("human_adjudication_required")
    return (None if reasons else label), reasons


def import_reviews(rows: list[dict], bundle: dict, submissions: list[dict], *,
                   allow_test: bool = False, purpose: str = "training") -> list[dict]:
    guard_test(rows, allow_test=allow_test, purpose=purpose)
    fields(bundle, {"schema_version", "rubric_version", "jobs", "private_map", "dataset"})
    version(bundle["schema_version"])
    require(bundle["dataset"] == dataset_identity(rows), "annotation dataset changed")
    by_id = {row["id"]: row for row in rows}
    text(bundle["rubric_version"])
    require(type(bundle["private_map"]) is dict, "invalid private job map")
    require(len(bundle["private_map"]) == len(rows), "annotation job coverage mismatch")
    for job_id, mapping in bundle["private_map"].items():
        require(len(text(job_id)) == 32 and all(c in "0123456789abcdef" for c in job_id),
                "invalid opaque job ID")
        fields(mapping, {"row_id", "binding", "row_hash", "input_hash"})
        require(text(mapping["row_id"]) in by_id, "unknown annotation row ID")
        row = by_id[mapping["row_id"]]
        require(mapping == {"row_id": row["id"], "binding": review_binding(row),
                            "row_hash": digest(row), "input_hash": digest(renderable_input(row))},
                "annotation private mapping mismatch")
    require({entry["row_id"] for entry in bundle["private_map"].values()} == set(by_id),
            "annotation row ID coverage mismatch")
    require(type(bundle["jobs"]) is list, "annotation public jobs must be an array")
    public = set()
    for job in bundle["jobs"]:
        fields(job, {"job_id", "rubric_version", "input"})
        job_id = text(job["job_id"])
        require(job_id in bundle["private_map"] and job_id not in public,
                "annotation public job coverage mismatch")
        public.add(job_id)
        row = by_id[bundle["private_map"][job_id]["row_id"]]
        require(job["rubric_version"] == bundle["rubric_version"]
                and job["input"] == renderable_input(row), "annotation public payload mismatch")
    require(public == set(bundle["private_map"]), "annotation public job coverage mismatch")
    require(type(submissions) is list, "annotation submissions must be an array")
    submitted = {}
    for submission in submissions:
        fields(submission, {"job_id", "votes", "adjudication"})
        key = text(submission["job_id"])
        require(key not in submitted, "duplicate annotation job")
        submitted[key] = submission
    require(set(submitted) == set(bundle["private_map"]), "annotation opaque ID coverage mismatch")
    output = []
    for job_id, mapping in bundle["private_map"].items():
        row = validate_example(by_id[mapping["row_id"]])
        require(row["quality"]["tier"] == "bronze",
                "accepted rows are immutable; create a new version")
        require(mapping["row_hash"] == digest(row), "annotation source row hash mismatch")
        evidence = {"schema_version": 1, "job_id": job_id, "binding": mapping["binding"],
                    "rubric_version": bundle["rubric_version"], "original_target": row["target"],
                    "votes": submitted[job_id]["votes"],
                    "adjudication": submitted[job_id]["adjudication"]}
        label, reasons = resolve(row, evidence)
        prior_quality = row["quality"]
        row["quality"] = {"tier": "gold" if label is not None else "bronze",
                          "audit_ref": "annotation:" + digest(evidence), "metadata": {
                              "annotation": evidence,
                              "history": {"schema_version": 1, "previous_quality": prior_quality},
                              "disposition": {"schema_version": 1,
                                              "state": "adjudicated" if label else "quarantined",
                                              "reasons": reasons},
                          }}
        if label is not None:
            row["target"] = {"kind": "hard", "option_id": label}
            if row["provenance"]["primary_route"] == "oracle":
                from .oracle import check_oracle

                check_oracle(row)
        output.append(validate_example(row))
    return sorted(output, key=lambda row: row["id"])


def validate_gold(row: dict) -> None:
    row = validate_example(row)
    require(row["quality"]["tier"] == "gold", "not Gold")
    evidence = row["quality"].get("metadata", {}).get("annotation")
    require(evidence is not None, "Gold requires actual human annotation evidence")
    label, reasons = resolve(row, evidence)
    require(not reasons and label is not None, "Gold human review gate failed")
    require(row["target"] == {"kind": "hard", "option_id": label},
            "Gold target differs from review")
    require(row["quality"]["audit_ref"] == "annotation:" + digest(evidence),
            "review reference mismatch")
