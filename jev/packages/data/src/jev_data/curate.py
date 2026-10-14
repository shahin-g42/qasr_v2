"""Dataset checks and reproducible family-level human audits, never tier-string approval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from jev_contracts import TRAINING_TIERS

from .annotation import validate_gold
from .arabic import fidelity_flags
from .dedup import duplicate_report, family_groups, validate_splits
from .io import (
    boolean,
    dataset_identity,
    digest,
    fields,
    integer,
    number,
    require,
    strings,
    text,
    version,
)
from .registry import Registry
from .verify import task_checks, validate_silver

RULE_VERSION = "jev-quality-v1"
DEFAULT_POLICY = {"schema_version": 1, "rule_version": RULE_VERSION, "sample_families": 500,
                  "overall_error_upper": 0.02, "cell_error_upper": 0.05,
                  "confidence": 0.95, "near_threshold": 0.92}


def policy_config(value: dict | None = None) -> dict:
    value = dict(DEFAULT_POLICY if value is None else value)
    fields(value, set(DEFAULT_POLICY))
    version(value["schema_version"])
    require(value["rule_version"] == RULE_VERSION, "unsupported quality rules")
    integer(value["sample_families"], 500, 1000000)
    number(value["overall_error_upper"], 0, 0.02)
    number(value["cell_error_upper"], 0, 0.05)
    require(type(value["confidence"]) is float and value["confidence"] == 0.95,
            "only preregistered 95% Wilson bounds are supported")
    number(value["near_threshold"], 0.5, 0.99)
    return value


def validate_training_row(row: dict, registry: Registry) -> dict:
    row = registry.validate_row(row, release=True)
    require(row["quality"]["tier"] in TRAINING_TIERS, "Bronze cannot enter an approved release")
    if row["quality"]["tier"] == "gold":
        validate_gold(row)
    else:
        validate_silver(row)
    if row["provenance"]["primary_route"] in {"oracle", "synthetic"}:
        task_checks(row)
    return row


def curate(rows: list[dict], registry: Registry, *, protected: list[dict] | None = None,
           union_groups: list[list[str]] | None = None, resolutions: list[dict] | None = None,
           policy: dict | None = None) -> dict:
    policy = policy_config(policy)
    protected, union_groups, resolutions = protected or [], union_groups or [], resolutions or []
    identity = dataset_identity(rows)
    protected_identity = dataset_identity(protected)
    all_rows = rows + protected
    dataset_identity(all_rows)
    require(all(row["split"] != "train" for row in protected), "protected pool contains train rows")
    for row in rows:
        validate_training_row(row, registry)
    for row in protected:
        registry.validate_row(row)
    split_index = validate_splits(all_rows, union_groups)
    duplicates = duplicate_report(all_rows, policy["near_threshold"])
    errors = [{"reason": "exact_duplicate", "ids": sorted(ids)} for ids in duplicates["exact"]]
    by_id = {row["id"]: row for row in all_rows}
    decisions = {}
    for resolution in resolutions:
        fields(resolution, {"ids", "row_hashes", "reviewer_id",
                            "independent", "decision", "reason"})
        ids = strings(resolution["ids"])
        require(len(ids) == 2 and set(ids) <= by_id.keys(),
                "invalid near-duplicate resolution IDs")
        key = tuple(sorted(ids))
        require(key not in decisions, "duplicate near-match resolution")
        require(resolution["row_hashes"] == {item: digest(by_id[item]) for item in ids},
                "near-match resolution belongs to different rows")
        text(resolution["reviewer_id"])
        text(resolution["reason"])
        require(boolean(resolution["independent"]) is True, "near-match review must be independent")
        require(resolution["decision"] in {"distinct", "linked"}, "invalid near-match decision")
        if resolution["decision"] == "linked":
            require(split_index["groups"][ids[0]] == split_index["groups"][ids[1]],
                    "linked near matches require explicit union before audit")
        decisions[key] = resolution
    for pair in duplicates["near"]:
        if tuple(sorted(pair)) not in decisions:
            errors.append({"reason": "near_duplicate_requires_review", "ids": pair})
    flags = []
    for row in rows:
        if type(row["state"]) is str:
            for flag in fidelity_flags(row["state"]):
                flags.append({"id": row["id"], "flag": flag})
                if flag != "negation_present":
                    errors.append({"reason": flag, "ids": [row["id"]]})
    distributions = {axis: dict(Counter(row[axis] for row in rows))
                     for axis in ("split", "task", "language", "variety")}
    distributions["source"] = dict(Counter(
        source for row in rows for source in row["provenance"]["source_ids"]))
    distributions["tier"] = dict(Counter(row["quality"]["tier"] for row in rows))
    distributions["option_count"] = dict(Counter(
        str(len(row["question"]["options"])) for row in rows))
    distributions["origin"] = dict(Counter(row["provenance"]["origin"] for row in rows))
    arabic = [row for row in rows if row["language"] == "ar"]
    if arabic:
        native_share = sum(row["provenance"]["native_origin"] for row in arabic) / len(arabic)
        localized_share = sum(row["provenance"]["localized"] for row in arabic) / len(arabic)
        if native_share < 0.70 or localized_share > 0.20:
            errors.append({"reason": "arabic_origin_mix_gate", "ids": []})
    synthetic_count = sum(row["provenance"]["primary_route"] == "synthetic" for row in rows)
    if rows and synthetic_count / len(rows) > 0.20:
        errors.append({"reason": "synthetic_route_cap_requires_new_mixture_revision", "ids": []})
    # Canonical order: reports must be a pure function of the row set, never of the
    # input sequence (freeze validates the stage order, load_release the split order).
    errors.sort(key=lambda item: (item["reason"], item["ids"]))
    flags.sort(key=lambda item: (item["id"], item["flag"]))
    return {"schema_version": 1, "rule_version": RULE_VERSION, "dataset": identity,
            "registry_hash": digest(registry.to_dict()), "protected_dataset": protected_identity,
            "split_index": split_index, "policy_hash": digest(policy), "passed": not errors,
            "errors": errors, "flags": flags, "distributions": distributions,
            "unique_decisions": len(rows),
            "unique_families": len({split_index["groups"][row["id"]] for row in rows})}


def wilson_upper(errors: int, n: int) -> float:
    integer(n, 1)
    integer(errors, 0, n)
    z = 1.959963984540054
    p = errors / n
    return (p + z*z / (2*n) + z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def audit_plan(rows: list[dict], registry: Registry, *, seed: str,
               policy: dict | None = None, union_groups: list[list[str]] | None = None) -> dict:
    policy = policy_config(policy)
    text(seed)
    require(bool(rows), "cannot audit an empty dataset")
    identity = dataset_identity(rows)
    groups = family_groups(rows, union_groups)
    cells: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key in ["overall", "task:" + row["task"], "variety:" + row["variety"],
                    "split:" + row["split"],
                    *["source:" + source for source in row["provenance"]["source_ids"]],
                    *["cell:" + digest([row["task"], row["variety"], source])
                      for source in row["provenance"]["source_ids"]]]:
            cells[key][groups[row["id"]]].append(row["id"])
    samples = {}
    for cell, families in sorted(cells.items()):
        selected = sorted(families, key=lambda family: digest([seed, cell, family]))
        selected = selected[:policy["sample_families"]]
        samples[cell] = {"families": {family: sorted(families[family]) for family in selected},
                         "population_families": len(families),
                         "inclusion_probability": len(selected) / len(families)}
    return {"schema_version": 1, "rule_version": RULE_VERSION, "dataset": identity,
            "registry_hash": digest(registry.to_dict()), "seed": seed, "policy": policy,
            "union_groups": union_groups or [], "sampling": "simple_random_families_per_cell",
            "samples": samples}


def audit(rows: list[dict], registry: Registry, *, plan: dict, reviews: list[dict],
          protected: list[dict] | None = None, resolutions: list[dict] | None = None) -> dict:
    fields(plan, {"schema_version", "rule_version", "dataset", "registry_hash", "seed", "policy",
                  "union_groups", "sampling", "samples"})
    expected = audit_plan(rows, registry, seed=plan["seed"], policy=plan["policy"],
                          union_groups=plan["union_groups"])
    require(expected == plan,
            "audit preregistration does not match dataset, registry or sampling rules")
    curation = curate(rows, registry, protected=protected, union_groups=plan["union_groups"],
                      resolutions=resolutions, policy=plan["policy"])
    by_id = {row["id"]: row for row in rows}
    needed = {key for sample in plan["samples"].values()
              for ids in sample["families"].values() for key in ids}
    evidence = {}
    for review in reviews:
        fields(review, {"id", "row_hash", "plan_hash", "reviewer_id", "reviewer_version",
                        "native_variety",
                        "independent", "critical_error", "evidence", "elapsed_seconds"})
        key = text(review["id"])
        require(key in needed and key not in evidence, "unexpected or duplicate audit review ID")
        require(review["row_hash"] == digest(by_id[key]), "audit row hash mismatch")
        require(review["plan_hash"] == digest(plan),
                "audit evidence from a different preregistration")
        for name in ("reviewer_id", "reviewer_version", "evidence"):
            text(review[name])
        require(review["native_variety"] == by_id[key]["variety"],
                "audit reviewer variety mismatch")
        require(boolean(review["independent"]) is True, "audit must be independent")
        boolean(review["critical_error"])
        number(review["elapsed_seconds"], 0.001, 86400)
        annotation = by_id[key]["quality"].get("metadata", {}).get("annotation", {})
        reviewers = {vote["reviewer"]["id"] for vote in annotation.get("votes", [])}
        if annotation.get("adjudication"):
            reviewers.add(annotation["adjudication"]["vote"]["reviewer"]["id"])
        author = by_id[key]["provenance"].get("metadata", {}).get("author", {}).get("reviewer_id")
        require(review["reviewer_id"] not in reviewers and review["reviewer_id"] != author,
                "quality auditor must be independent of original author/labelers")
        evidence[key] = review
    missing = sorted(needed - evidence.keys())
    results = {}
    for cell, sample in plan["samples"].items():
        families = sample["families"]
        complete = {family: ids for family, ids in families.items() if set(ids) <= evidence.keys()}
        errors = sum(any(evidence[key]["critical_error"] for key in ids)
                     for ids in complete.values())
        upper = wilson_upper(errors, len(complete)) if complete else 1.0
        threshold = plan["policy"][
            "overall_error_upper" if cell == "overall" else "cell_error_upper"]
        results[cell] = {"reviewed_families": len(complete), "required_families": len(families),
                         "critical_error_families": errors, "wilson_upper_95": upper,
                         "threshold": threshold,
                         "inclusion_probability": sample["inclusion_probability"],
                         "passed": len(complete) == len(families) and upper <= threshold}
    return {"schema_version": 1, "kind": "dataset_quality_audit", "rule_version": RULE_VERSION,
            "dataset": dataset_identity(rows), "registry_hash": digest(registry.to_dict()),
            "plan": plan, "reviews": sorted(reviews, key=lambda item: item["id"]),
            "resolutions": resolutions or [], "protected_rows": protected or [],
            "curation": curation, "results": results, "missing_review_ids": missing,
            "passed": (curation["passed"] and not missing
                       and all(item["passed"] for item in results.values()))}


def validate_audit(rows: list[dict], registry: Registry, report: dict) -> None:
    fields(report, {"schema_version", "kind", "rule_version", "dataset", "registry_hash", "plan",
                    "reviews", "resolutions", "protected_rows", "curation", "results",
                    "missing_review_ids", "passed"})
    version(report["schema_version"])
    require(report["kind"] == "dataset_quality_audit", "invalid audit kind")
    require(report["passed"] is True, "human quality audit has not passed")
    require(report["dataset"] == dataset_identity(rows),
            "audit cannot be reused with different data")
    expected = audit(rows, registry, plan=report["plan"], reviews=report["reviews"],
                     protected=report["protected_rows"], resolutions=report["resolutions"])
    require(expected == report, "audit evidence/results cannot be reproduced")
