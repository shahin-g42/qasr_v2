"""Deterministic family plans, explicit task cards and honest option-count quotas."""

from __future__ import annotations

from collections import Counter

from jev_contracts import CALIBRATION_SUBSETS, SPLITS, validate_example

from .dedup import family_groups, validate_splits
from .io import digest, fields, integer, number, require, strings, text, version

DEFAULT_SPLITS = {"train": 100000, "development": 10000, "calibration": 5000, "test": 12000}
OPTION_QUOTAS = {"2": 0.10, "3-5": 0.40, "6-16": 0.25, "17-64": 0.20, "65-255": 0.05}


def task_card(value: dict) -> dict:
    fields(value, {"schema_version", "id", "predicate", "semantics", "evidence_requirements",
                   "rubric", "abstention", "deployment_priors", "locale", "risk",
                   "reviewer_qualifications", "reserved_domains", "reserved_ontologies"})
    version(value["schema_version"])
    for key in ("id", "predicate", "evidence_requirements", "abstention", "locale"):
        text(value[key])
    require(value["semantics"] in {"mutually_exclusive", "binary_proposition", "ordinal"},
            "unsupported task semantics; multilabel must use separate decisions")
    require(value["risk"] in {"low", "high"}, "invalid risk")
    strings(value["reviewer_qualifications"])
    strings(value["reserved_domains"])
    strings(value["reserved_ontologies"])
    strings(value["rubric"])
    if value["semantics"] == "ordinal":
        require(2 <= len(value["rubric"]) <= 10, "ordinal task needs 2..10 ordered rubric levels")
    require(type(value["deployment_priors"]) is dict, "priors must be an object")
    if value["deployment_priors"]:
        require(abs(sum(number(p) for p in value["deployment_priors"].values()) - 1) <= 1e-6,
                "priors must sum to one")
    return value


def _choose(identity: str, seed: str, weights: dict[str, int]) -> str:
    total = sum(integer(weight, 0) for weight in weights.values())
    require(total > 0, "split weights cannot all be zero")
    position = int(digest([seed, identity]), 16) % total
    for name, weight in sorted(weights.items()):
        if position < weight:
            return name
        position -= weight
    raise AssertionError("unreachable weighted interval")


def plan(rows: list[dict], *, seed: str, cards: list[dict],
         weights: dict[str, int] | None = None,
         union_groups: list[list[str]] | None = None,
         reservations: dict[str, str] | None = None) -> dict:
    text(seed)
    weights = DEFAULT_SPLITS if weights is None else weights
    require(set(weights) <= SPLITS, "invalid split weight keys")
    cards = [task_card(card) for card in cards]
    require(len({card["id"] for card in cards}) == len(cards), "duplicate task card")
    groups = family_groups(rows, union_groups)
    reservations = dict(reservations or {})
    calibration_groups = {}
    for row in rows:
        require("plan" not in row["provenance"].get("metadata", {}),
                "planned membership is immutable; use a new explicitly versioned input")
        if row["split"] != "train":
            require(reservations.get(row["id"], row["split"]) == row["split"],
                    "protected input cannot be reassigned")
            reservations[row["id"]] = row["split"]
        if row["split"] == "calibration":
            group = groups[row["id"]]
            subset = row["provenance"]["metadata"]["calibration"]["subset"]
            require(calibration_groups.get(group, subset) == subset,
                    "protected calibration subgroups cannot merge")
            calibration_groups[group] = subset
    require(set(reservations) <= set(groups), "reservation references unknown row")
    reserved_groups = {}
    for row_id, split in reservations.items():
        require(split in SPLITS, "invalid reserved split")
        group = groups[row_id]
        require(group not in reserved_groups or reserved_groups[group] == split,
                "conflicting family reservations")
        reserved_groups[group] = split
    output, slots = [], []
    counts: Counter = Counter()
    card_ids = {card["id"]: card for card in cards}
    members: dict[str, list[str]] = {}
    for row_id, group in groups.items():
        members.setdefault(group, []).append(row_id)
    for original in sorted(rows, key=lambda row: row["id"]):
        row = validate_example(original)
        require(row["quality"]["tier"] == "bronze", "accepted rows cannot be replanned")
        require(row["task"] in card_ids, "row has no task card")
        card = card_ids[row["task"]]
        group = groups[row["id"]]
        row["split"] = reserved_groups.get(group, _choose(group, seed, weights))
        meta = row["provenance"].setdefault("metadata", {})
        scope = meta.get("scope", {})
        if (scope.get("domain") in card["reserved_domains"]
                or scope.get("ontology") in card["reserved_ontologies"]):
            require(group in reserved_groups and row["split"] != "train",
                    "reserved domain/ontology requires an explicit protected reservation")
        meta["task"] = {"schema_version": 1, "risk": card["risk"],
                        "qualifications": card["reviewer_qualifications"]}
        meta.pop("calibration", None)
        if row["split"] == "calibration":
            subset = calibration_groups.get(group) or _choose(group, seed + ":calibration", {
                "temperature_fit": 2, "policy_selection": 1, "risk_verification": 1,
            })
            require(subset in CALIBRATION_SUBSETS, "invalid calibration subset")
            meta["calibration"] = {"schema_version": 1, "subset": subset}
        lineage = meta.setdefault("lineage", {"schema_version": 1})
        lineage["group_ids"] = sorted(
            set(lineage.get("group_ids", [])) | {digest(sorted(members[group]))})
        meta["plan"] = {"schema_version": 1, "seed_hash": digest(seed),
                        "task_cards_hash": digest(cards)}
        output.append(validate_example(row))
        count = len(row["question"]["options"])
        bucket = next((name for name, low, high in [
            ("2", 2, 2), ("3-5", 3, 5), ("6-16", 6, 16), ("17-64", 17, 64), ("65-255", 65, 255),
        ] if low <= count <= high), "single")
        if row["question"]["type"] == "choice":
            counts[bucket] += 1
        slots.append({"id": row["id"], "family_group": group, "split": row["split"],
                      "task": row["task"], "option_count": count})
    validate_splits(output)
    total_choice = sum(counts.values())
    return {"schema_version": 1, "seed_hash": digest(seed), "rows": output, "slots": slots,
            "task_cards": cards, "option_quota": {key: {
                "target_fraction": fraction, "actual_unique": counts[key],
                "shortfall": max(0, round(total_choice * fraction) - counts[key]),
            } for key, fraction in OPTION_QUOTAS.items()}}
