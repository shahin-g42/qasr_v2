"""Train-pool-only acquisition with uncertainty and representative diversity controls."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .dedup import duplicate_report, family_groups, validate_splits
from .io import dataset_identity, digest, fields, integer, number, require, text
from .registry import Registry


def acquire(rows: list[dict], predictions: list[dict], registry: Registry, *, count: int,
            seed: str, random_fraction: float = 0.25, high_confidence: float = 0.9,
            protected: list[dict] | None = None) -> dict:
    integer(count, 1)
    text(seed)
    number(random_fraction, 0.1, 0.9)
    number(high_confidence, 0.5, 1)
    identity = dataset_identity(rows)
    require(all(row["split"] == "train" for row in rows),
            "acquisition cannot mine development, calibration or sealed test")
    protected = protected or []
    require(all(row["split"] != "train" for row in protected), "invalid protected pool")
    validate_splits(rows + protected)
    for row in rows:
        registry.validate_row(row)
    by_id = {row["id"]: row for row in rows}
    if protected:
        for row in protected:
            registry.validate_row(row)
        duplicates = duplicate_report(rows + protected)
        protected_ids = {row["id"] for row in protected}
        for ids in duplicates["exact"] + duplicates["near"]:
            require(not (set(ids) & by_id.keys() and set(ids) & protected_ids),
                    "acquisition overlaps protected content; independent review required")
    ranked = []
    seen = set()
    groups = family_groups(rows)
    frequency = Counter((row["variety"], row["task"]) for row in rows)
    for prediction in predictions:
        fields(prediction, {"id", "row_hash", "model_revision", "probabilities"}, {"teacher_label"})
        key = text(prediction["id"])
        require(key in by_id and key not in seen, "unknown or duplicate prediction ID")
        seen.add(key)
        row = by_id[key]
        require(prediction["row_hash"] == digest(row), "prediction row binding mismatch")
        text(prediction["model_revision"])
        probabilities = prediction["probabilities"]
        ids = {option["id"] for option in row["question"]["options"]}
        require(type(probabilities) is dict and set(probabilities) == ids,
                "prediction option coverage mismatch")
        values = [number(value) for value in probabilities.values()]
        require(abs(math.fsum(values) - 1) <= 1e-6, "prediction probabilities must sum to one")
        winner = max(sorted(probabilities), key=probabilities.get)
        confidence = probabilities[winner]
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        entropy = entropy / math.log(len(values)) if len(values) > 1 else 0
        flags = []
        if entropy >= 0.5:
            flags.append("uncertain")
        if (row["quality"]["tier"] in {"gold", "silver"} and row["target"]["kind"] == "hard"
                and confidence >= high_confidence and winner != row["target"]["option_id"]):
            from .curate import validate_training_row

            validate_training_row(row, registry)
            flags.append("high_confidence_error")
        if "teacher_label" in prediction:
            require(type(prediction["teacher_label"]) is str and prediction["teacher_label"] in ids,
                    "teacher label outside supplied options")
            if prediction["teacher_label"] != winner:
                flags.append("teacher_student_disagreement")
        cell = (row["variety"], row["task"])
        score = entropy + 2 * ("high_confidence_error" in flags) + (
            "teacher_student_disagreement" in flags) + 1 / frequency[cell]
        ranked.append({"id": key, "row_hash": digest(row), "family_group": groups[key],
                       "flags": flags, "entropy": entropy, "confidence": confidence,
                       "score": score, "cell": list(cell)})
    require(seen == set(by_id), "predictions must cover the exact train pool")
    selected, used = [], set()
    cells = defaultdict(list)
    for item in ranked:
        cells[tuple(item["cell"])].append(item)
    for items in cells.values():
        items.sort(key=lambda item: digest([seed, "random", item["id"]]))
    random_budget = min(count, max(1, round(count * random_fraction)))
    cell_keys = sorted(cells, key=lambda cell: digest([seed, list(cell)]))
    while len(selected) < random_budget and any(cells.values()):
        for cell in cell_keys:
            if cells[cell] and len(selected) < random_budget:
                item = cells[cell].pop()
                if item["family_group"] not in used:
                    selected.append({**item, "flags": item["flags"] + ["random_diversity_control"]})
                    used.add(item["family_group"])
    for item in sorted(ranked, key=lambda item: (-item["score"], digest([seed, item["id"]]))):
        if len(selected) == count:
            break
        if item["family_group"] not in used:
            selected.append(item)
            used.add(item["family_group"])
    return {"schema_version": 1, "dataset": identity, "seed_hash": digest(seed),
            "predictions_hash": digest(predictions), "selected": selected,
            "requested": count, "shortfall": max(0, count - len(selected)),
            "requires_verification": True, "approved": False}
