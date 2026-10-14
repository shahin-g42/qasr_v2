"""Offline decision metrics over cached logits, with grouped summaries.

This module is pure Python: data workers can score predictions without ever
importing Torch. Metrics are computed from complete candidate distributions
only — no verbal confidence, no renormalization, no silently dropped examples.
The paired family bootstrap resamples whole families so related views cannot
manufacture statistical significance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from jev_contracts import atomic_write_json, target_distribution, validate_example

ECE_BINS = 10
_BOOTSTRAP_METRICS = ("nll", "brier", "accuracy")


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not bool")
    return float(value)


def probabilities(logits: Sequence[float], *, temperature: float) -> list[float]:
    """Softmax over exactly the supplied candidate logits at one temperature."""
    if (type(temperature) not in (int, float) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be a positive finite number, not bool")
    values = [_finite_float(value, "logit") for value in logits]
    if not values:
        raise ValueError("Candidate logits must not be empty")
    peak = max(values)
    weights = [math.exp((value - peak) / temperature) for value in values]
    total = sum(weights)
    if not total > 0:
        raise ValueError("Logit range underflows at this temperature; refine inputs or temperature")
    result = [weight / total for weight in weights]
    if abs(sum(result) - 1.0) > 1e-9:
        raise ValueError("Probability normalization failed")
    return result


def example_metrics(row: dict, logits: Sequence[float], *, temperature: float = 1.0) -> dict:
    """Every per-example quantity the aggregates and policies are built from."""
    validated = validate_example(row)
    options = validated["question"]["options"]
    count = len(options)
    if len(logits) != count:
        raise ValueError(
            f"Row {validated['id']!r} has {count} candidates but {len(logits)} logits"
        )
    predicted = probabilities(logits, temperature=temperature)
    target = target_distribution(validated)
    if len(target) != count:
        raise ValueError("Target distribution and candidate count disagree")
    nll = 0.0
    for probability, weight in zip(predicted, target, strict=True):
        if weight > 0:
            if probability <= 0:
                raise ValueError("Positive-target candidate received zero probability")
            nll -= weight * math.log(probability)
    brier = sum(
        (probability - weight) ** 2
        for probability, weight in zip(predicted, target, strict=True)
    )
    expected_prediction = sum(
        index * probability for index, probability in enumerate(predicted)
    )
    expected_target = sum(index * weight for index, weight in enumerate(target))
    cumulative = 0.0
    cumulative_target = 0.0
    rps = 0.0
    for probability, weight in zip(predicted, target, strict=True):
        cumulative += probability
        cumulative_target += weight
        rps += (cumulative - cumulative_target) ** 2
    question_type = validated["question"]["type"]
    correct = max(range(count), key=lambda index: predicted[index]) == max(
        range(count), key=lambda index: target[index]
    )
    return {
        "id": validated["id"], "type": question_type, "count": count, "nll": nll,
        "brier": brier, "accuracy": float(correct), "confidence": max(predicted),
        "binary_brier": (predicted[1] - target[1]) ** 2 if question_type == "noul" else None,
        "ordinal_mae": (
            abs(expected_prediction - expected_target) if question_type == "score" else None
        ),
        "rps": rps if question_type == "score" else None,
        "expected_prediction": expected_prediction, "expected_target": expected_target,
    }


def _aggregate(metrics: Sequence[dict]) -> dict:
    count = len(metrics)
    if count == 0:
        raise ValueError("Cannot aggregate an empty metric set")

    def mean(name: str, subset: Sequence[dict]) -> float:
        return sum(entry[name] for entry in subset) / len(subset)

    binary = [entry for entry in metrics if entry["binary_brier"] is not None]
    ordinal = [entry for entry in metrics if entry["ordinal_mae"] is not None]
    result = {
        "count": count, "nll": mean("nll", metrics), "brier": mean("brier", metrics),
        "accuracy": mean("accuracy", metrics),
        "binary_brier_count": len(binary),
        "ordinal_count": len(ordinal),
    }
    if binary:
        result["binary_brier"] = mean("binary_brier", binary)
    if ordinal:
        result["ordinal_mae"] = mean("ordinal_mae", ordinal)
        result["rps"] = mean("rps", ordinal)
    bins: list[list[dict]] = [[] for _ in range(ECE_BINS)]
    for entry in metrics:
        index = min(ECE_BINS - 1, int(entry["confidence"] * ECE_BINS))
        bins[index].append(entry)
    ece = 0.0
    for slot in bins:
        if slot:
            ece += (len(slot) / count) * abs(mean("accuracy", slot) - mean("confidence", slot))
    result["ece"] = ece
    return result


def _group_keys(row: dict) -> list[tuple[str, str]]:
    """Preregistered groupings; every summary stays deterministic and sortable."""
    provenance = row["provenance"]
    origin = ",".join(
        f"{flag}={str(provenance.get(flag)).lower()}"
        for flag in ("native_origin", "human_authored", "localized", "synthetic")
    )
    return [
        ("family", row["family_id"]), ("task", row["task"]),
        ("type", row["question"]["type"]), ("language", row["language"]),
        ("variety", row["variety"]), ("origin", origin),
    ]


def evaluate_predictions(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], *, temperature: float = 1.0,
) -> dict:
    """Overall and grouped metrics for aligned rows and cached logits."""
    if len(rows) != len(logits):
        raise ValueError(f"{len(rows)} rows but {len(logits)} logit vectors")
    metrics = [
        example_metrics(row, values, temperature=temperature)
        for row, values in zip(rows, logits, strict=True)
    ]
    groups: dict[str, dict[str, list[dict]]] = {}
    for row, entry in zip(rows, metrics, strict=True):
        for name, key in _group_keys(row):
            groups.setdefault(name, {}).setdefault(key, []).append(entry)
    return {
        "schema_version": 1, "count": len(metrics), "temperature": float(temperature),
        "overall": _aggregate(metrics),
        "groups": {
            name: {key: _aggregate(entries) for key, entries in sorted(buckets.items())}
            for name, buckets in sorted(groups.items())
        },
    }


def evaluate_records(
    rows: Sequence[dict], records: Sequence[dict], *, temperature: float = 1.0,
) -> dict:
    """Score prediction records; every row is matched exactly once, in order."""
    by_id: dict[str, dict] = {}
    for row in rows:
        validated = validate_example(row)
        if validated["id"] in by_id:
            raise ValueError(f"Duplicate row ID: {validated['id']!r}")
        by_id[validated["id"]] = validated
    seen: set[str] = set()
    aligned_rows: list[dict] = []
    aligned_logits: list[list[float]] = []
    for record in records:
        if type(record) is not dict:
            raise ValueError("Each prediction record must be an object")
        row_id = record.get("id")
        option_ids = record.get("option_ids")
        values = record.get("logits")
        if type(row_id) is not str or row_id in seen:
            raise ValueError(f"Prediction records must carry unique string IDs: {row_id!r}")
        if row_id not in by_id:
            raise ValueError(f"Prediction for unknown row ID: {row_id!r}")
        if (not isinstance(option_ids, list) or not isinstance(values, list)
                or len(option_ids) != len(values)):
            raise ValueError("Records need aligned option_ids and logits arrays")
        row = by_id[row_id]
        expected = [option["id"] for option in row["question"]["options"]]
        if list(option_ids) != expected:
            raise ValueError(
                f"Record {row_id!r} candidate order differs from the row's option order"
            )
        seen.add(row_id)
        aligned_rows.append(row)
        aligned_logits.append([_finite_float(value, "logit") for value in values])
    if seen != set(by_id):
        missing = sorted(set(by_id) - seen)
        raise ValueError(
            f"Rows without predictions: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return evaluate_predictions(aligned_rows, aligned_logits, temperature=temperature)


def paired_family_bootstrap(
    rows: Sequence[dict], logits_a: Sequence[Sequence[float]],
    logits_b: Sequence[Sequence[float]], *, metric: str = "nll", seed: int = 0,
    replicates: int = 1000, temperature: float = 1.0,
) -> dict:
    """Family-level paired resampling of the mean metric difference (A minus B)."""
    if metric not in _BOOTSTRAP_METRICS:
        raise ValueError(f"Bootstrap metric must be one of {sorted(_BOOTSTRAP_METRICS)}")
    if type(seed) is not int or seed < 0 or type(replicates) is not int or replicates < 1:
        raise ValueError("seed must be a nonnegative int and replicates a positive int")
    if len(rows) != len(logits_a) or len(rows) != len(logits_b):
        raise ValueError("Both systems must predict the identical row sequence")
    import random

    families: dict[str, list[tuple[float, float]]] = {}
    for row, values_a, values_b in zip(rows, logits_a, logits_b, strict=True):
        first = example_metrics(row, values_a, temperature=temperature)
        second = example_metrics(row, values_b, temperature=temperature)
        families.setdefault(row["family_id"], []).append((first[metric], second[metric]))
    if not families:
        raise ValueError("Cannot bootstrap without examples")
    flat_a = [value for pairs in families.values() for value, _ in pairs]
    flat_b = [value for pairs in families.values() for _, value in pairs]
    difference = sum(flat_a) / len(flat_a) - sum(flat_b) / len(flat_b)
    order = sorted(families)
    generator = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        total_a = total_b = total_n = 0.0
        for _ in range(len(order)):
            pairs = families[order[generator.randrange(len(order))]]
            total_a += sum(value for value, _ in pairs)
            total_b += sum(value for _, value in pairs)
            total_n += len(pairs)
        deltas.append((total_a - total_b) / total_n)
    deltas.sort()
    low = deltas[int(math.floor(0.025 * (replicates - 1)))]
    high = deltas[int(math.ceil(0.975 * (replicates - 1)))]
    return {
        "schema_version": 1, "metric": metric, "difference": difference,
        "ci_low": low, "ci_high": high, "replicates": replicates, "seed": seed,
        "families": len(order), "count": len(flat_a),
    }


def read_jsonl(path: str | Path) -> list:
    """Strict JSONL rows: blank lines and non-object entries fail."""
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line at {path}:{number}")
        value = json.loads(line)
        if type(value) is not dict:
            raise ValueError(f"JSONL entries must be objects at {path}:{number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-eval",
        description="Score cached decision logits against a labeled row set",
    )
    parser.add_argument("--rows", required=True, help="Labeled rows as strict JSONL")
    parser.add_argument("--predictions", required=True, help="Prediction records as strict JSONL")
    parser.add_argument("--output", required=True, help="Where to write the canonical result JSON")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--compare", default=None, help="Second prediction file for a paired bootstrap")
    parser.add_argument("--bootstrap-metric", default="nll", choices=sorted(_BOOTSTRAP_METRICS))
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    rows = read_jsonl(args.rows)
    records = read_jsonl(args.predictions)
    result = evaluate_records(rows, records, temperature=args.temperature)
    if args.compare:
        compared = read_jsonl(args.compare)
        result["bootstrap"] = paired_family_bootstrap(
            rows, [record["logits"] for record in records],
            [record["logits"] for record in compared],
            metric=args.bootstrap_metric, seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates, temperature=args.temperature,
        )
    atomic_write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
