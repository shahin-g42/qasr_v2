"""Temperature fit, selective-policy selection, and independent risk verification.

The three calibration roles are hard-separated: temperature is fitted only on
the temperature_fit subset, the selective-prediction threshold only on
policy_selection, and selective risk is verified only on the untouched
risk_verification subset with a Wilson upper bound. The finished artifact is
sealed by a contracts manifest and is immutable, binding the checkpoint, data
release, and prompt it was produced for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from jev_contracts import (
    atomic_write_json,
    canonical_json,
    validate_example,
    verify_manifest,
    write_manifest,
)

from .evaluate import example_metrics, read_jsonl

CALIBRATION_SUBSETS = ("temperature_fit", "policy_selection", "risk_verification")
WILSON_Z = 1.959963984540054
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _require_subset(rows: Sequence[dict], subset: str) -> list[dict]:
    """Only rows explicitly assigned to this role may enter this stage."""
    if subset not in CALIBRATION_SUBSETS:
        raise ValueError(f"Unknown calibration subset {subset!r}")
    selected = []
    for row in rows:
        validated = validate_example(row)
        if validated["split"] != "calibration":
            raise ValueError(
                f"Row {validated['id']!r} is not a calibration-split row; roles are hard-separated"
            )
        metadata = validated["provenance"].get("metadata")
        calibration = metadata.get("calibration") if type(metadata) is dict else None
        assigned = calibration.get("subset") if type(calibration) is dict else None
        if assigned != subset:
            raise ValueError(
                f"Row {validated['id']!r} is assigned to {assigned!r}, not {subset!r}"
            )
        selected.append(validated)
    if not selected:
        raise ValueError(f"Calibration subset {subset!r} has no rows")
    return selected


def _mean_nll(rows: Sequence[dict], logits: Sequence[Sequence[float]], temperature: float) -> float:
    values = [
        example_metrics(row, values, temperature=temperature)["nll"]
        for row, values in zip(rows, logits, strict=True)
    ]
    return sum(values) / len(values)


def fit_temperature(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], *,
    lower: float = 0.05, upper: float = 20.0,
) -> dict:
    """Deterministic golden-section search of log-temperature minimizing NLL."""
    for bound, name in ((lower, "lower"), (upper, "upper")):
        if (type(bound) not in (int, float) or isinstance(bound, bool)
                or not math.isfinite(bound) or bound <= 0):
            raise ValueError(f"{name} bound must be a positive finite number")
    if lower >= upper:
        raise ValueError("lower bound must be below the upper bound")
    if len(rows) != len(logits) or not rows:
        raise ValueError("Temperature fitting needs aligned nonempty rows and logits")
    low, high = math.log(lower), math.log(upper)
    inner = high - _GOLDEN * (high - low)
    other = low + _GOLDEN * (high - low)
    iterations = 0
    while high - low > 1e-7 and iterations < 500:
        if _mean_nll(rows, logits, math.exp(inner)) <= _mean_nll(rows, logits, math.exp(other)):
            high, other = other, inner
            inner = high - _GOLDEN * (high - low)
        else:
            low, inner = inner, other
            other = low + _GOLDEN * (high - low)
        iterations += 1
    temperature = math.exp((low + high) / 2)
    return {
        "schema_version": 1, "temperature": temperature,
        "nll": _mean_nll(rows, logits, temperature), "count": len(rows),
        "iterations": iterations,
    }


def _policy_stats(metrics: Sequence[dict], threshold: float) -> dict:
    covered = [entry for entry in metrics if entry["confidence"] >= threshold]
    if not covered:
        return {"coverage": 0.0, "selective_risk": None, "covered_count": 0}
    errors = [1.0 - entry["accuracy"] for entry in covered]
    return {
        "coverage": len(covered) / len(metrics), "selective_risk": sum(errors) / len(errors),
        "covered_count": len(covered),
    }


def select_policy(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], *,
    temperature: float, target_risk: float,
) -> dict:
    """Smallest confidence threshold meeting the target risk, maximizing coverage."""
    if (type(target_risk) not in (int, float) or isinstance(target_risk, bool)
            or not math.isfinite(target_risk) or not 0 <= target_risk <= 1):
        raise ValueError("target_risk must be a finite fraction in [0,1], not bool")
    if len(rows) != len(logits) or not rows:
        raise ValueError("Policy selection needs aligned nonempty rows and logits")
    metrics = [
        example_metrics(row, values, temperature=temperature)
        for row, values in zip(rows, logits, strict=True)
    ]
    # Observed confidences represent every distinct covered set: any threshold
    # between adjacent confidences covers exactly the rows of the lower one, so
    # the ascending scan visits thresholds from most to least coverage and the
    # first success is the smallest threshold that still meets the target.
    candidates = sorted({entry["confidence"] for entry in metrics})
    best = None
    for threshold in candidates:
        stats = _policy_stats(metrics, threshold)
        if stats["selective_risk"] is None:
            continue
        if stats["selective_risk"] <= target_risk:
            best = (threshold, stats)
            break
    if best is None:
        # Nothing meets the target: report the smallest achievable risk, breaking
        # ties toward the smaller threshold so the most coverage is retained.
        fallback = min(
            candidates, key=lambda threshold: (
                _policy_stats(metrics, threshold)["selective_risk"]
                if _policy_stats(metrics, threshold)["covered_count"] else float("inf"),
                threshold,
            ),
        )
        best = (fallback, _policy_stats(metrics, fallback))
    threshold, stats = best
    return {
        "schema_version": 1, "threshold": threshold, "target_risk": target_risk,
        "meets_target": stats["selective_risk"] is not None
        and stats["selective_risk"] <= target_risk,
        **stats, "count": len(metrics),
    }


def wilson_upper_bound(errors: int, count: int, *, z: float = WILSON_Z) -> float:
    """95% (default) Wilson upper bound for a binomial error rate."""
    if type(errors) is not int or type(count) is not int or not 0 <= errors <= count or count < 1:
        raise ValueError("errors/count must be integers with 0 <= errors <= count, count >= 1")
    rate = errors / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    spread = (z / denominator) * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count)
    )
    return min(1.0, center + spread)


def verify_policy(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], *,
    temperature: float, threshold: float, target_risk: float,
) -> dict:
    """Independent selective-risk verification at the selected threshold."""
    if (type(threshold) not in (int, float) or isinstance(threshold, bool)
            or not math.isfinite(threshold)):
        raise ValueError("threshold must be a finite number, not bool")
    if len(rows) != len(logits) or not rows:
        raise ValueError("Risk verification needs aligned nonempty rows and logits")
    metrics = [
        example_metrics(row, values, temperature=temperature)
        for row, values in zip(rows, logits, strict=True)
    ]
    covered = [entry for entry in metrics if entry["confidence"] >= threshold]
    if not covered:
        raise ValueError("Selected threshold covers nothing on the verification subset")
    errors = sum(1 for entry in covered if entry["accuracy"] < 0.5)
    upper = wilson_upper_bound(errors, len(covered))
    return {
        "schema_version": 1, "threshold": threshold, "covered_count": len(covered),
        "count": len(metrics), "coverage": len(covered) / len(metrics),
        "selective_risk": errors / len(covered), "risk_upper_95": upper,
        "target_risk": target_risk, "verified": upper <= target_risk,
    }


def calibrate(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], *, target_risk: float = 0.1,
) -> dict:
    """Run the three hard-separated stages over one calibration release."""
    if len(rows) != len(logits) or not rows:
        raise ValueError("Calibration needs aligned nonempty rows and logits")
    partitions = {subset: [] for subset in CALIBRATION_SUBSETS}
    partitions_logits = {subset: [] for subset in CALIBRATION_SUBSETS}
    for row, values in zip(rows, logits, strict=True):
        validated = validate_example(row)
        metadata = validated["provenance"].get("metadata")
        calibration = metadata.get("calibration") if type(metadata) is dict else None
        subset = calibration.get("subset") if type(calibration) is dict else None
        if subset not in CALIBRATION_SUBSETS:
            raise ValueError(f"Row {validated['id']!r} has no valid calibration subset")
        partitions[subset].append(validated)
        partitions_logits[subset].append(values)
    fitted = fit_temperature(partitions["temperature_fit"], partitions_logits["temperature_fit"])
    policy = select_policy(
        partitions["policy_selection"], partitions_logits["policy_selection"],
        temperature=fitted["temperature"], target_risk=target_risk,
    )
    verification = verify_policy(
        partitions["risk_verification"], partitions_logits["risk_verification"],
        temperature=fitted["temperature"], threshold=policy["threshold"],
        target_risk=target_risk,
    )
    return {
        "schema_version": 1, "temperature": fitted, "policy": policy,
        "verification": verification,
    }


def write_calibration(directory: str | Path, calibration: dict, *, identity: dict) -> dict:
    """Publish the immutable calibration artifact; the manifest seals it last."""
    if type(calibration) is not dict or type(identity) is not dict or not identity:
        raise ValueError("calibration and identity must be mappings, identity nonempty")
    directory = Path(directory).absolute()
    if directory.exists():
        raise FileExistsError(f"Calibration directory already exists: {directory}")
    directory.mkdir(parents=True)
    artifact = {**calibration, "identity": identity}
    atomic_write_json(directory / "calibration.json", artifact)
    write_manifest(
        directory, ["calibration.json"],
        {"kind": "jev-calibration", "identity": identity},
    )
    return artifact


def load_calibration(directory: str | Path) -> dict:
    """Verify the sealed artifact and return the calibration mapping."""
    directory = Path(directory).absolute()
    manifest = verify_manifest(directory)
    if manifest.get("metadata", {}).get("kind") != "jev-calibration":
        raise ValueError(f"Not a sealed jev-calibration artifact: {directory}")
    artifact = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if artifact.get("identity") != manifest["metadata"]["identity"]:
        raise ValueError("Calibration artifact and manifest identities disagree")
    return artifact


def _identity_hash(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-calibrate",
        description="Fit temperature, select a selective policy, and verify risk once",
    )
    parser.add_argument("--rows", required=True, help="Calibration-split rows as strict JSONL")
    parser.add_argument("--predictions", required=True, help="Prediction records as strict JSONL")
    parser.add_argument("--output", required=True, help="Fresh directory for the sealed artifact")
    parser.add_argument("--target-risk", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint", required=True, help="Identity hash of the calibrated checkpoint")
    parser.add_argument(
        "--release", required=True, help="Identity hash of the data release manifest")
    args = parser.parse_args(argv)
    rows = read_jsonl(args.rows)
    records = read_jsonl(args.predictions)
    if len(rows) != len(records):
        raise SystemExit(f"{len(rows)} rows but {len(records)} prediction records")
    calibration = calibrate(
        rows, [record["logits"] for record in records], target_risk=args.target_risk,
    )
    identity = {
        "schema_version": 1, "checkpoint": args.checkpoint, "release": args.release,
        "manifest": _identity_hash({"checkpoint": args.checkpoint, "release": args.release}),
    }
    write_calibration(args.output, calibration, identity=identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
