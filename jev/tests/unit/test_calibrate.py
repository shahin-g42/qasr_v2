from __future__ import annotations

import hashlib
import json
import math

import pytest

from jev import calibrate
from jev.calibrate import (
    fit_temperature,
    load_calibration,
    select_policy,
    verify_policy,
    wilson_upper_bound,
    write_calibration,
)
from jev_contracts import (
    ArtifactError,
    atomic_write_json,
    canonical_json,
    write_manifest,
)

from .test_processing import make_row


def calibration_row(index, subset, *, count=2, target=None, family=None):
    row = make_row("choice", count)
    row["id"] = f"cal-{index:03d}"
    row["family_id"] = family or f"cal-family-{index % 2}"
    row["split"] = "calibration"
    row["provenance"]["metadata"] = {
        "calibration": {"schema_version": 1, "subset": subset},
    }
    if target is not None:
        row["target"] = target
    return row


def hard(option_id):
    return {"kind": "hard", "option_id": option_id}


def test_require_subset_enforces_hard_role_separation():
    rows = [calibration_row(0, "temperature_fit"), calibration_row(1, "policy_selection")]
    selected = calibrate._require_subset(rows[:1], "temperature_fit")
    assert [row["id"] for row in selected] == ["cal-000"]
    with pytest.raises(ValueError, match="not a calibration-split"):
        calibrate._require_subset([make_row()], "temperature_fit")
    with pytest.raises(ValueError, match="assigned to"):
        calibrate._require_subset(rows, "temperature_fit")
    with pytest.raises(ValueError, match="no rows"):
        calibrate._require_subset([], "temperature_fit")
    with pytest.raises(ValueError, match="Unknown calibration subset"):
        calibrate._require_subset(rows, "other")


def test_fit_temperature_recovers_a_known_scale():
    # Teacher distribution softmax([8, 0] / 4) seen through the same raw logits
    # [8, 0]: the NLL-minimizing temperature divides the logit gap back down to
    # 2, recovering the teacher's scale exactly.
    weight = 1.0 / (1.0 + math.exp(-2.0))
    distribution = {"kind": "distribution", "probabilities": {"0": weight, "1": 1.0 - weight}}
    rows = [calibration_row(index, "temperature_fit", target=distribution)
            for index in range(4)]
    logits = [[8.0, 0.0]] * 4
    fitted = fit_temperature(rows, logits)
    assert fitted["schema_version"] == 1 and fitted["count"] == 4
    assert fitted["temperature"] == pytest.approx(4.0, abs=1e-3)
    assert fitted == fit_temperature(rows, logits)
    unit = fit_temperature(rows, logits, lower=0.999, upper=1.001)
    assert fitted["nll"] <= unit["nll"]
    for bounds in ((0.0, 1.0), (2.0, 2.0), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            fit_temperature(rows, logits, lower=bounds[0], upper=bounds[1])
    with pytest.raises(ValueError, match="aligned"):
        fit_temperature(rows, logits[:1])
    with pytest.raises(ValueError, match="aligned"):
        fit_temperature([], [])


def test_select_policy_prefers_the_smallest_threshold_meeting_target():
    low, high = math.log(1.5), math.log(9.0)  # confidences exactly 0.6 and 0.9
    plan = [(low, True), (low, False), (high, True), (high, True)]
    rows, logits = [], []
    for index, (gap, correct) in enumerate(plan):
        rows.append(calibration_row(index, "policy_selection", target=hard(
            "0" if correct else "1")))
        logits.append([gap, 0.0])
    generous = select_policy(rows, logits, temperature=1.0, target_risk=0.25)
    assert generous["threshold"] == pytest.approx(0.6)
    assert generous["meets_target"] is True
    assert generous["coverage"] == 1.0 and generous["covered_count"] == 4
    assert generous["selective_risk"] == pytest.approx(0.25)
    strict = select_policy(rows, logits, temperature=1.0, target_risk=0.1)
    assert strict["threshold"] == pytest.approx(0.9)
    assert strict["meets_target"] is True
    assert strict["coverage"] == pytest.approx(0.5)
    for bad_risk in (1.5, -0.1, True, float("nan")):
        with pytest.raises(ValueError, match="target_risk"):
            select_policy(rows, logits, temperature=1.0, target_risk=bad_risk)
    with pytest.raises(ValueError, match="aligned"):
        select_policy(rows, logits[:1], temperature=1.0, target_risk=0.1)


def test_select_policy_falls_back_to_the_most_coverage():
    rows, logits = [], []
    for index, gap in enumerate([math.log(1.5)] * 2 + [math.log(9.0)] * 2):
        rows.append(calibration_row(index, "policy_selection", target=hard("1")))
        logits.append([gap, 0.0])
    result = select_policy(rows, logits, temperature=1.0, target_risk=0.1)
    assert result["meets_target"] is False
    assert result["selective_risk"] == pytest.approx(1.0)
    # Every threshold carries risk 1.0; ties break toward the smaller threshold.
    assert result["threshold"] == pytest.approx(0.6)
    assert result["coverage"] == 1.0


def test_wilson_upper_bound_properties():
    zero = wilson_upper_bound(0, 100)
    assert 0.0 < zero < 0.05
    assert wilson_upper_bound(0, 400) < zero
    assert wilson_upper_bound(10, 10) == pytest.approx(1.0)
    assert wilson_upper_bound(5, 10) > 0.5
    for errors, count in ((11, 10), (0, 0), (-1, 10)):
        with pytest.raises(ValueError):
            wilson_upper_bound(errors, count)
    with pytest.raises(ValueError):
        wilson_upper_bound(True, 10)


def test_verify_policy_counts_only_covered_hard_errors():
    rows, logits = [], []
    plan = [(math.log(9.0), True), (math.log(9.0), False), (math.log(1.5), True), (0.0, True)]
    for index, (gap, correct) in enumerate(plan):
        rows.append(calibration_row(index, "risk_verification", target=hard(
            "0" if correct else "1")))
        logits.append([gap, 0.0])
    result = verify_policy(rows, logits, temperature=1.0, threshold=0.6, target_risk=0.5)
    assert result["covered_count"] == 3 and result["count"] == 4
    assert result["coverage"] == pytest.approx(0.75)
    assert result["selective_risk"] == pytest.approx(1 / 3)
    assert result["risk_upper_95"] > 1 / 3
    assert result["verified"] is (result["risk_upper_95"] <= 0.5)
    with pytest.raises(ValueError, match="covers nothing"):
        verify_policy(rows, logits, temperature=1.0, threshold=0.95, target_risk=0.1)
    with pytest.raises(ValueError, match="threshold"):
        verify_policy(rows, logits, temperature=1.0, threshold=True, target_risk=0.1)
    with pytest.raises(ValueError, match="aligned"):
        verify_policy([], [], temperature=1.0, threshold=0.5, target_risk=0.1)


def test_calibrate_runs_the_three_hard_separated_stages():
    rows, logits = [], []
    weight = 1.0 / (1.0 + math.exp(-2.0))
    distribution = {"kind": "distribution", "probabilities": {"0": weight, "1": 1.0 - weight}}
    for index in range(2):
        rows.append(calibration_row(index, "temperature_fit", target=distribution,
                                    family="tf"))
        logits.append([8.0, 0.0])
    for subset, base in (("policy_selection", 2), ("risk_verification", 4)):
        for index in range(2):
            rows.append(calibration_row(base + index, subset,
                                        target=hard("0" if index == 0 else "1"),
                                        family=subset))
            logits.append([math.log(9.0), 0.0])
    artifact = calibrate.calibrate(rows, logits, target_risk=0.5)
    assert artifact["schema_version"] == 1
    assert artifact["temperature"]["temperature"] == pytest.approx(4.0, abs=1e-3)
    assert artifact["policy"]["meets_target"] is True
    assert artifact["verification"]["covered_count"] == 2
    assert artifact["verification"]["verified"] is False
    with pytest.raises(ValueError, match="no valid calibration subset"):
        calibrate.calibrate([make_row()], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="aligned"):
        calibrate.calibrate(rows, logits[:1])


def test_calibration_artifact_is_sealed_and_identity_bound(tmp_path):
    artifact = {
        "schema_version": 1,
        "temperature": {"schema_version": 1, "temperature": 1.5, "nll": 0.4,
                        "count": 2, "iterations": 7},
        "policy": {"schema_version": 1, "threshold": 0.9},
        "verification": {"schema_version": 1, "verified": True},
    }
    identity = {"checkpoint": "c" * 64, "release": "r" * 64}
    directory = tmp_path / "calibration"
    written = write_calibration(directory, artifact, identity=identity)
    assert written == {**artifact, "identity": identity}
    assert load_calibration(directory) == written
    (directory / "calibration.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_calibration(directory)
    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    atomic_write_json(mismatch / "calibration.json", {**artifact, "identity": identity})
    write_manifest(mismatch, ["calibration.json"],
                   {"kind": "jev-calibration", "identity": {"checkpoint": "x" * 64}})
    with pytest.raises(ValueError, match="disagree"):
        load_calibration(mismatch)
    plain = tmp_path / "plain"
    plain.mkdir()
    atomic_write_json(plain / "calibration.json", artifact)
    write_manifest(plain, ["calibration.json"], {"kind": "jev-model-components"})
    with pytest.raises(ValueError, match="Not a sealed"):
        load_calibration(plain)
    with pytest.raises(FileExistsError):
        write_calibration(directory, artifact, identity=identity)
    with pytest.raises(ValueError):
        write_calibration(tmp_path / "bad", artifact, identity={})


def test_cli_writes_the_sealed_artifact(tmp_path):
    rows, logits = [], []
    weight = 1.0 / (1.0 + math.exp(-2.0))
    distribution = {"kind": "distribution", "probabilities": {"0": weight, "1": 1.0 - weight}}
    for index in range(2):
        rows.append(calibration_row(index, "temperature_fit", target=distribution))
        logits.append([8.0, 0.0])
    for subset, base in (("policy_selection", 2), ("risk_verification", 4)):
        for index in range(2):
            rows.append(calibration_row(base + index, subset,
                                        target=hard("0" if index == 0 else "1")))
            logits.append([math.log(9.0), 0.0])
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps({"logits": values}) + "\n" for values in logits), encoding="utf-8",
    )
    output = tmp_path / "calibration"
    arguments = [
        "--rows", str(rows_path), "--predictions", str(predictions_path),
        "--output", str(output), "--target-risk", "0.5",
        "--checkpoint", "c" * 64, "--release", "r" * 64,
    ]
    assert calibrate.main(arguments) == 0
    artifact = load_calibration(output)
    assert artifact["identity"]["checkpoint"] == "c" * 64
    assert artifact["identity"]["manifest"] == hashlib.sha256(canonical_json(
        {"checkpoint": "c" * 64, "release": "r" * 64},
    ).encode("utf-8")).hexdigest()
    short = tmp_path / "short.jsonl"
    short.write_text(
        "".join(json.dumps({"logits": values}) + "\n" for values in logits[:3]),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="prediction records"):
        calibrate.main([
            "--rows", str(rows_path), "--predictions", str(short),
            "--output", str(tmp_path / "other"), "--target-risk", "0.5",
            "--checkpoint", "c" * 64, "--release", "r" * 64,
        ])
