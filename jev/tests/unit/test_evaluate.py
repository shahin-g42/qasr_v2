from __future__ import annotations

import json
import math

import pytest

from jev import evaluate
from jev.evaluate import (
    evaluate_predictions,
    evaluate_records,
    example_metrics,
    paired_family_bootstrap,
    probabilities,
    read_jsonl,
)

from .test_processing import make_row


def row(index, *, kind="choice", count=2, family="f", target=None):
    value = make_row(kind, count)
    value["id"] = f"row-{index:03d}"
    value["family_id"] = family
    if target is not None:
        value["target"] = target
    return value


def records(rows, logits):
    return [
        {"id": r["id"], "option_ids": [option["id"] for option in r["question"]["options"]],
         "logits": values}
        for r, values in zip(rows, logits, strict=True)
    ]


def write_jsonl(path, values):
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_probabilities_is_exact_softmax_with_validation():
    assert probabilities([0.0, 0.0], temperature=1.0) == [0.5, 0.5]
    values = probabilities([2.0, 0.0], temperature=2.0)
    expected = [math.exp(1.0), 1.0]
    assert values[0] == pytest.approx(expected[0] / (expected[0] + 1))
    assert values[1] == pytest.approx(1 / (expected[0] + 1))
    assert sum(values) == pytest.approx(1.0)
    sharp = probabilities([100.0, -100.0], temperature=1.0)
    assert sharp == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError, match="not be empty"):
        probabilities([], temperature=1.0)
    for bad_temperature in (0.0, -1.0, float("inf"), True):
        with pytest.raises(ValueError, match="temperature"):
            probabilities([0.0, 0.0], temperature=bad_temperature)
    for bad_logits in ([0.0, float("nan")], [0.0, True], "01"):
        with pytest.raises(ValueError):
            probabilities(bad_logits, temperature=1.0)


def test_example_metrics_for_hard_choice_rows():
    entry = example_metrics(row(0), [0.0, 0.0])
    assert entry["id"] == "row-000" and entry["type"] == "choice" and entry["count"] == 2
    assert entry["nll"] == pytest.approx(math.log(2))
    assert entry["brier"] == pytest.approx(0.5)
    assert entry["accuracy"] == 1.0 and entry["confidence"] == pytest.approx(0.5)
    assert entry["binary_brier"] is None and entry["ordinal_mae"] is None
    assert entry["expected_prediction"] == pytest.approx(0.5)
    wrong = example_metrics(
        row(1, target={"kind": "hard", "option_id": "1"}), [0.0, 0.0])
    assert wrong["accuracy"] == 0.0
    assert wrong["nll"] == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="candidates"):
        example_metrics(row(2), [0.0])
    with pytest.raises(ValueError):
        example_metrics(row(3), [0.0, float("inf")])


def test_example_metrics_noul_and_score_extras():
    noul = example_metrics(row(0, kind="noul"), [1.0, 0.0])
    tail = 1.0 / (1.0 + math.exp(1.0))
    assert noul["binary_brier"] == pytest.approx(tail ** 2)
    assert noul["ordinal_mae"] is None and noul["rps"] is None
    score = example_metrics(
        row(1, kind="score", count=3, target={"kind": "hard", "option_id": "0"}),
        [0.0, 0.0, 0.0],
    )
    assert score["ordinal_mae"] == pytest.approx(1.0)
    assert score["rps"] == pytest.approx(5 / 9)
    distribution = example_metrics(
        row(2, target={"kind": "distribution", "probabilities": {"0": 0.25, "1": 0.75}}),
        [0.0, 0.0],
    )
    assert distribution["nll"] == pytest.approx(math.log(2))
    assert distribution["accuracy"] == 0.0


def test_aggregate_ece_bins_confidence():
    gap = math.log(9.0)
    rows = [row(0), row(1), row(2), row(3)]
    rows[1]["target"] = {"kind": "hard", "option_id": "1"}
    logits = [[gap, 0.0], [gap, 0.0], [0.0, 0.0], [0.0, 0.0]]
    report = evaluate_predictions(rows, logits)
    overall = report["overall"]
    assert overall["count"] == 4
    assert overall["accuracy"] == pytest.approx(3 / 4)
    # Bin 9 (confidence 0.9): one correct, one wrong. Bin 5 (0.5): both correct.
    assert overall["ece"] == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.5))
    with pytest.raises(ValueError, match="empty"):
        evaluate._aggregate([])


def test_evaluate_predictions_covers_every_preregistered_group():
    rows = [row(0, family="a"), row(1, family="a"), row(2, family="b")]
    logits = [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]
    report = evaluate_predictions(rows, logits, temperature=1.5)
    assert report["schema_version"] == 1
    assert report["count"] == 3 and report["temperature"] == 1.5
    assert report["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert set(report["groups"]) == {"family", "task", "type", "language", "variety", "origin"}
    assert set(report["groups"]["family"]) == {"a", "b"}
    assert report["groups"]["family"]["a"]["count"] == 2
    assert report["groups"]["language"] == {"ar": report["overall"]}
    assert set(report["groups"]["origin"]) == {
        "native_origin=none,human_authored=none,localized=none,synthetic=none",
    }
    with pytest.raises(ValueError, match="rows but"):
        evaluate_predictions(rows, logits[:1])


def test_evaluate_records_enforces_exact_ordered_coverage():
    rows = [row(0, family="a"), row(1, family="a")]
    logits = [[1.0, 0.0], [0.0, 1.0]]
    report = evaluate_records(rows, records(rows, logits))
    assert report == evaluate_predictions(rows, logits)
    wrong_order = [
        {"id": "row-000", "option_ids": ["1", "0"], "logits": [1.0, 0.0]},
        {"id": "row-001", "option_ids": ["0", "1"], "logits": [0.0, 1.0]},
    ]
    with pytest.raises(ValueError, match="order"):
        evaluate_records(rows, wrong_order)
    with pytest.raises(ValueError, match="unique"):
        evaluate_records(rows, records(rows, logits) + [records(rows, logits)[0]])
    with pytest.raises(ValueError, match="unknown row ID"):
        evaluate_records(rows, records(rows, logits) + [
            {"id": "ghost", "option_ids": ["0", "1"], "logits": [0.0, 0.0]},
        ])
    with pytest.raises(ValueError, match="without predictions"):
        evaluate_records(rows, records(rows, logits)[:1])
    with pytest.raises(ValueError, match="Duplicate row ID"):
        evaluate_records(rows + [dict(rows[0])], records(rows, logits))
    with pytest.raises(ValueError, match="object"):
        evaluate_records(rows, ["not-an-object"])


def test_paired_family_bootstrap_resamples_families_deterministically():
    rows = [row(index, family=f"f{index % 2}") for index in range(8)]
    logits_a = [[2.0, 0.0]] * 8
    logits_b = [[0.0, 2.0]] * 8
    result = paired_family_bootstrap(rows, logits_a, logits_b, metric="accuracy",
                                     seed=3, replicates=200)
    assert result["schema_version"] == 1 and result["metric"] == "accuracy"
    assert result["difference"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(result["ci_high"])
    assert result["families"] == 2 and result["count"] == 8
    assert paired_family_bootstrap(rows, logits_a, logits_b, metric="accuracy",
                                   seed=3, replicates=200) == result
    identical = paired_family_bootstrap(rows, logits_a, logits_a, metric="nll", seed=1)
    assert identical["difference"] == pytest.approx(0.0)
    assert identical["ci_low"] <= 0.0 <= identical["ci_high"]
    with pytest.raises(ValueError, match="metric"):
        paired_family_bootstrap(rows, logits_a, logits_b, metric="brier_x")
    with pytest.raises(ValueError, match="identical row sequence"):
        paired_family_bootstrap(rows, logits_a, logits_b[:1])
    with pytest.raises(ValueError, match="replicates"):
        paired_family_bootstrap(rows, logits_a, logits_b, replicates=0)


def test_read_jsonl_is_strict(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Blank"):
        read_jsonl(path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="objects"):
        read_jsonl(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No records"):
        read_jsonl(path)


def test_cli_scores_and_seals_a_report(tmp_path):
    rows = [row(index, family=f"f{index}") for index in range(3)]
    logits = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    rows_path = tmp_path / "rows.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(rows_path, rows)
    write_jsonl(predictions_path, records(rows, logits))
    output = tmp_path / "report.json"
    arguments = [
        "--rows", str(rows_path), "--predictions", str(predictions_path),
        "--output", str(output), "--temperature", "1.0",
        "--compare", str(predictions_path), "--bootstrap-replicates", "50",
    ]
    assert evaluate.main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["count"] == 3 and report["groups"]["family"]["f0"]["count"] == 1
    assert report["bootstrap"]["replicates"] == 50
    with pytest.raises(FileExistsError):
        evaluate.main(arguments)
