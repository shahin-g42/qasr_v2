"""CPU schema, metadata isolation, and standalone package regression tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from jev_contracts import (
    CALIBRATION_SUBSETS,
    TRAINING_TIERS,
    ValidationError,
    options_for,
    renderable_input,
    target_distribution,
    validate_example,
    validate_inference,
    validate_provenance,
    validate_quality,
)


def make_row(kind="choice", count=3):
    ids = ["false", "true"] if kind == "noul" else [str(i) for i in range(count)]
    return {
        "schema_version": 2,
        "id": "مثال-١",
        "family_id": "policy-strict-boundary-v1",
        "split": "train",
        "task": "numeric_policy",
        "language": "ar",
        "variety": "MSA",
        "state": "لَيْسَ الطلب ١٠٠ مؤهلاً؛ shipping requires more than 100.",
        "question": {
            "type": kind,
            "instructions": "هل يستحق الطلب الشحن المجاني وفق السياسة المذكورة فقط؟",
            "options": [{"id": key, "description": f"وصف المستوى {key}"} for key in ids],
        },
        "target": {"kind": "hard", "option_id": ids[0]},
        "provenance": {
            "origin": "native_arabic_authored",
            "primary_route": "oracle",
            "license_ref": "commissioned-internal-v1",
            "source_ids": ["policy-template-strict-boundary-v1"],
            "derivation_ids": [],
            "evidence_ref": "audit://example-1",
        },
        "quality": {"tier": "gold", "audit_ref": "audit://example-1"},
    }


@pytest.fixture
def row():
    return make_row()


@pytest.mark.parametrize(
    "kind,count", [("choice", 1), ("choice", 255), ("noul", 2), ("score", 2), ("score", 10)]
)
def test_valid_limits_and_all_hard_targets(kind, count):
    value = make_row(kind, count)
    assert validate_example(value) == value
    for i, option in enumerate(value["question"]["options"]):
        value["target"]["option_id"] = option["id"]
        assert target_distribution(value) == [float(j == i) for j in range(count)]


@pytest.mark.parametrize(
    "kind,count", [("choice", 0), ("choice", 256), ("score", 1), ("score", 11)]
)
def test_invalid_option_counts(kind, count):
    value = make_row(kind, max(1, count))
    if count == 0:
        value["question"]["options"] = []
    with pytest.raises(ValidationError, match="question.options"):
        validate_example(value)


def test_deep_copy_and_arabic_fidelity(row):
    original = copy.deepcopy(row)
    result = validate_example(row)
    assert result == original
    result["question"]["options"][0]["description"] = "changed"
    result["provenance"]["source_ids"].append("changed")
    assert row == original
    options_for(row["question"])[0]["id"] = "changed"
    assert row == original


@pytest.mark.parametrize("field", list(make_row()))
def test_all_required_fields(row, field):
    del row[field]
    with pytest.raises(ValidationError, match="missing required"):
        validate_example(row)


@pytest.mark.parametrize("field", ["id", "family_id", "task", "language", "variety"])
@pytest.mark.parametrize("value", [None, False, 0, "", " \t\n"])
def test_nonempty_string_fields(row, field, value):
    row[field] = value
    with pytest.raises(ValidationError, match=field):
        validate_example(row)


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("schema_version",), True, "schema_version"),
        (("schema_version",), 2.0, "schema_version"),
        (("schema_version",), 1, "schema_version"),
        (("split",), "validation", "split"),
        (("question", "type"), "binary", "question.type"),
        (("question", "instructions"), "  ", "instructions"),
        (("question", "options"), {}, "ordered array"),
        (("question", "options", 0, "id"), False, "id"),
        (("question", "options", 0, "id"), 0, "id"),
        (("question", "options", 0, "description"), "", "description"),
        (("question", "options", 1, "id"), "0", "duplicate option ID"),
        (("question", "options", 1, "description"), "وصف المستوى 0", "indistinguishable"),
        (("target",), 0.5, "target"),
        (("target", "option_id"), True, "option_id"),
        (("target", "option_id"), "missing", "supplied option"),
        (("target", "kind"), "scalar", "target.kind"),
        (("provenance", "license_ref"), "", "license_ref"),
        (("provenance", "source_ids"), ["a", "a"], "duplicate identifier"),
        (("provenance", "derivation_ids"), "a", "derivation_ids"),
        (("quality", "tier"), "accepted", "quality.tier"),
        (("quality", "audit_ref"), None, "audit_ref"),
    ],
)
def test_invalid_fields(row, path, value, match):
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        validate_example(row)


@pytest.mark.parametrize(
    "path",
    [(), ("question",), ("question", "options", 0), ("target",), ("provenance",), ("quality",)],
)
def test_unknown_fields_rejected_at_each_boundary(row, path):
    target = row
    for key in path:
        target = target[key]
    target["answer_hint"] = "secret"
    with pytest.raises(ValidationError, match="unknown fields"):
        validate_example(row)


@pytest.mark.parametrize("descriptions", [(" Same  text ", "same\ttext"), ("café", "cafe\u0301")])
def test_duplicate_text_fingerprint(row, descriptions):
    for option, description in zip(row["question"]["options"], descriptions, strict=False):
        option["description"] = description
    with pytest.raises(ValidationError, match="indistinguishable"):
        options_for(row["question"])


@pytest.mark.parametrize(
    "ids", [["true", "false"], ["False", "True"], [False, True], ["0", "1"], ["false"]]
)
def test_noul_requires_ordered_literal_string_ids(ids):
    value = make_row("noul")
    value["question"]["options"] = [{"id": key, "description": f"Option {key}"} for key in ids]
    with pytest.raises(ValidationError, match="question.options"):
        validate_example(value)


@pytest.mark.parametrize(
    "ids", [[0, 1], [False, True], ["1", "0"], ["1", "2"], ["0", "2"], ["0", "1.0"]]
)
def test_score_rejects_coercion_gaps_and_raw_rating_ids(ids):
    value = make_row("score", 2)
    value["question"]["options"] = [{"id": key, "description": f"Level {key}"} for key in ids]
    with pytest.raises(ValidationError, match="question.options"):
        validate_example(value)


def test_distribution_alignment_and_no_silent_normalization(row):
    row["target"] = {"kind": "distribution", "probabilities": {"2": 0.5, "0": 0.2, "1": 0.3}}
    assert target_distribution(row) == [0.2, 0.3, 0.5]
    row["question"]["options"].reverse()
    assert target_distribution(row) == [0.5, 0.3, 0.2]
    row["target"]["probabilities"]["0"] += 5e-7
    assert target_distribution(row)[2] == 0.2000005
    row["target"]["probabilities"]["0"] += 2e-6
    with pytest.raises(ValidationError, match="sum"):
        target_distribution(row)


@pytest.mark.parametrize(
    "value",
    [True, False, "0", None, -0.1, 1.1, float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_probability_types_and_range(row, value):
    row["target"] = {"kind": "distribution", "probabilities": {"0": value, "1": 0.5, "2": 0.5}}
    with pytest.raises(ValidationError, match="probabilities"):
        target_distribution(row)


@pytest.mark.parametrize(
    "probabilities",
    [{"0": 1}, {"0": 1, "1": 0, "2": 0, "3": 0}, {"0": 0, "1": 0, "2": 0}, [1, 0, 0]],
)
def test_probability_coverage_and_total(row, probabilities):
    row["target"] = {"kind": "distribution", "probabilities": probabilities}
    with pytest.raises(ValidationError, match="probabilities"):
        target_distribution(row)


def test_noul_integer_probabilities_are_not_boolean_labels():
    value = make_row("noul")
    for target in ("false", "true"):
        value["target"] = {
            "kind": "distribution",
            "probabilities": {key: int(key == target) for key in ("true", "false")},
        }
        assert target_distribution(value) == [float(target == "false"), float(target == "true")]


@pytest.mark.parametrize(
    "state",
    [None, True, 1, 1.2, "", "  ", (1, 2), {1: "bad"}, {"a": float("nan")},
     {"a": object()}, {"a": "\ud800"}],
)
def test_invalid_state(row, state):
    row["state"] = state
    with pytest.raises(ValidationError, match="state"):
        validate_example(row)


def test_structured_state_copy_and_cycle_rejection(row):
    row["state"] = {"facts": [None, False, 12, 1.25, {"نفي": "لم"}], "empty": ""}
    assert validate_example(row) == row
    result = renderable_input(row)
    result["state"]["facts"].append(5)
    assert len(row["state"]["facts"]) == 5
    row["state"]["cycle"] = row["state"]
    with pytest.raises(ValidationError, match="cyclic or excessively nested"):
        validate_example(row)


@pytest.mark.parametrize("tier", ["gold", "silver", "bronze"])
def test_quality_schema_is_separate_from_training_acceptance(row, tier):
    row["quality"]["tier"] = tier
    assert validate_example(row) == row
    assert (tier in TRAINING_TIERS) == (tier != "bronze")


@pytest.mark.parametrize("subset", sorted(CALIBRATION_SUBSETS))
def test_calibration_subset_binding(row, subset):
    row["provenance"]["metadata"] = {"calibration": {"schema_version": 1, "subset": subset}}
    with pytest.raises(ValidationError, match="iff split"):
        validate_example(row)
    row["split"] = "calibration"
    assert validate_example(row) == row
    del row["provenance"]["metadata"]
    with pytest.raises(ValidationError, match="iff split"):
        validate_example(row)


@pytest.mark.parametrize(
    "metadata",
    [
        {"x": 1}, {"x": {}}, {"x": {"schema_version": True}}, {"x": {"schema_version": 0}},
        {"x": {"schema_version": 1, "a": float("inf")}},
        {"calibration": {"schema_version": 1, "subset": "all"}},
    ],
)
def test_named_versioned_metadata(row, metadata):
    row["provenance"]["metadata"] = metadata
    with pytest.raises(ValidationError, match="metadata"):
        validate_provenance(row["provenance"])


def test_provenance_known_metadata_and_literal_flags(row):
    provenance = row["provenance"]
    provenance.update(native_origin=True, human_authored=False, localized=False, synthetic=True,
                      content_sha256="a" * 64, source_revision="v1", original_language="ar")
    assert validate_provenance(provenance) == provenance
    provenance["synthetic"] = 1
    with pytest.raises(ValidationError, match="literal boolean"):
        validate_provenance(provenance)
    provenance["synthetic"] = True
    provenance["content_sha256"] = "A" * 64
    with pytest.raises(ValidationError, match="SHA-256"):
        validate_provenance(provenance)
    with pytest.raises(ValidationError, match="missing required"):
        validate_quality({"tier": "gold"})


def test_template_boundary_excludes_all_metadata_and_preserves_allowed_text(row):
    secret = "PRIVATE_EXPECTED_ANSWER_DO_NOT_RENDER"
    row["id"] = secret
    row["task"] = secret
    row["provenance"]["metadata"] = {"teacher": {"schema_version": 1, "explanation": secret}}
    row["quality"]["metadata"] = {"review": {"schema_version": 1, "judgment": secret}}
    row["images"] = ["sha256:" + "1" * 64]
    visible = renderable_input(row)
    assert set(visible) == {"state", "question", "images"}
    assert secret not in json.dumps(visible)
    assert visible["state"] == row["state"]
    assert validate_inference(visible) == visible
    visible["images"].clear()
    assert len(row["images"]) == 1
    request = {"state": row["state"], "question": row["question"], "schema_version": 2}
    assert "schema_version" not in renderable_input(request)
    request["target"] = row["target"]
    with pytest.raises(ValidationError):
        validate_inference(request)
    with pytest.raises(ValidationError):
        renderable_input(request)


@pytest.mark.parametrize(
    "images",
    [
        ["/etc/passwd"], ["https://example.com/image"], ["../a"], [{"sha256": "a" * 64}],
        ["sha256:" + "A" * 64], ["sha256:" + "1" * 64] * 2, "sha256:abcd",
    ],
)
def test_images_cannot_reference_paths_or_urls(row, images):
    row["images"] = images
    with pytest.raises(ValidationError, match="images"):
        validate_example(row)


def test_ids_are_opaque_not_paths(row):
    row["id"] = "opaque://collection/../row"
    assert validate_example(row)["id"] == row["id"]


def test_cpu_isolation_and_workspace_configuration():
    root = Path(__file__).resolve().parents[2]
    workspace = tomllib.loads((root / "pyproject.toml").read_text())
    package = tomllib.loads((root / "packages/contracts/pyproject.toml").read_text())
    assert workspace["tool"]["uv"]["workspace"]["members"] == [
        "packages/contracts", "packages/data", "packages/core"
    ]
    assert package["project"]["dependencies"] == []
    assert package["project"]["requires-python"] == ">=3.12"
    assert package["build-system"]["build-backend"] == "hatchling.build"
    assert workspace["tool"]["pytest"]["ini_options"]["consider_namespace_packages"] is True
    script = (
        f"import sys; sys.path.insert(0, {str(root / 'packages/contracts/src')!r}); "
        "import jev_contracts; "
        "assert not {'qasr', 'torch', 'transformers', 'numpy', 'pydantic'} & sys.modules.keys(); "
        "assert issubclass(jev_contracts.ValidationError, ValueError)"
    )
    subprocess.run([sys.executable, "-I", "-S", "-c", script], check=True, cwd=root)
