"""Batch helpers reuse the exact per-row contracts; fixtures are test evidence only."""

from __future__ import annotations

import copy
import json

import pytest

from jev_contracts import ValidationError
from jev_data.batch import collect_rows, oracle_batch
from jev_data.io import publish_rows, read_rows
from jev_data.registry import Registry


def registry():
    return Registry.from_dict({"schema_version": 1, "sources": [{
        "id": "commissioned", "status": "approved", "url": "internal://fixture-source",
        "revision": "fixture-v1", "sha256": "a" * 64, "license_ref": "fixture-rights",
        "attribution": "Test authors", "intended_uses": [
            "training", "evaluation", "redistribution", "derivatives"],
        "training": True, "redistribution": True, "derivatives": True, "consent": True,
        "approval_ref": "test-only-approval", "data_class": "commissioned",
        "original_language": "ar", "original_split": "unassigned",
        "collected_at": "2026-01-01", "related_sources": [],
    }]})


def world(index=1, operator=">"):
    return {"schema_version": 1, "kind": "comparison", "family_id": f"family-{index}",
            "template_id": f"independent-fixture-policy-{index}", "entity": f"طلب رقم {index}",
            "quantity": {"value": "١٠٠", "unit": "SAR"},
            "threshold": {"value": "100", "unit": "SAR"}, "operator": operator}


def test_oracle_batch_publishes_one_stage(tmp_path):
    rows = oracle_batch([world(1), world(2, "<")], prefix="oracle-", source_id="commissioned",
                        registry=registry(), output=tmp_path / "stage")
    assert [row["id"] for row in rows] == ["oracle-000000", "oracle-000001"]
    assert all(row["quality"]["tier"] == "bronze" for row in rows)
    assert read_rows(tmp_path / "stage") == rows


def test_oracle_batch_rejects_invalid_world_without_partial_output(tmp_path):
    with pytest.raises(ValidationError):
        oracle_batch([world(1), {**world(2), "operator": "~"}], prefix="oracle-",
                     source_id="commissioned", registry=registry(), output=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_collect_rows_filters_tiers_and_splits(tmp_path):
    bronze = oracle_batch([world(1)], prefix="oracle-", source_id="commissioned",
                          registry=registry(), output=tmp_path / "bronze")
    gold = copy.deepcopy(bronze)
    gold[0]["quality"] = {"tier": "gold", "audit_ref": "fixture://gold"}
    publish_rows(tmp_path / "gold", gold, "human_reviewed")
    merged = collect_rows([str(tmp_path / "bronze"), str(tmp_path / "gold")],
                          output=tmp_path / "merged", tiers=["gold"], splits=["train"])
    assert [row["id"] for row in merged] == ["oracle-000000"]
    with pytest.raises(ValidationError, match="empty"):
        collect_rows([str(tmp_path / "gold")], output=tmp_path / "none", tiers=["bronze"])


def test_collect_rows_exclude_ids(tmp_path):
    oracle_batch([world(1), world(2, "<")], prefix="oracle-", source_id="commissioned",
                 registry=registry(), output=tmp_path / "bronze")
    merged = collect_rows([str(tmp_path / "bronze")], output=tmp_path / "merged",
                          exclude_ids={"oracle-000001"})
    assert [row["id"] for row in merged] == ["oracle-000000"]
    selection = json.loads((tmp_path / "merged" / "selection.json").read_text(encoding="utf-8"))
    assert selection["dropped_ids"] == ["oracle-000001"]
    assert selection["excluded_ids"] == ["oracle-000001"]
    assert selection["read"] == 2 and selection["selected"] == 1
    with pytest.raises(ValidationError, match="empty"):
        collect_rows([str(tmp_path / "bronze")], output=tmp_path / "none",
                     exclude_ids={"oracle-000000", "oracle-000001"})
