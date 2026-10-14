"""Batch authoring and stage selection helpers; every row passes the same gates."""

from __future__ import annotations

from pathlib import Path

from .generate import oracle_candidate
from .io import dataset_identity, publish_rows, read_rows, require, text
from .registry import Registry


def oracle_batch(worlds: list[dict], *, prefix: str, source_id: str, registry: Registry,
                 split: str = "train", calibration_subset: str | None = None,
                 allow_test: bool = False, purpose: str = "training",
                 output: str | Path | None = None) -> list[dict]:
    """Author one oracle row per world; identifiers are prefix plus a zero-padded index."""
    require(type(worlds) is list and bool(worlds), "worlds must be a nonempty array")
    text(prefix)
    rows = [oracle_candidate(world, identifier=f"{prefix}{index:06d}", source_id=source_id,
                             registry=registry, split=split,
                             calibration_subset=calibration_subset,
                             allow_test=allow_test, purpose=purpose)
            for index, world in enumerate(worlds)]
    dataset_identity(rows)
    if output is not None:
        publish_rows(output, rows, "authored", artifacts={"worlds.json": worlds})
    return rows


def collect_rows(paths: list[str], *, output: str | Path, tiers: list[str] | None = None,
                 splits: list[str] | None = None, exclude_ids: set[str] | None = None,
                 stage: str = "collected") -> list[dict]:
    """Merge stage artifacts or JSONL files into one new stage, with optional filters."""
    require(type(paths) is list and bool(paths), "input stage paths required")
    incoming = [row for path in paths for row in read_rows(path)]
    selected = [row for row in incoming
                if (tiers is None or row["quality"]["tier"] in tiers)
                and (splits is None or row["split"] in splits)
                and (exclude_ids is None or row["id"] not in exclude_ids)]
    require(bool(selected), "selection is empty; check filters against the inputs")
    dropped_ids = sorted({row["id"] for row in incoming} - {row["id"] for row in selected})
    dataset_identity(selected)
    publish_rows(output, selected, stage, artifacts={
        "selection.json": {"schema_version": 1, "inputs": [str(path) for path in paths],
                           "tiers": tiers or [], "splits": splits or [],
                           "excluded_ids": sorted(exclude_ids) if exclude_ids else [],
                           "dropped_ids": dropped_ids,
                           "read": len(incoming), "selected": len(selected)}})
    return selected
