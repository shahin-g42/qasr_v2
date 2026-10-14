"""Local JSONL ingestion only; remote candidates remain explicit pending records."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .dedup import validate_splits
from .io import dataset_identity, guard_test, parse_jsonl, publish_rows, read_bytes, require
from .registry import Registry


def ingest(path: str | Path, registry: Registry, source_id: str, *,
           output: str | Path | None = None, allow_test: bool = False,
           purpose: str = "training") -> list[dict]:
    source = registry.source(source_id)
    source.allow(purpose)
    raw = read_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == source.sha256,
            "source file checksum differs from pin")
    rows = parse_jsonl(raw)
    dataset_identity(rows)
    guard_test(rows, allow_test=allow_test, purpose=purpose)
    for row in rows:
        require(source_id in row["provenance"]["source_ids"], "ingest source identity mismatch")
        require(row["quality"]["tier"] == "bronze",
                "ingestion cannot import self-certified quality")
        metadata = row["provenance"].setdefault("metadata", {})
        pins = registry.pins(row["provenance"]["source_ids"])
        if "source_pins" in metadata:
            require(metadata["source_pins"] == pins, "imported source pins conflict with registry")
        metadata["source_pins"] = pins
        metadata["ingest"] = {"schema_version": 1, "raw_sha256": source.sha256,
                              "source_revision": source.revision, "original_split": row["split"]}
        registry.validate_row(row)
    validate_splits(rows)
    if output is not None:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        from .io import atomic_bytes

        atomic_bytes(root / "source.raw.jsonl", raw)
        child = publish_rows(root / "validated", rows, "ingested")
        from jev_contracts import write_manifest

        write_manifest(root, ["source.raw.jsonl", "validated/manifest.json",
                              *["validated/" + name for name in child["files"]]],
                       {"kind": "local_ingest", "raw_sha256": source.sha256,
                        "validated": "validated", "source_id": source_id})
    return rows


def pending_adapter(source_id: str, revision: str, url: str) -> dict:
    from .io import text

    return {"schema_version": 1, "source_id": text(source_id), "revision": text(revision),
            "url": text(url), "status": "pending", "downloaded": False,
            "reason": "Provide a locally obtained pinned JSONL and exact-release rights approval."}
