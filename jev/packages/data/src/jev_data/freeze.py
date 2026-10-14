"""Immutable schema-v2 dataset releases using the shared schema-v1 manifest contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jev_contracts import (
    SPLITS,
    artifact_path,
    atomic_write_json,
    safe_relative_path,
    verify_manifest,
    write_manifest,
)

from .curate import validate_audit
from .io import (
    atomic_bytes,
    boolean,
    dataset_identity,
    digest,
    fields,
    guard_test,
    jsonl_bytes,
    parse_json,
    parse_jsonl,
    read_bytes,
    require,
    strings,
    text,
    version,
)
from .registry import Registry


def validate_approval(approval: dict, rows: list[dict], registry: Registry, audit: dict) -> None:
    fields(approval, {"schema_version", "approved", "approver", "approval_ref", "dataset_hash",
                      "audit_hash", "registry_hash", "coverage"})
    version(approval["schema_version"])
    require(boolean(approval["approved"]) is True, "release approval not granted")
    text(approval["approver"])
    text(approval["approval_ref"])
    require(approval["dataset_hash"] == dataset_identity(rows)["sha256"],
            "approval dataset mismatch")
    require(approval["audit_hash"] == digest(audit), "approval audit mismatch")
    require(approval["registry_hash"] == digest(registry.to_dict()),
            "approval rights ledger mismatch")
    coverage = approval["coverage"]
    fields(coverage, {"planned_ids", "accepted_ids", "quarantined", "excluded", "pending_ids",
                      "accepted_subset_approved"})
    planned = set(strings(coverage["planned_ids"]))
    accepted = set(strings(coverage["accepted_ids"]))
    pending = set(strings(coverage["pending_ids"]))
    boolean(coverage["accepted_subset_approved"])
    require(not pending, "pending work blocks completion")
    require(accepted == {row["id"] for row in rows}, "coverage differs from actual accepted rows")
    accounted = set(accepted)
    for disposition in ("quarantined", "excluded"):
        require(type(coverage[disposition]) is list, "coverage dispositions must be arrays")
        for item in coverage[disposition]:
            fields(item, {"id", "reason"})
            key = text(item["id"])
            text(item["reason"])
            require(key not in accounted, "duplicate coverage disposition")
            accounted.add(key)
    require(accounted == planned, "planned items are not fully reconciled")
    if accepted != planned:
        require(coverage["accepted_subset_approved"] is True,
                "accepted subset needs explicit approval")


def freeze(root: str | Path, rows: list[dict], registry: Registry, audit: dict, *,
           approval: dict, allow_test: bool = False, purpose: str = "training") -> dict:
    """Publish a new root only after rights, human audit and exact-subset sign-off pass."""
    require(isinstance(registry, Registry), "freeze requires an approved rights Registry")
    identity = dataset_identity(rows)
    require(identity["count"] > 0, "empty release is not approvable")
    guard_test(rows, allow_test=allow_test, purpose=purpose)
    validate_audit(rows, registry, audit)
    validate_approval(approval, rows, registry, audit)
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    (root / "shards").mkdir(mode=0o700)
    files, splits = [], {}
    for split in sorted(SPLITS):
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["id"])
        if not selected:
            continue
        data = jsonl_bytes(selected)
        shard = "shards/" + hashlib.sha256(data).hexdigest() + ".jsonl"
        atomic_bytes(root / shard, data)
        filename = split + ".jsonl"
        atomic_bytes(root / filename, data)
        files.extend([shard, filename])
        splits[split] = filename
    for name, value in {
        "audit.json": audit, "registry.json": registry.to_dict(), "approval.json": approval,
        "families.json": audit["curation"]["split_index"],
    }.items():
        atomic_write_json(root / name, value)
        files.append(name)
    return write_manifest(root, files, {"kind": "dataset_release", "approved": True,
                                       "schema_version": 2, "splits": splits,
                                       "audit": "audit.json", "registry": "registry.json"})


def _read_member(root: Path, manifest: dict, name: str) -> bytes:
    safe_relative_path(name)
    require(name in manifest["files"], "required release member absent from manifest")
    data = read_bytes(artifact_path(root, name))
    info = manifest["files"][name]
    require(hashlib.sha256(data).hexdigest() == info["sha256"] and len(data) == info["bytes"],
            "release member changed during read")
    return data


def load_release(root: str | Path, split: str, *, allow_test: bool = False,
                 purpose: str = "training") -> list[dict]:
    """Verify the entire release, including audit/data binding, before returning any rows.

    Test access additionally requires allow_test=True and purpose='evaluation'.
    The trusted root/sign-off must be supplied by the caller; checksums are not signatures.
    """
    require(type(split) is str and split in SPLITS, "invalid release split")
    if split == "test":
        require(allow_test is True and purpose == "evaluation", "sealed test is evaluation-only")
    root = Path(root).absolute()
    manifest = verify_manifest(root)
    meta = manifest["metadata"]
    fields(meta, {"kind", "approved", "schema_version", "splits", "audit", "registry"})
    require(meta["kind"] == "dataset_release" and meta["approved"] is True,
            "dataset release is not approved")
    require(type(meta["schema_version"]) is int and meta["schema_version"] == 2,
            "release example schema must be 2")
    require(type(meta["splits"]) is dict and bool(meta["splits"]),
            "release split map is empty/invalid")
    require(set(meta["splits"]) <= SPLITS and split in meta["splits"],
            "requested split not in release")
    all_rows, selected = [], []
    for key, name in sorted(meta["splits"].items()):
        require(name == key + ".jsonl", "split must use its canonical safe JSONL path")
        rows = parse_jsonl(_read_member(root, manifest, name))
        require(bool(rows), "manifest contains an empty split")
        require(all(row.get("split") == key for row in rows),
                "row split differs from manifest split")
        all_rows.extend(rows)
        if key == split:
            selected = rows
    dataset_identity(all_rows)
    require(meta["audit"] == "audit.json" and meta["registry"] == "registry.json",
            "noncanonical audit/registry release paths")
    registry = Registry.from_dict(parse_json(_read_member(root, manifest, meta["registry"])))
    audit = parse_json(_read_member(root, manifest, meta["audit"]))
    approval = parse_json(_read_member(root, manifest, "approval.json"))
    validate_audit(all_rows, registry, audit)
    validate_approval(approval, all_rows, registry, audit)
    families = parse_json(_read_member(root, manifest, "families.json"))
    require(families == audit["curation"]["split_index"], "family index differs from audited data")
    require(verify_manifest(root) == manifest, "release manifest changed during load")
    return selected
