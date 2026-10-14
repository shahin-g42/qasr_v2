"""Strict JSON, immutable local artifacts and content identities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from jev_contracts import (
    ValidationError,
    artifact_path,
    atomic_write_json,
    canonical_json,
    sha256_file,
    validate_example,
    verify_manifest,
    write_manifest,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def fields(value: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    require(type(value) is dict, "expected an object")
    require(not (required - value.keys()), f"missing fields: {sorted(required - value.keys())}")
    require(not (value.keys() - required - optional), "unknown object fields")
    canonical_json(value)
    return value


def text(value: Any) -> str:
    require(type(value) is str and bool(value.strip()), "expected nonempty text")
    return value


def boolean(value: Any) -> bool:
    require(type(value) is bool, "expected a literal boolean")
    return value


def integer(value: Any, minimum: int = 0, maximum: int = 10**9) -> int:
    require(type(value) is int and minimum <= value <= maximum, "integer outside bounds")
    return value


def number(value: Any, minimum: float = 0, maximum: float = 1) -> float:
    require(type(value) in (int, float), "expected a number, not bool")
    require(math.isfinite(value) and minimum <= value <= maximum, "number outside finite bounds")
    return float(value)


def strings(value: Any) -> list[str]:
    require(type(value) is list, "expected an array")
    for item in value:
        text(item)
    require(len(set(value)) == len(value), "duplicate identifiers")
    return value


def version(value: Any) -> None:
    require(type(value) is int and value == 1, "expected schema_version 1")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def parse_json(data: str | bytes) -> Any:
    def reject(value: str) -> None:
        raise ValidationError("non-finite JSON constant")

    try:
        result = json.loads(data, object_pairs_hook=_object, parse_constant=reject)
        canonical_json(result)
        return result
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid JSON") from exc


def read_bytes(path: str | Path) -> bytes:
    path = Path(path).absolute()
    expected = sha256_file(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "file changed during read")
    require(sha256_file(path) == expected, "file changed during read")
    return data


def read_json(path: str | Path) -> Any:
    return parse_json(read_bytes(path))


def read_verified_json(root: str | Path, name: str) -> Any:
    manifest = verify_manifest(root)
    require(name in manifest["files"], "artifact member not in manifest")
    data = read_bytes(artifact_path(root, name))
    require(hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"],
            "artifact member changed")
    return parse_json(data)


def read_config(path: str | Path) -> dict:
    if Path(path).suffix == ".json":
        value = read_json(path)
    else:
        import yaml

        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader: Any, node: Any) -> dict:
            return _object([(loader.construct_object(k), loader.construct_object(v))
                            for k, v in node.value])

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        value = yaml.load(read_bytes(path), Loader=UniqueLoader)
    require(type(value) is dict, "configuration must be an object")
    canonical_json(value)
    return value


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def parse_jsonl(data: bytes) -> list[dict]:
    rows = [parse_json(line) for line in data.splitlines() if line.strip()]
    require(all(type(row) is dict for row in rows), "JSONL rows must be objects")
    return rows


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        manifest = verify_manifest(path)
        meta = manifest["metadata"]
        require(meta.get("kind") == "data_stage", "expected a data-stage artifact")
        name = meta.get("rows")
        require(name in manifest["files"], "rows missing from manifest")
        data = read_bytes(artifact_path(path, name))
        require(hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"],
                "stage checksum mismatch")
        rows = parse_jsonl(data)
        require(dataset_identity(rows) == meta.get("dataset"), "stage identity mismatch")
        return rows
    rows = parse_jsonl(read_bytes(path))
    dataset_identity(rows)
    return rows


def dataset_identity(rows: list[dict]) -> dict:
    hashes = {}
    for row in rows:
        row = validate_example(row)
        require(row["id"] not in hashes, "duplicate example ID")
        hashes[row["id"]] = digest(row)
    return {"sha256": digest(hashes), "rows": hashes, "count": len(hashes)}


def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path).absolute()
    artifact_path(path.parent, path.name)
    fd, temporary = tempfile.mkstemp(prefix=".jev-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.unlink(temporary)


def publish_rows(root: str | Path, rows: list[dict], stage: str, *,
                 artifacts: dict[str, Any] | None = None) -> dict:
    identity = dataset_identity(rows)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    name = f"{identity['sha256']}.jsonl"
    atomic_bytes(root / name, jsonl_bytes(rows))
    files = [name]
    for filename, value in (artifacts or {}).items():
        atomic_write_json(artifact_path(root, filename), value)
        files.append(filename)
    return write_manifest(root, files, {
        "kind": "data_stage", "stage": text(stage), "rows": name, "dataset": identity,
    })


def guard_test(rows: list[dict], *, allow_test: bool = False, purpose: str = "training") -> None:
    require(purpose in {"training", "evaluation"}, "invalid purpose")
    if any(row["split"] == "test" for row in rows):
        require(allow_test is True and purpose == "evaluation",
                "sealed test requires explicit allow_test and evaluation purpose")
