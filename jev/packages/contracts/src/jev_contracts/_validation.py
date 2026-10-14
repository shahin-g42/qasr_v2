"""Shared strict JSON checks; deliberately no coercion or third-party imports."""

from __future__ import annotations

import math
from typing import Any

MAX_JSON_DEPTH = 64


class ValidationError(ValueError):
    """A contract violation, with the offending field path in the message."""


def text(value: Any, path: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise ValidationError(f"{path}: expected a string")
    if nonempty and not value.strip():
        raise ValidationError(f"{path}: expected a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValidationError(f"{path}: unpaired Unicode surrogate is not valid UTF-8") from exc
    return value


def object_fields(
    value: Any, path: str, required: set[str], optional: set[str] | None = None
) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{path}: expected an object")
    if any(type(key) is not str for key in value):
        raise ValidationError(f"{path}: object keys must be strings")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing:
        raise ValidationError(f"{path}: missing required fields {sorted(missing)!r}")
    if unknown:
        raise ValidationError(f"{path}: unknown fields {sorted(unknown)!r}")
    return value


def json_copy(value: Any, path: str = "$", *, _depth: int = 0) -> Any:
    """Copy only actual JSON types; reject cycles/deep nesting and non-finite numbers.

    Boolean values are valid JSON here, but numeric schema fields check their
    exact type separately. Tuples, custom objects, non-string keys, and implicit
    conversions are never accepted. The maximum container nesting depth is 64.
    """
    if _depth > MAX_JSON_DEPTH:
        raise ValidationError(f"{path}: cyclic or excessively nested JSON (limit {MAX_JSON_DEPTH})")
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{path}: expected a finite JSON number")
        return value
    if type(value) is str:
        return text(value, path, nonempty=False)
    if type(value) is list:
        return [json_copy(item, f"{path}[{i}]", _depth=_depth + 1) for i, item in enumerate(value)]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            text(key, f"{path} key", nonempty=False)
            result[key] = json_copy(item, f"{path}.{key}", _depth=_depth + 1)
        return result
    raise ValidationError(f"{path}: expected a JSON value, got {type(value).__name__}")


def string_list(value: Any, path: str) -> None:
    if type(value) is not list:
        raise ValidationError(f"{path}: expected an array of unique nonempty strings")
    seen: set[str] = set()
    for i, item in enumerate(value):
        text(item, f"{path}[{i}]")
        if item in seen:
            raise ValidationError(f"{path}[{i}]: duplicate identifier {item!r}")
        seen.add(item)


def version(value: Any, expected: int, path: str) -> None:
    if type(value) is not int or value != expected:
        raise ValidationError(f"{path}: expected integer schema version {expected}")


def metadata(value: Any, path: str) -> None:
    """Extensions are named objects, each with its own positive schema_version.

    For example: {"annotation": {"schema_version": 1, "reviewer_ids": ["r1"]}}.
    Their other fields may contain any strict finite JSON, including raw audit
    evidence. These extensions are never made available to prompt rendering.
    """
    if type(value) is not dict:
        raise ValidationError(f"{path}: expected named, versioned metadata objects")
    for name, extension in value.items():
        text(name, f"{path} namespace")
        if type(extension) is not dict:
            raise ValidationError(f"{path}.{name}: expected an object with schema_version")
        number = extension.get("schema_version")
        if type(number) is not int or number < 1:
            raise ValidationError(f"{path}.{name}.schema_version: expected a positive integer")
