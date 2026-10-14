"""Provenance structure, not a substitute for rights or human-audit approval.

Origin and primary_route are nonempty registry-defined strings, not a closed
ontology. Optional source_revision, content_sha256, original_language and the
four independent origin flags have explicit types. Further details (teachers,
reviewers, prompt hashes, translations, dispositions, etc.) belong in metadata:
namespace -> {schema_version: positive integer, ...strict finite JSON...}.
Unknown sibling fields fail, so accidental hints cannot migrate into model input.
"""

from __future__ import annotations

import re

from ._validation import ValidationError, json_copy, metadata, object_fields, string_list, text

QUALITY_TIERS = frozenset({"gold", "silver", "bronze"})
TRAINING_TIERS = frozenset({"gold", "silver"})
CALIBRATION_SUBSETS = frozenset({"temperature_fit", "policy_selection", "risk_verification"})

_REQUIRED = {
    "origin", "primary_route", "license_ref", "source_ids", "derivation_ids", "evidence_ref"
}
_TEXT_FIELDS = {"origin", "primary_route", "license_ref", "evidence_ref"}
_OPTIONAL_TEXT = {"source_revision", "original_language"}
_FLAGS = {"native_origin", "human_authored", "localized", "synthetic"}


def validate_provenance(value: dict) -> dict:
    """Return a detached validated copy, without claiming legal/quality approval.

    source_ids and derivation_ids are arrays of unique nonempty opaque strings;
    empty arrays are legal (a pipeline must enforce its own lineage requirements).
    References and IDs are never implicitly interpreted as filesystem paths.
    The optional metadata.calibration object has exactly schema_version=1 and
    subset in temperature_fit, policy_selection, risk_verification. Full example
    validation additionally requires this object iff the split is calibration.
    """
    result = json_copy(value, "provenance")
    object_fields(
        result, "provenance", _REQUIRED,
        _OPTIONAL_TEXT | _FLAGS | {"content_sha256", "metadata"},
    )
    for field in _TEXT_FIELDS | (_OPTIONAL_TEXT & result.keys()):
        text(result[field], f"provenance.{field}")
    for field in ("source_ids", "derivation_ids"):
        string_list(result[field], f"provenance.{field}")
    for field in _FLAGS & result.keys():
        if type(result[field]) is not bool:
            raise ValidationError(f"provenance.{field}: expected a literal boolean")
    if "content_sha256" in result:
        digest = text(result["content_sha256"], "provenance.content_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValidationError("provenance.content_sha256: expected lowercase SHA-256 hex")
    if "metadata" in result:
        metadata(result["metadata"], "provenance.metadata")
        calibration = result["metadata"].get("calibration")
        if calibration is not None:
            object_fields(
                calibration, "provenance.metadata.calibration", {"schema_version", "subset"}
            )
            if type(calibration["schema_version"]) is not int or calibration["schema_version"] != 1:
                raise ValidationError("provenance.metadata.calibration.schema_version: expected 1")
            subset = text(calibration["subset"], "provenance.metadata.calibration.subset")
            if subset not in CALIBRATION_SUBSETS:
                raise ValidationError("provenance.metadata.calibration.subset: invalid subset")
    return result


def validate_quality(value: dict) -> dict:
    """Validate tier/audit_ref and optional versioned metadata, returning a copy.

    Bronze is structurally valid for staging/quarantine. Training consumers must
    separately gate on TRAINING_TIERS, release approval, rights, and audit evidence;
    a claimed gold/silver tier alone does not establish any of those facts.
    """
    result = json_copy(value, "quality")
    object_fields(result, "quality", {"tier", "audit_ref"}, {"metadata"})
    tier = text(result["tier"], "quality.tier")
    if tier not in QUALITY_TIERS:
        raise ValidationError(f"quality.tier: expected one of {sorted(QUALITY_TIERS)!r}")
    text(result["audit_ref"], "quality.audit_ref")
    if "metadata" in result:
        metadata(result["metadata"], "quality.metadata")
    return result
