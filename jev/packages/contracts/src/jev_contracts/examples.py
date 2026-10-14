"""Native schema v2: ordered string-ID options for every decision primitive.

A full example has schema_version, id, family_id, split, task, language, variety,
state, question, target, provenance, quality, and optionally images. Inference
accepts only state, question, optional images and optional schema_version=2.
IDs are opaque text, NOT safe filenames. No validation normalizes Arabic text.
"""

from __future__ import annotations

import math
import re
import unicodedata

from ._validation import ValidationError, json_copy, object_fields, text, version
from .provenance import validate_provenance, validate_quality

SCHEMA_VERSION = 2
PROBABILITY_SUM_TOLERANCE = 1e-6
SPLITS = frozenset({"train", "development", "calibration", "test"})
QUESTION_TYPES = frozenset({"choice", "noul", "score"})
_EXAMPLE_FIELDS = {
    "schema_version", "id", "family_id", "split", "task", "language", "variety",
    "state", "question", "target", "provenance", "quality",
}


def _question(question: dict) -> None:
    object_fields(question, "question", {"type", "instructions", "options"})
    kind = text(question["type"], "question.type")
    if kind not in QUESTION_TYPES:
        raise ValidationError(f"question.type: expected one of {sorted(QUESTION_TYPES)!r}")
    text(question["instructions"], "question.instructions")
    options = question["options"]
    if type(options) is not list:
        raise ValidationError("question.options: expected an ordered array")
    lower, upper = {"choice": (1, 255), "noul": (2, 2), "score": (2, 10)}[kind]
    if not lower <= len(options) <= upper:
        raise ValidationError(f"question.options: {kind} requires {lower}..{upper} options")
    ids: set[str] = set()
    descriptions: set[str] = set()
    for i, option in enumerate(options):
        path = f"question.options[{i}]"
        object_fields(option, path, {"id", "description"})
        option_id = text(option["id"], f"{path}.id")
        description = text(option["description"], f"{path}.description")
        if option_id in ids:
            raise ValidationError(f"{path}.id: duplicate option ID {option_id!r}")
        fingerprint = " ".join(unicodedata.normalize("NFC", description).casefold().split())
        if fingerprint in descriptions:
            raise ValidationError(f"{path}.description: duplicate/indistinguishable option text")
        ids.add(option_id)
        descriptions.add(fingerprint)
    ordered_ids = [option["id"] for option in options]
    if kind == "noul" and ordered_ids != ["false", "true"]:
        raise ValidationError('question.options: noul IDs must be ["false", "true"] in that order')
    if kind == "score" and ordered_ids != [str(i) for i in range(len(options))]:
        raise ValidationError("question.options: score IDs must be ordered strings '0' through K-1")


def _input(row: dict) -> None:
    state = row["state"]
    if type(state) is str:
        text(state, "state")
    elif type(state) not in (dict, list):
        raise ValidationError("state: expected nonempty text or a structured JSON object/array")
    _question(row["question"])
    if "images" in row:
        images = row["images"]
        if type(images) is not list:
            raise ValidationError("images: expected an array of content-addressed references")
        seen: set[str] = set()
        for i, image in enumerate(images):
            reference = text(image, f"images[{i}]")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", reference) is None:
                raise ValidationError(
                    f"images[{i}]: expected sha256: followed by 64 lowercase hex digits"
                )
            if reference in seen:
                raise ValidationError(f"images[{i}]: duplicate image reference")
            seen.add(reference)


def _target(target: dict, options: list[dict]) -> None:
    if type(target) is not dict:
        raise ValidationError("target: expected an object")
    kind = target.get("kind")
    option_ids = {option["id"] for option in options}
    if kind == "hard":
        object_fields(target, "target", {"kind", "option_id"})
        option_id = text(target["option_id"], "target.option_id")
        if option_id not in option_ids:
            raise ValidationError("target.option_id: must identify a supplied option")
    elif kind == "distribution":
        object_fields(target, "target", {"kind", "probabilities"})
        probabilities = target["probabilities"]
        if type(probabilities) is not dict or set(probabilities) != option_ids:
            raise ValidationError("target.probabilities: must map exactly every supplied option ID")
        for option_id, probability in probabilities.items():
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                raise ValidationError(
                    f"target.probabilities.{option_id}: "
                    "expected a finite number in [0, 1], not bool"
                )
        if abs(math.fsum(probabilities.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValidationError(
                f"target.probabilities: sum must be 1 within {PROBABILITY_SUM_TOLERANCE}"
            )
    else:
        raise ValidationError("target.kind: expected 'hard' or 'distribution'")


def validate_example(row: dict) -> dict:
    """Validate a complete v2 row and return an unchanged, deeply detached copy.

    Score IDs are string indices '0'..'K-1'; noul IDs are exactly 'false', 'true'.
    Structured state and metadata must contain strict finite JSON, never Python
    objects. No top-level extension fields are allowed. Bronze is valid here;
    loaders must independently enforce gold/silver and release approval.
    Calibration rows require provenance.metadata.calibration =
    {schema_version: 1, subset: temperature_fit|policy_selection|risk_verification}.
    That namespace is forbidden on other splits. Family-disjointness is a
    dataset-level check, not something a single row validator can establish.
    """
    result = json_copy(row)
    object_fields(result, "example", _EXAMPLE_FIELDS, {"images"})
    version(result["schema_version"], SCHEMA_VERSION, "schema_version")
    for field in ("id", "family_id", "split", "task", "language", "variety"):
        text(result[field], field)
    if result["split"] not in SPLITS:
        raise ValidationError(f"split: expected one of {sorted(SPLITS)!r}")
    _input(result)
    _target(result["target"], result["question"]["options"])
    result["provenance"] = validate_provenance(result["provenance"])
    result["quality"] = validate_quality(result["quality"])
    calibration = result["provenance"].get("metadata", {}).get("calibration")
    if (result["split"] == "calibration") != (calibration is not None):
        raise ValidationError("provenance.metadata.calibration: required iff split is calibration")
    return result


def validate_inference(row: dict) -> dict:
    """Validate {state, question, images?, schema_version?} without supervision.

    When supplied, schema_version must be the integer 2. No fields are inserted,
    removed, coerced, or normalized; a deeply detached copy is returned. Images
    are only 'sha256:<64 lowercase hex digits>' references. Resolution, rights,
    modality support, and size limits belong to a separately approved consumer.
    """
    result = json_copy(row)
    object_fields(result, "inference", {"state", "question"}, {"images", "schema_version"})
    if "schema_version" in result:
        version(result["schema_version"], SCHEMA_VERSION, "schema_version")
    _input(result)
    return result


def options_for(question: dict) -> list[dict]:
    """Return detached [{id: str, description: str}, ...] in validated input order.

    Accepts the question object, not the whole row. Validation detects textual
    duplicates using NFC, casefold, and collapsed whitespace without altering
    any returned text. Semantic equivalence requires separate human/data checks.
    """
    result = json_copy(question, "question")
    _question(result)
    return result["options"]


def target_distribution(row: dict) -> list[float]:
    """Validate a full row and align its hard/soft target to question.options.

    Hard labels become one-hot; distributions are neither rounded nor silently
    renormalized. Both false and true are valid noul targets (indices 0 and 1).
    """
    result = validate_example(row)
    target = result["target"]
    options = result["question"]["options"]
    if target["kind"] == "hard":
        return [float(option["id"] == target["option_id"]) for option in options]
    return [float(target["probabilities"][option["id"]]) for option in options]


def renderable_input(row: dict) -> dict:
    """Validate an example/inference request and return ONLY model-visible fields.

    Returns {state, question, images?}; schema_version and all supervision,
    provenance, IDs, split/task hints, and quality metadata are excluded. Copies
    are detached, including nested state/options. This boundary prevents metadata
    leakage; it cannot detect a human placing an answer inside the allowed state.
    """
    if type(row) is not dict:
        raise ValidationError("input: expected an object")
    result = validate_example(row) if "target" in row else validate_inference(row)
    return {key: result[key] for key in ("state", "question", "images") if key in result}
