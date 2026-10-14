"""Dispatching facade over the frozen executable oracle rules.

Every rule family lives in jev_data.rules with its own exact semantics,
independent solver and multilingual surface. This module owns the generic
audit envelope that is identical for all of them: evidence schema, rule
version pin, world hash, family identity, rendered-surface equality, surface
hash, hard-target agreement, independent-solver agreement and template
lineage. The first released rule (rational quantity comparison) is also
re-exported here unchanged for backward compatibility.
"""

from __future__ import annotations

from jev_contracts import renderable_input, validate_example

from .io import digest, fields, require, version
from .rules import comparison as _comparison
from .rules import rule_for

# Backward-compatible aliases for the released comparison rule.
UNITS = _comparison.UNITS
COMPARATORS = _comparison.COMPARATORS
Quantity = _comparison.Quantity


def validate_world(world: dict) -> dict:
    """Legacy alias: validate a comparison world (the first released rule)."""
    _comparison.validate(world)
    return world


def solve(world: dict) -> str:
    """Legacy alias: solve a comparison world."""
    return _comparison.solve(world)


def independent_solve(world: dict) -> str:
    """Legacy alias: independently solve a comparison world."""
    return _comparison.independent_solve(world)


def surface(world: dict) -> dict:
    """Legacy alias: render a comparison world's MSA surface."""
    return _comparison.surface(world)


def check_oracle(row: dict) -> None:
    row = validate_example(row)
    evidence = row["provenance"].get("metadata", {}).get("oracle")
    require(evidence is not None, "oracle evidence missing")
    fields(evidence, {"schema_version", "rule_version", "world", "world_hash", "surface_hash"})
    version(evidence["schema_version"])
    world = evidence["world"]
    rule = rule_for(world)
    require(evidence["rule_version"] == rule.RULE_VERSION, "unsupported oracle version")
    rule.validate(world)
    require(digest(world) == evidence["world_hash"], "symbolic world hash mismatch")
    require(world["family_id"] == row["family_id"], "oracle family mismatch")
    require(row["task"] == rule.TASK, "oracle task does not match its rule")
    rendered = rule.surface(world)
    require(renderable_input(row) == rendered,
            "rendered surface does not express the symbolic world")
    require(digest(rendered) == evidence["surface_hash"], "oracle surface hash mismatch")
    require(row["target"] == {"kind": "hard", "option_id": rule.solve(world)},
            "oracle target mismatch")
    require(rule.solve(world) == rule.independent_solve(world),
            "independent solver disagreement")
    attributes = rule.attributes_for(world)
    for key in ("task", "language", "variety"):
        require(row[key] == attributes[key], f"oracle row {key} does not match its rule")
    require(row["provenance"]["origin"] == attributes["origin"],
            "oracle provenance origin does not match its rule")
    require(row["provenance"]["original_language"] == attributes["language"],
            "oracle original language does not match its rule")
    lineage = row["provenance"]["metadata"].get("lineage", {})
    require(world["template_id"] in lineage.get("template_ids", []),
            "oracle template lineage missing")
