"""Boolean requirement policies over typed facts, with three-valued gaps.

Semantics are frozen under rule version policy-logic-v1. A policy is a flat
'all' or 'any' combination of requirements, at most one requirement per fact,
over integer and boolean facts. A requirement whose fact is not stated is a
truth gap: three-valued evaluation answers whenever the stated facts already
force the outcome and 'insufficient' only when completions disagree. The
independent solver enumerates every two-valued completion of the gaps through
explicit truth/falsity witnesses and accepts only a unanimous verdict.
"""

from __future__ import annotations

from ..io import fields, integer, require, text, version
from ._shared import (
    OPERATORS_AR,
    OPERATORS_EN,
    attributes,
    boolean_word,
    decision_options,
    digits,
    join_ar,
    join_en,
    noised,
    style_of,
    surface_words,
    variant_of,
)

KIND = "policy_logic"
RULE_VERSION = "policy-logic-v1"
TASK = "policy_logic"

MAX_CLAUSES = 8
BOUND = 10**9


def _clause(value: dict) -> None:
    fields(value, {"fact", "op", "value"})
    text(value["fact"])
    require(value["op"] in OPERATORS_AR, "invalid requirement operator")
    if type(value["value"]) is bool:
        require(value["op"] in ("==", "!="), "boolean requirements support only equality")
    else:
        integer(value["value"], -BOUND, BOUND)


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "entity", "combine", "clauses", "labels", "facts"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id", "entity"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    require(world["combine"] in {"all", "any"}, "combine must be all or any")
    clauses = world["clauses"]
    require(type(clauses) is list and 1 <= len(clauses) <= MAX_CLAUSES,
            "policy needs 1..8 requirements")
    for clause in clauses:
        _clause(clause)
    facts = [clause["fact"] for clause in clauses]
    require(len(set(facts)) == len(facts), "each fact may carry at most one requirement")
    labels = world["labels"]
    fields(labels, set(facts))
    for name in facts:
        fields(labels[name], {"ar", "en"})
        text(labels[name]["ar"])
        text(labels[name]["en"])
    stated = world["facts"]
    require(type(stated) is dict, "facts must be an object")
    require(set(stated) <= set(facts), "facts must be the policy's own requirements")
    by_fact = {clause["fact"]: clause for clause in clauses}
    for name, value in stated.items():
        clause = by_fact[name]
        if type(value) is bool:
            require(clause["op"] in ("==", "!="), "boolean facts support only equality")
        else:
            integer(value, -BOUND, BOUND)
        require(type(value) is type(clause["value"]), "fact type differs from its requirement")


def _holds(clause: dict, value: int | bool) -> bool:
    left, right = value, clause["value"]
    return {"<": left < right, "<=": left <= right, ">": left > right,
            ">=": left >= right, "==": left == right, "!=": left != right}[clause["op"]]


def solve(world: dict) -> str:
    validate(world)
    stated = world["facts"]
    results = [_holds(clause, stated[clause["fact"]])
               for clause in world["clauses"] if clause["fact"] in stated]
    gaps = [clause for clause in world["clauses"] if clause["fact"] not in stated]
    if world["combine"] == "all":
        if False in results:
            return "ineligible"
        return "insufficient" if gaps else "eligible"
    if True in results:
        return "eligible"
    return "insufficient" if gaps else "ineligible"


def _witnesses(clause: dict) -> tuple[int | bool, int | bool]:
    """Return concrete values making the clause (true, false) respectively."""
    value, op = clause["value"], clause["op"]
    if type(value) is bool:
        return (value, not value) if op == "==" else (not value, value)
    table = {"<": (value - 1, value), "<=": (value, value + 1), ">": (value + 1, value),
             ">=": (value, value - 1), "==": (value, value + 1), "!=": (value + 1, value)}
    return table[op]


def independent_solve(world: dict) -> str:
    validate(world)
    stated = world["facts"]
    gaps = {clause["fact"]: clause for clause in world["clauses"]
            if clause["fact"] not in stated}
    names = sorted(gaps)
    verdicts = set()
    for mask in range(1 << len(names)):
        closure = []
        for clause in world["clauses"]:
            if clause["fact"] in stated:
                closure.append(_holds(clause, stated[clause["fact"]]))
            else:
                truth_maker, false_maker = _witnesses(clause)
                take_true = (mask >> names.index(clause["fact"])) & 1
                closure.append(_holds(clause, truth_maker if take_true else false_maker))
        combined = all(closure) if world["combine"] == "all" else any(closure)
        verdicts.add("eligible" if combined else "ineligible")
        if len(verdicts) > 1:
            return "insufficient"
    return next(iter(verdicts))


def _clause_text(clause: dict, labels: dict, style: str) -> str:
    label = labels[clause["fact"]]["en" if style == "english" else "ar"]
    value = (boolean_word(clause["value"], style) if type(clause["value"]) is bool
             else digits(str(clause["value"]), style))
    operators = OPERATORS_EN if style == "english" else OPERATORS_AR
    return f"{label} {operators[clause['op']]} {value}"


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    stated = world["facts"]
    rendered = [_clause_text(clause, world["labels"], style) for clause in world["clauses"]]
    provided = [
        f"{world['labels'][name]['ar' if arabic else 'en']} = "
        f"{boolean_word(value, style) if type(value) is bool else digits(str(value), style)}"
        for name, value in [(clause['fact'], stated[clause['fact']])
                            for clause in world["clauses"] if clause["fact"] in stated]
    ]
    missing = [world["labels"][clause["fact"]]["ar" if arabic else "en"]
               for clause in world["clauses"] if clause["fact"] not in stated]
    head = words["head_all"] if world["combine"] == "all" else words["head_any"]
    if arabic:
        combined = (join_ar(rendered) if world["combine"] == "all"
                    else " أو ".join(rendered))
        state = f"{words['state_label']}: {world['entity']}. {head}{combined}."
        if provided:
            state += f" {words['provided_label']}: " + join_ar(provided) + "."
        if missing:
            state += f" {words['missing_label']}: " + join_ar(missing) + "."
    else:
        combined = (join_en(rendered) if world["combine"] == "all"
                    else " or ".join(rendered))
        state = f"{words['state_label']}: {world['entity']}. {head}{combined}."
        if provided:
            state += f" {words['provided_label']}: " + ", ".join(provided) + "."
        if missing:
            state += f" {words['missing_label']}: " + ", ".join(missing) + "."
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": words["question"],
        "options": decision_options(style),
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
