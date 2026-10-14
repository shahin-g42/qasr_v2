"""Ordered first-match safety gates with an explicit human-escalation default.

Semantics are frozen under rule version safety-gate-v1. Rules are evaluated in
the stated order; a rule fires when all its stated facts satisfy its clauses.
A rule whose stated clauses already fail is skipped even if it also references
an unstated fact; a rule that could still fire but references an unstated fact
stops evaluation and escalates, because an unverifiable gate must reach a
human. Unmatched requests receive the declared default outcome. The three
outcomes (allow, block, escalate) are the question's options in that order.
"""

from __future__ import annotations

from ..io import fields, integer, require, text, version
from ._shared import (
    OPERATORS_AR,
    OPERATORS_EN,
    attributes,
    boolean_word,
    digits,
    join_ar,
    join_en,
    noised,
    style_of,
    surface_words,
    variant_of,
)

KIND = "safety_gate"
RULE_VERSION = "safety-gate-v1"
TASK = "safety_gate"

MAX_RULES = 6
MAX_CLAUSES = 4
BOUND = 10**9
OUTCOMES = ("allow", "block", "escalate")
OUTCOME_WORDS = {
    "allow": {"msa": "السماح", "egyptian": "السماح", "levantine": "السماح",
              "gulf": "السماح", "code_switch": "السماح", "english": "Allow"},
    "block": {"msa": "الحجب", "egyptian": "الحجب", "levantine": "الحجب",
              "gulf": "الحجب", "code_switch": "الحجب", "english": "Block"},
    "escalate": {"msa": "التحويل إلى مراجعة بشرية", "egyptian": "التحويل لمراجعة بشرية",
                 "levantine": "التحويل لمراجعة بشرية", "gulf": "التحويل لمراجعة بشرية",
                 "code_switch": "التحويل إلى مراجعة بشرية",
                 "english": "Escalate to human review"},
}
CONVENTION = {
    "msa": "أي شرط يحتاج معلومة غير مذكورة يوقف التقييم ويُحوَّل الطلب إلى مراجعة بشرية.",
    "egyptian": "أي شرط محتاج معلومة مش مذكورة يوقف التقييم ويحوّل الطلب لمراجعة بشرية.",
    "levantine": "أي شرط بحاجة معلومة ما ذكرت يوقف التقييم ويحوّل الطلب لمراجعة بشرية.",
    "gulf": "أي شرط يحتاج معلومة ما ذُكرت يوقف التقييم ويحوّل الطلب لمراجعة بشرية.",
    "code_switch": "أي شرط يحتاج معلومة غير مذكورة يوقف التقييم ويُحوَّل الطلب إلى مراجعة بشرية.",
    "english": "Any condition needing an unstated fact stops evaluation and escalates "
               "the request to human review.",
}


def _clause(value: dict) -> None:
    fields(value, {"fact", "op", "value"})
    text(value["fact"])
    require(value["op"] in OPERATORS_AR, "invalid condition operator")
    if type(value["value"]) is bool:
        require(value["op"] in ("==", "!="), "boolean conditions support only equality")
    else:
        integer(value["value"], -BOUND, BOUND)


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "request", "facts", "fact_labels", "rules", "default"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    fields(world["request"], {"ar", "en"})
    text(world["request"]["ar"])
    text(world["request"]["en"])
    rules = world["rules"]
    require(type(rules) is list and 1 <= len(rules) <= MAX_RULES,
            "safety gate needs 1..6 rules")
    for rule in rules:
        fields(rule, {"when", "outcome"})
        require(rule["outcome"] in OUTCOMES, "invalid gate outcome")
        when = rule["when"]
        require(type(when) is list and 1 <= len(when) <= MAX_CLAUSES,
                "each rule needs 1..4 conditions")
        for clause in when:
            _clause(clause)
        names = [clause["fact"] for clause in when]
        require(len(set(names)) == len(names), "each fact may appear once per rule")
    require(world["default"] in OUTCOMES, "invalid default outcome")
    referenced = sorted({clause["fact"] for rule in rules for clause in rule["when"]})
    labels = world["fact_labels"]
    fields(labels, set(referenced))
    for name in referenced:
        fields(labels[name], {"ar", "en"})
        text(labels[name]["ar"])
        text(labels[name]["en"])
    stated = world["facts"]
    require(type(stated) is dict, "facts must be an object")
    require(set(stated) <= set(referenced), "facts must be the gate's own conditions")
    by_fact = {clause["fact"]: clause for rule in rules for clause in rule["when"]}
    for name, value in stated.items():
        clause = by_fact[name]
        if type(value) is bool:
            require(clause["op"] in ("==", "!="), "boolean facts support only equality")
        else:
            integer(value, -BOUND, BOUND)
        require(type(value) is type(clause["value"]),
                "fact type differs from its condition")


def _holds(clause: dict, value: int | bool) -> bool:
    left, right = value, clause["value"]
    return {"<": left < right, "<=": left <= right, ">": left > right,
            ">=": left >= right, "==": left == right, "!=": left != right}[clause["op"]]


def solve(world: dict) -> str:
    validate(world)
    stated = world["facts"]
    for rule in world["rules"]:
        present = [(clause, stated[clause["fact"]]) for clause in rule["when"]
                   if clause["fact"] in stated]
        if any(not _holds(clause, value) for clause, value in present):
            continue
        if any(clause["fact"] not in stated for clause in rule["when"]):
            return "escalate"
        return rule["outcome"]
    return world["default"]


def independent_solve(world: dict) -> str:
    validate(world)
    stated = world["facts"]
    # Two-pass form: classify every rule, then compare the first hitting and
    # the first gap index instead of short-circuiting sequentially.
    verdicts = []
    for rule in world["rules"]:
        present_false = any(not _holds(clause, stated[clause["fact"]])
                            for clause in rule["when"] if clause["fact"] in stated)
        has_gap = any(clause["fact"] not in stated for clause in rule["when"])
        verdicts.append("miss" if present_false else ("gap" if has_gap else "hit"))
    first_gap = next((index for index, kind in enumerate(verdicts) if kind == "gap"), None)
    first_hit = next((index for index, kind in enumerate(verdicts) if kind == "hit"), None)
    if first_gap is not None and (first_hit is None or first_gap < first_hit):
        return "escalate"
    if first_hit is None:
        return world["default"]
    return world["rules"][first_hit]["outcome"]


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
    language = "ar" if arabic else "en"
    join = join_ar if arabic else join_en
    stated = world["facts"]
    provided = [
        f"{world['fact_labels'][name][language]} = "
        f"{boolean_word(value, style) if type(value) is bool else digits(str(value), style)}"
        for name, value in sorted(stated.items())
    ]
    missing = [world["fact_labels"][name][language]
               for name in sorted({clause["fact"] for rule in world["rules"]
                                   for clause in rule["when"]} - set(stated))]
    listed = []
    for index, rule in enumerate(world["rules"]):
        clauses = join([_clause_text(clause, world["fact_labels"], style)
                        for clause in rule["when"]])
        listed.append(f"{digits(str(index + 1), style)}) "
                      f"{clauses} ← {OUTCOME_WORDS[rule['outcome']][style]}")
    listed.append(f"{words['default_label']}: {OUTCOME_WORDS[world['default']][style]}")
    rules_text = ". ".join(listed)
    nothing = "لا شيء" if arabic else "none"
    state = (f"{words['request_label']}: {world['request'][language]}. "
             f"{words['facts_label']}: {join(provided) if provided else nothing}.")
    if missing:
        state += f" {words['missing_label']}: {join(missing)}."
    state += f" {words['rules_label']}: {rules_text}. {CONVENTION[style]}"
    options = [{"id": outcome, "description": OUTCOME_WORDS[outcome][style]}
               for outcome in OUTCOMES]
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": words["question"], "options": options,
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
