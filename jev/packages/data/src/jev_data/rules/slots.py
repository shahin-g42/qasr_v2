"""Slot-assertion verification against a filled form, answered true/false.

Semantics are frozen under rule version slot-assertion-v1. The form holds at
most one value per field. A claim may assert a literal value (true exactly
when the field carries that exact value) or the field's absence (true exactly
when the field is not present). Comparison is literal text equality; an
unstated field holds no value. The independent solver keys on (field, value)
pairs and set membership instead of dictionary lookup.

A true/false decision uses the noul primitive: options are exactly the
false/true words in that order.
"""

from __future__ import annotations

from ..io import fields, require, text, version
from ._shared import (
    attributes,
    join_ar,
    join_en,
    noised,
    noul_options,
    style_of,
    surface_words,
    variant_of,
)

KIND = "slot_assertion"
RULE_VERSION = "slot-assertion-v1"
TASK = "slot_assertion"

CONVENTION = {
    "msa": "قارن الادعاء بمحتوى النموذج حرفياً؛ حقل غير مذكور لا يحمل أي قيمة.",
    "egyptian": "قارن الادعاء بمحتوى النموذج حرفياً؛ الحقل المش مذكور ما يحملش أي قيمة.",
    "levantine": "قارن الادعاء بمحتوى النموذج حرفياً؛ الحقل اللي ما ذكر ما يحمل أي قيمة.",
    "gulf": "قارن الادعاء بمحتوى النموذج حرفياً؛ الحقل اللي ما ذُكر ما يحمل أي قيمة.",
    "code_switch": "قارن الادعاء بمحتوى النموذج حرفياً؛ حقل غير مذكور لا يحمل أي قيمة.",
    "english": "Compare the claim against the form literally; an unstated field holds "
               "no value.",
}


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "claim", "form", "field_labels"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    form = world["form"]
    require(type(form) is list and 1 <= len(form) <= 6, "form needs 1..6 fields")
    names = []
    for entry in form:
        fields(entry, {"field", "value"})
        names.append(text(entry["field"]))
        text(entry["value"])
    require(len(set(names)) == len(names), "duplicate form field")
    claim = world["claim"]
    fields(claim, {"field", "value"})
    text(claim["field"])
    if claim["value"] is not None:
        text(claim["value"])
    references = set(names) | {claim["field"]}
    labels = world["field_labels"]
    fields(labels, references)
    for name in sorted(references):
        fields(labels[name], {"ar", "en"})
        text(labels[name]["ar"])
        text(labels[name]["en"])


def solve(world: dict) -> str:
    validate(world)
    entries = {entry["field"]: entry["value"] for entry in world["form"]}
    claim = world["claim"]
    if claim["value"] is None:
        return "false" if claim["field"] in entries else "true"
    return "true" if entries.get(claim["field"]) == claim["value"] else "false"


def independent_solve(world: dict) -> str:
    validate(world)
    pairs = {(entry["field"], entry["value"]) for entry in world["form"]}
    present = {field for field, _ in pairs}
    claim = world["claim"]
    if claim["value"] is None:
        matched = claim["field"] not in present
    else:
        matched = (claim["field"], claim["value"]) in pairs
    return "true" if matched else "false"


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    language = "ar" if arabic else "en"
    join = join_ar if arabic else join_en
    labels = world["field_labels"]
    form = join([f"{labels[entry['field']][language]} = {entry['value']}"
                 for entry in world["form"]])
    claim = world["claim"]
    claim_label = labels[claim["field"]][language]
    asserted = words["missing_value"] if claim["value"] is None else claim["value"]
    state = (f"{words['form_label']}: {form}. {words['claim_label']}: "
             f"{words['claim_intro']} {claim_label} {words['value_intro']} {asserted}. "
             f"{CONVENTION[style]}")
    return {"state": noised(state, noise), "question": {
        "type": "noul", "instructions": words["question"],
        "options": noul_options(style),
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
