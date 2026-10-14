"""Exact ordinal triage: weighted rubric items mapped to ordered levels.

Semantics are frozen under rule version triage-rubric-v1. Only met items
contribute points; the total is an exact integer sum. Levels are ordered and
each carries an ascending minimum threshold starting at zero; the answer is
the highest level whose threshold does not exceed the total. The independent
solver sums in a different order and locates the level by binary search over
the thresholds instead of a linear scan.
"""

from __future__ import annotations

from bisect import bisect_right

from ..io import boolean, fields, integer, require, text, version
from ._shared import (
    attributes,
    digits,
    join_ar,
    join_en,
    noised,
    style_of,
    surface_words,
    variant_of,
)

KIND = "triage_rubric"
RULE_VERSION = "triage-rubric-v1"
TASK = "triage_rubric"

MAX_ITEMS = 12
MAX_LEVELS = 10
MAX_POINTS = 5
MAX_THRESHOLD = 1000
MET = {"msa": "محقق", "egyptian": "متحقق", "levantine": "متحقق", "gulf": "متحقق",
       "code_switch": "محقق", "english": "met"}
NOT_MET = {"msa": "غير محقق", "egyptian": "مش متحقق", "levantine": "ما تحقق",
           "gulf": "ما تحقق", "code_switch": "غير محقق", "english": "not met"}
SCORING = {
    "msa": "قاعدة الحساب: اجمع نقاط البنود المتحققة فقط، ثم اختر أعلى مستوى لا يتجاوز "
           "مجموع النقاط عتبته.",
    "egyptian": "قاعدة الحساب: اجمع نقاط البنود المتحققة بس، وبعدين اختر أعلى مستوى عتبته "
                "ما تتجاوزش مجموع النقاط.",
    "levantine": "قاعدة الحساب: اجمع نقاط البنود المتحققة بس، وبعدين اختر أعلى مستوى عتبته "
                 "ما تتجاوز مجموع النقاط.",
    "gulf": "قاعدة الحساب: اجمع نقاط البنود المتحققة بس، وعقب اختر أعلى مستوى عتبته ما "
            "تتجاوز مجموع النقاط.",
    "code_switch": "قاعدة الحساب: اجمع نقاط البنود المتحققة فقط، ثم اختر أعلى مستوى لا "
                   "يتجاوز مجموع النقاط عتبته.",
    "english": "Scoring: add the points of met items only, then pick the highest level "
               "whose threshold does not exceed the total.",
}


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "entity", "items", "levels", "thresholds"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id", "entity"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    items = world["items"]
    require(type(items) is list and 2 <= len(items) <= MAX_ITEMS,
            "triage needs 2..12 rubric items")
    identifiers = []
    for item in items:
        fields(item, {"id", "label", "points", "met"})
        identifiers.append(text(item["id"]))
        fields(item["label"], {"ar", "en"})
        text(item["label"]["ar"])
        text(item["label"]["en"])
        integer(item["points"], 1, MAX_POINTS)
        boolean(item["met"])
    require(len(set(identifiers)) == len(identifiers), "duplicate rubric item ID")
    levels = world["levels"]
    require(type(levels) is list and 2 <= len(levels) <= MAX_LEVELS,
            "triage needs 2..10 levels")
    for level in levels:
        fields(level, {"label"})
        fields(level["label"], {"ar", "en"})
        text(level["label"]["ar"])
        text(level["label"]["en"])
    for language in ("ar", "en"):
        labels = [level["label"][language] for level in levels]
        require(len(set(labels)) == len(labels), "level labels must be distinguishable")
    thresholds = world["thresholds"]
    require(type(thresholds) is list and len(thresholds) == len(levels),
            "one threshold per level")
    for value in thresholds:
        integer(value, 0, MAX_THRESHOLD)
    require(thresholds[0] == 0, "the first level must start at zero")
    require(all(later > earlier for earlier, later in
                zip(thresholds, thresholds[1:], strict=False)),
            "thresholds must strictly increase")


def solve(world: dict) -> str:
    validate(world)
    score = sum(item["points"] for item in world["items"] if item["met"])
    level = 0
    for index, threshold in enumerate(world["thresholds"]):
        if threshold <= score:
            level = index
    return str(level)


def independent_solve(world: dict) -> str:
    validate(world)
    score = sum(item["points"] for _, item in sorted(
        (item["id"], item) for item in world["items"]) if item["met"])
    return str(bisect_right(world["thresholds"], score) - 1)


def _level_text(index: int, world: dict, style: str, language: str) -> str:
    low = digits(str(world["thresholds"][index]), style)
    if index + 1 < len(world["thresholds"]):
        high = digits(str(world["thresholds"][index + 1] - 1), style)
        span = f"{low}–{high}" if language == "ar" else f"{low}-{high}"
    else:
        span = f"{low} فما فوق" if language == "ar" else f"{low} and above"
    return f"{digits(str(index), style)}) {world['levels'][index]['label'][language]}: {span}"


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    language = "ar" if arabic else "en"
    join = join_ar if arabic else join_en
    items = []
    for item in world["items"]:
        state_word = MET[style] if item["met"] else NOT_MET[style]
        separator = "، " if arabic else ", "
        points = digits(str(item["points"]), style)
        items.append(f"{item['label'][language]}: +{points}{separator}{state_word}")
    levels = [_level_text(index, world, style, language)
              for index in range(len(world["levels"]))]
    options = [{"id": str(index), "description": level["label"][language]}
               for index, level in enumerate(world["levels"])]
    state = (f"{words['state_label']}: {world['entity']}. {words['items_label']}: "
             f"{join(items)}. {words['levels_label']}: {join(levels)}. {SCORING[style]}")
    return {"state": noised(state, noise), "question": {
        "type": "score", "instructions": words["question"], "options": options,
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
