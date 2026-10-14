"""Exact integer budget coverage checks for spend requests and installments.

Semantics are frozen under rule version budget-check-v1. Amounts are integer
minor units of one stated unit and all arithmetic is exact Python integers.
'eligible' means the request fits: spent + requested <= total for a single
draw, installment * months >= price for an installment plan. The independent
solver runs a sequential drawdown state machine instead of closed-form sums.
A null amount is an abstention and answers 'insufficient'.
"""

from __future__ import annotations

from ..io import fields, integer, require, text, version
from ._shared import (
    attributes,
    decision_options,
    digits,
    missing_word,
    noised,
    style_of,
    surface_words,
    variant_of,
)

KIND = "budget_check"
RULE_VERSION = "budget-check-v1"
TASK = "budget_check"

AMOUNT = 10**12


def _amount(value: object) -> None:
    if value is not None:
        integer(value, 0, AMOUNT)


def validate(world: dict) -> None:
    require(type(world) is dict, "expected an object")
    scenario = world.get("scenario")
    require(scenario in ("single", "installments"), "unknown budget scenario")
    common = {"schema_version", "kind", "family_id", "template_id", "rendering",
              "scenario", "entity", "unit"}
    if scenario == "single":
        fields(world, common | {"total", "spent", "requested"})
    else:
        fields(world, common | {"price", "installment", "months"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id", "entity"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    fields(world["unit"], {"ar", "en"})
    text(world["unit"]["ar"])
    text(world["unit"]["en"])
    if scenario == "single":
        integer(world["total"], 0, AMOUNT)
        _amount(world["spent"])
        _amount(world["requested"])
    else:
        integer(world["price"], 1, AMOUNT)
        _amount(world["installment"])
        if world["months"] is not None:
            integer(world["months"], 1, 120)


def solve(world: dict) -> str:
    validate(world)
    if world["scenario"] == "single":
        if world["spent"] is None or world["requested"] is None:
            return "insufficient"
        return ("eligible" if world["spent"] + world["requested"] <= world["total"]
                else "ineligible")
    if world["installment"] is None or world["months"] is None:
        return "insufficient"
    return "eligible" if world["installment"] * world["months"] >= world["price"] else "ineligible"


def independent_solve(world: dict) -> str:
    validate(world)
    if world["scenario"] == "single":
        if world["spent"] is None or world["requested"] is None:
            return "insufficient"
        # Sequential drawdown: subtract each withdrawal from the budget in turn.
        remaining = world["total"]
        for amount in (world["spent"], world["requested"]):
            if amount > remaining:
                return "ineligible"
            remaining -= amount
        return "eligible"
    if world["installment"] is None or world["months"] is None:
        return "insufficient"
    # Repeated addition instead of multiplication.
    paid = 0
    for _ in range(world["months"]):
        paid += world["installment"]
    return "eligible" if paid >= world["price"] else "ineligible"


def _word(value: int | None, unit: str, style: str) -> str:
    if value is None:
        return missing_word(style)
    return digits(str(value), style) + " " + unit


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    unit = world["unit"]["en" if style == "english" else "ar"]
    if world["scenario"] == "single":
        spent = _word(world["spent"], unit, style)
        requested = _word(world["requested"], unit, style)
        total = digits(str(world["total"]), style) + " " + unit
        state = (f"{words['state_label']}: {world['entity']}. {words['total_label']}: "
                 f"{total}. {words['spent_label']}: {spent}. "
                 f"{words['requested_label']}: {requested}.")
        instructions = words["question_single"]
    else:
        price = digits(str(world["price"]), style) + " " + unit
        installment = _word(world["installment"], unit, style)
        months = (missing_word(style) if world["months"] is None
                  else digits(str(world["months"]), style))
        state = (f"{words['state_label']}: {world['entity']}. {words['price_label']}: "
                 f"{price}. {words['installment_label']}: {installment}. "
                 f"{words['months_label']}: {months}.")
        instructions = words["question_installments"]
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": instructions,
        "options": decision_options(style),
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
