"""Rational quantity comparison; the original pinned oracle rule.

Semantics are frozen under rule version rational-comparison-v1: exact Fraction
arithmetic over the audited unit table, an independent cross-multiplied sign
derivation, and the exact MSA surface the released pilot rows pin by hash.
Rendering styles only vary the surface phrasing and digit script.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..arabic import numeral_to_ascii
from ..io import fields, require, text, version
from ._shared import attributes, decision_options, numeral, noised, style_of, surface_words, variant_of

KIND = "comparison"
RULE_VERSION = "rational-comparison-v1"
TASK = "numeric_policy"

UNITS = {
    "SAR": ("money:SAR", Fraction(1), "ريال", "riyals"),
    "halala": ("money:SAR", Fraction(1, 100), "هللة", "halalas"),
    "m": ("length", Fraction(1), "متر", "meters"),
    "cm": ("length", Fraction(1, 100), "سنتيمتر", "centimeters"),
    "kg": ("mass", Fraction(1), "كيلوغرام", "kilograms"),
    "g": ("mass", Fraction(1, 1000), "غرام", "grams"),
    "ratio": ("ratio", Fraction(1), "من الواحد", "of one"),
    "percent": ("ratio", Fraction(1, 100), "بالمئة", "percent"),
}
COMPARATORS = {"<": "أقل من", "<=": "أقل من أو يساوي", ">": "أكبر من",
               ">=": "أكبر من أو يساوي", "==": "يساوي", "!=": "لا يساوي"}
ENGLISH_COMPARATORS = {"<": "less than", "<=": "at most", ">": "greater than",
                       ">=": "at least", "==": "exactly", "!=": "different from"}


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    unit: str

    @classmethod
    def from_dict(cls, value: dict) -> Quantity:
        fields(value, {"value", "unit"})
        require(value["unit"] in UNITS, "unsupported unit")
        return cls(Fraction(numeral_to_ascii(value["value"])), value["unit"])

    @property
    def base(self) -> Fraction:
        return self.value * UNITS[self.unit][1]


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "entity", "quantity",
                   "threshold", "operator"}, {"rendering"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id", "entity"):
        text(world[key])
    require(world["operator"] in COMPARATORS, "invalid comparison operator")
    variant_of(world, KIND, optional=True)
    threshold = Quantity.from_dict(world["threshold"])
    if world["quantity"] is not None:
        quantity = Quantity.from_dict(world["quantity"])
        require(UNITS[quantity.unit][0] == UNITS[threshold.unit][0], "incompatible dimensions")


def solve(world: dict) -> str:
    validate(world)
    if world["quantity"] is None:
        return "insufficient"
    left = Quantity.from_dict(world["quantity"]).base
    right = Quantity.from_dict(world["threshold"]).base
    result = {"<": left < right, "<=": left <= right, ">": left > right,
              ">=": left >= right, "==": left == right, "!=": left != right}[world["operator"]]
    return "eligible" if result else "ineligible"


def independent_solve(world: dict) -> str:
    validate(world)
    if world["quantity"] is None:
        return "insufficient"
    left = Quantity.from_dict(world["quantity"]).base
    right = Quantity.from_dict(world["threshold"]).base
    # Independently derive comparison sign by integer cross-products, not float arithmetic.
    difference = left.numerator * right.denominator - right.numerator * left.denominator
    sign = (difference > 0) - (difference < 0)
    allowed = {"<": {-1}, "<=": {-1, 0}, ">": {1}, ">=": {0, 1}, "==": {0}, "!=": {-1, 1}}
    return "eligible" if sign in allowed[world["operator"]] else "ineligible"


def _unit_word(unit: str, style: str) -> str:
    return UNITS[unit][3] if style in {"english", "code_switch"} else UNITS[unit][2]


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=True)
    words = surface_words(KIND, style, phrase)

    def quantity(value: dict) -> str:
        return numeral(numeral_to_ascii(value["value"]), style) + " " + _unit_word(
            value["unit"], style)

    threshold = quantity(world["threshold"])
    actual = (words["value_missing"] if world["quantity"] is None
              else words["value_prefix"] + quantity(world["quantity"]))
    if style == "english":
        state = (f"{words['state_label']}: {world['entity']}. {words['condition_label']}: "
                 f"{words['connector']} {ENGLISH_COMPARATORS[world['operator']]} {threshold}. "
                 f"{actual if world['quantity'] is None else actual + '.'}")
    else:
        state = (f"{words['state_label']}: {world['entity']}. {words['condition_label']}: "
                 f"{words['connector']} {COMPARATORS[world['operator']]} {threshold}. "
                 f"{actual}.")
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": words["question"],
        "options": decision_options(style),
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=True), TASK)
