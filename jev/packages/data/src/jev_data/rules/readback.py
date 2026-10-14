"""Voice-agent readback verification: does the read-back quantity match?

Semantics are frozen under rule version readback-match-v1. Quantities are
compared after exact unit normalization on the audited rational unit table; a
read-back in a compatible dimension whose normalized value is equal matches,
an incompatible dimension or a missing read-back does not. The primary solver
uses Fraction arithmetic; the independent solver parses the decimal manually
into integer (numerator, denominator) pairs and compares integer cross
products only, so the two derivations share no arithmetic path.

A true/false decision uses the noul primitive: options are exactly the
false/true words in that order.
"""

from __future__ import annotations

from fractions import Fraction

from ..arabic import numeral_to_ascii
from ..io import fields, require, text, version
from ._shared import (
    attributes,
    digits,
    noised,
    noul_options,
    style_of,
    surface_words,
    variant_of,
)

KIND = "readback_match"
RULE_VERSION = "readback-match-v1"
TASK = "readback_match"

# audited unit table: unit -> (dimension, numerator, denominator) in base units
UNITS = {
    "SAR": ("money:SAR", 1, 1),
    "halala": ("money:SAR", 1, 100),
    "m": ("length", 1, 1),
    "cm": ("length", 1, 100),
    "kg": ("mass", 1, 1),
    "g": ("mass", 1, 1000),
    "hour": ("duration", 60, 1),
    "minute": ("duration", 1, 1),
    "second": ("duration", 1, 60),
}
UNIT_WORDS = {
    "SAR": ("ريال", "riyals"), "halala": ("هللة", "halalas"),
    "m": ("متر", "meters"), "cm": ("سنتيمتر", "centimeters"),
    "kg": ("كيلوغرام", "kilograms"), "g": ("غرام", "grams"),
    "hour": ("ساعة", "hours"), "minute": ("دقيقة", "minutes"),
    "second": ("ثانية", "seconds"),
}
CONVENTION = {
    "msa": "تُقارن القيمتان بعد توحيد الوحدات المتفقة في البُعد؛ التطابق التام فقط يعني نعم.",
    "egyptian": "القيمتين يتقارنوا بعد توحيد الوحدات المتفقة؛ لازم تطابق تام عشان يكون نعم.",
    "levantine": "القيمتين يتقارنوا بعد توحيد الوحدات المتفقة؛ لازم تطابق تام عشان يكون نعم.",
    "gulf": "القيمتين يتقارنون بعد توحيد الوحدات المتفقة؛ لازم تطابق تام عشان يكون نعم.",
    "code_switch": "تُقارن القيمتان بعد توحيد الوحدات المتفقة في البُعد؛ التطابق التام فقط "
                   "يعني نعم.",
    "english": "The quantities are compared after normalizing to a shared dimension; "
               "only an exact match means yes.",
}


def _quantity(value: dict) -> None:
    fields(value, {"value", "unit"})
    numeral_to_ascii(text(value["value"]))
    require(value["unit"] in UNITS, "unsupported unit")


def validate(world: dict) -> None:
    fields(world, {"schema_version", "kind", "family_id", "template_id", "rendering",
                   "expected", "readback"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    _quantity(world["expected"])
    if world["readback"] is not None:
        _quantity(world["readback"])


def _base(value: dict) -> Fraction:
    _, numerator, denominator = UNITS[value["unit"]]
    return Fraction(numeral_to_ascii(value["value"])) * Fraction(numerator, denominator)


def solve(world: dict) -> str:
    validate(world)
    if world["readback"] is None:
        return "false"
    expected, readback = world["expected"], world["readback"]
    if UNITS[expected["unit"]][0] != UNITS[readback["unit"]][0]:
        return "false"
    return "true" if _base(expected) == _base(readback) else "false"


def _pair(value: str) -> tuple[int, int]:
    """Parse a validated numeral into an exact (numerator, denominator) pair."""
    ascii_value = numeral_to_ascii(value)
    sign = -1 if ascii_value.startswith("-") else 1
    whole, _, fraction = ascii_value.lstrip("+-").partition(".")
    return sign * int(whole + fraction), 10 ** len(fraction)


def independent_solve(world: dict) -> str:
    validate(world)
    if world["readback"] is None:
        return "false"
    expected, readback = world["expected"], world["readback"]
    if UNITS[expected["unit"]][0] != UNITS[readback["unit"]][0]:
        return "false"
    # Cross-multiplied integer comparison of value * unit as exact ratios.
    e_num, e_den = _pair(expected["value"])
    r_num, r_den = _pair(readback["value"])
    _, e_unit_num, e_unit_den = UNITS[expected["unit"]]
    _, r_unit_num, r_unit_den = UNITS[readback["unit"]]
    left = e_num * e_unit_num * r_den * r_unit_den
    right = r_num * r_unit_num * e_den * e_unit_den
    return "true" if left == right else "false"


def _render(value: dict, style: str, arabic: bool) -> str:
    word = UNIT_WORDS[value["unit"]][0 if arabic else 1]
    return digits(numeral_to_ascii(value["value"]), style) + " " + word


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    expected = _render(world["expected"], style, arabic)
    readback = (words["missing_readback"] if world["readback"] is None
                else _render(world["readback"], style, arabic))
    state = (f"{words['expected_label']}: {expected}. {words['readback_label']}: "
             f"{readback}. {CONVENTION[style]}")
    return {"state": noised(state, noise), "question": {
        "type": "noul", "instructions": words["question"],
        "options": noul_options(style),
    }}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
