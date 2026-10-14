"""Multilingual rendering primitives shared by every executable oracle rule.

A rendering style is a deterministic surface variant of one symbolic world; it
never changes the gold decision. Arabic-script styles keep Arabic-Indic
numerals; code-switch and English styles use Western digits. Each style maps to
the language/variety pair recorded on the row and to an honest origin string.
"""

from __future__ import annotations

from ..arabic import arabic_numeral
from ..io import require, text
from .phrasings import NOISE_PROFILES, apply_noise, phrase_slots

STYLES = {
    "msa": ("ar", "MSA"),
    "egyptian": ("ar", "Egyptian"),
    "levantine": ("ar", "Levantine"),
    "gulf": ("ar", "Gulf"),
    "code_switch": ("ar", "code_switch"),
    "english": ("en", "EN"),
}
ORIGINS = {"ar": "native_arabic_synthetic", "en": "native_english_synthetic"}
WESTERN_STYLES = frozenset({"code_switch", "english"})

# Colloquial affirm/deny words; kept to well-established, unambiguous forms.
AFFIRM = {"msa": "نعم", "egyptian": "أيوه", "levantine": "إي", "gulf": "أي",
          "code_switch": "نعم", "english": "yes"}
DENY = {"msa": "لا", "egyptian": "لأ", "levantine": "لأ", "gulf": "لا",
        "code_switch": "لا", "english": "no"}

ELIGIBLE = {
    "msa": "يتحقق الشرط",
    "egyptian": "الشرط متحقق",
    "levantine": "الشرط تحقق",
    "gulf": "الشرط صار متحقق",
    "code_switch": "يتحقق الشرط",
    "english": "The condition holds",
}
INELIGIBLE = {
    "msa": "لا يتحقق الشرط",
    "egyptian": "الشرط مش متحقق",
    "levantine": "الشرط ما تحقق",
    "gulf": "الشرط ما صار متحقق",
    "code_switch": "لا يتحقق الشرط",
    "english": "The condition does not hold",
}
INSUFFICIENT = {
    "msa": "المعلومات غير كافية",
    "egyptian": "المعلومات مش كافية",
    "levantine": "المعلومات ناقصة",
    "gulf": "المعلومات ما تكفي",
    "code_switch": "المعلومات غير كافية",
    "english": "Insufficient information",
}
NOUL_FALSE = {
    "msa": "لا، العبارة غير صحيحة",
    "egyptian": "لأ، الكلام مش صحيح",
    "levantine": "لأ، العبارة مش صحيحة",
    "gulf": "لا، العبارة ما هي صحيحة",
    "code_switch": "لا، العبارة غير صحيحة",
    "english": "No, the statement is false",
}
NOUL_TRUE = {
    "msa": "نعم، العبارة صحيحة",
    "egyptian": "أيوه، الكلام صحيح",
    "levantine": "إي، العبارة صحيحة",
    "gulf": "أي، العبارة صحيحة",
    "code_switch": "نعم، العبارة صحيحة",
    "english": "Yes, the statement is true",
}
NOT_STATED = {
    "msa": "غير مذكور",
    "egyptian": "مش مذكور",
    "levantine": "ما ذُكر",
    "gulf": "ما ذُكر",
    "code_switch": "غير مذكور",
    "english": "not stated",
}

# Clause operator vocabulary for fact-based rules. comparison.py keeps its own
# frozen copies because its released surfaces pin them by hash.
OPERATORS_AR = {"<": "أقل من", "<=": "أقل من أو يساوي", ">": "أكبر من",
                ">=": "أكبر من أو يساوي", "==": "يساوي", "!=": "لا يساوي"}
OPERATORS_EN = {"<": "less than", "<=": "at most", ">": "greater than",
                ">=": "at least", "==": "equals", "!=": "differs from"}


def style_of(world: dict, *, optional: bool) -> str:
    """Validate the optional rendering field and default missing ones to MSA.

    Accepts the plain v1 style string and the variant dict {"style", "phrase",
    "noise"}; both carry the style, which is all this accessor returns. Bounds
    of the phrase/noise indices are validated by variant_of, which knows the
    rule kind's bank size.
    """
    rendering = world.get("rendering")
    if rendering is None:
        require(optional, "this symbolic world requires an explicit rendering style")
        return "msa"
    if type(rendering) is dict:
        require(set(rendering) <= {"style", "phrase", "noise"},
                "unknown rendering variant keys")
        require("style" in rendering, "rendering variant requires a style")
        rendering = rendering["style"]
    require(type(rendering) is str and rendering in STYLES, "unknown rendering style")
    return rendering


def variant_of(world: dict, kind: str, *, optional: bool) -> tuple[str, int | None, int | None]:
    """Parse the rendering field: a plain style string (v1 bytes) or a variant

    dict {"style": ..., "phrase": n, "noise": m} selecting a phrasing bank entry
    and a transcript-noise profile. phrase/noise are optional and default to the
    v1 surface (index 0 of each bank is the byte-exact v1 wording).
    """
    from .phrasings import phrase_count

    rendering = world.get("rendering")
    if rendering is None:
        require(optional, "this symbolic world requires an explicit rendering style")
        return "msa", None, None
    if type(rendering) is str:
        require(rendering in STYLES, "unknown rendering style")
        return rendering, None, None
    require(type(rendering) is dict, "rendering must be a style string or a variant object")
    require(set(rendering) <= {"style", "phrase", "noise"}, "unknown rendering variant keys")
    require("style" in rendering, "rendering variant requires a style")
    style = rendering["style"]
    require(type(style) is str and style in STYLES, "unknown rendering style")
    phrase = rendering.get("phrase", 0)
    noise = rendering.get("noise", 0)
    require(type(phrase) is int and not isinstance(phrase, bool), "phrase index must be an int")
    require(type(noise) is int and not isinstance(noise, bool), "noise index must be an int")
    require(0 <= phrase < phrase_count(kind, style), "phrase index outside the bank")
    require(0 <= noise < len(NOISE_PROFILES), "noise index outside the profiles")
    return style, phrase, noise


def surface_words(kind: str, style: str, phrase: int | None) -> dict[str, str]:
    """Slot wording for one phrasing; None keeps the byte-exact v1 surface."""
    return phrase_slots(kind, style, phrase if phrase is not None else 0)


def noised(state: str, noise: int | None) -> str:
    """Apply the world's transcript-noise profile to a rendered state."""
    return apply_noise(state, noise if noise is not None else 0)


def numeral(value: str, style: str) -> str:
    """Render a validated ASCII numeral string in the style's digit script."""
    text(value)
    return value if style in WESTERN_STYLES else arabic_numeral(value)


def digits(value: str, style: str) -> str:
    """Map ASCII digits to the style's digit script without numeric validation.

    For display-only strings (dates, clock times, stamps, labels) where the
    character repertoire is wider than a single numeral.
    """
    text(value)
    if style in WESTERN_STYLES:
        return value
    return value.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def boolean_word(value: bool, style: str) -> str:
    require(type(value) is bool, "expected a literal boolean")
    return AFFIRM[style] if value else DENY[style]


def decision_options(style: str) -> list[dict]:
    """Ordered eligible/ineligible/insufficient options for choice questions."""
    return [{"id": "eligible", "description": ELIGIBLE[style]},
            {"id": "ineligible", "description": INELIGIBLE[style]},
            {"id": "insufficient", "description": INSUFFICIENT[style]}]


def noul_options(style: str) -> list[dict]:
    """Ordered false/true options exactly as the noul contract requires."""
    return [{"id": "false", "description": NOUL_FALSE[style]},
            {"id": "true", "description": NOUL_TRUE[style]}]


def missing_word(style: str) -> str:
    return NOT_STATED[style]


def join_ar(parts: list[str]) -> str:
    """Arabic-style enumeration: '، ' between items and 'و' before the last."""
    require(bool(parts), "enumeration requires at least one item")
    if len(parts) == 1:
        return parts[0]
    return "، ".join(parts[:-1]) + " و" + parts[-1]


def join_en(parts: list[str]) -> str:
    require(bool(parts), "enumeration requires at least one item")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def language_of(style: str) -> str:
    return STYLES[style][0]


def attributes(style: str, task: str) -> dict:
    language, variety = STYLES[style]
    return {"task": task, "language": language, "variety": variety,
            "origin": ORIGINS[language]}
