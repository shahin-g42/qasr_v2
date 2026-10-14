"""Conservative fingerprint views; original strings are never rewritten."""

from __future__ import annotations

import re
import unicodedata

from .io import require, text

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalized_view(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def numeral_to_ascii(value: str) -> str:
    text(value)
    value = value.translate(_DIGITS).replace("٫", ".").replace("−", "-")
    require(re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value) is not None,
            "invalid numeral; grouping separators, exponent and mixed text are unsupported")
    return value


def arabic_numeral(value: str) -> str:
    return numeral_to_ascii(value).translate(str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫"))


def fidelity_flags(value: str) -> list[str]:
    flags = []
    if any(unicodedata.category(char) == "Cf" for char in value):
        flags.append("format_control")
    if "\ufffd" in value:
        flags.append("replacement_character")
    if re.search(r"(?:ليس|لم|لن|مو|مش)", value):
        flags.append("negation_present")
    return flags
