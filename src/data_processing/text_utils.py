"""Language-agnostic text utilities for the generic (non-Arabic) cleaners.

Fast, local operations applied BEFORE the LLM call to reduce token waste,
plus per-language script detection used for validation hard checks.
Arabic has its own specialized module (arabic_utils.py).
"""

from __future__ import annotations

import re

from .arabic_utils import (
    collapse_repeated_chars,
    normalize_unicode,
    normalize_whitespace,
    strip_control_chars,
    strip_filler_markers,
    strip_html,
)


# Human-readable language names for prompt construction
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (Mandarin)",
    "hi": "Hindi",
    "ml": "Malayalam",
}

# Expected script ranges per language (used for hard validation checks)
LANGUAGE_SCRIPTS: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"[A-Za-z]"),
    "zh": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
    "ml": re.compile(r"[\u0d00-\u0d7f]"),
}

# Devanagari digits: ०१२३४५६७८९ (Hindi manifests may contain them)
DEVANAGARI_DIGITS = "०१२३४५६७८९"
WESTERN_DIGITS = "0123456789"


def preprocess_text_generic(text: str) -> str:
    """Apply fast language-agnostic pre-processing steps in order.

    Mirrors arabic_utils.preprocess_text but without Arabic-specific
    steps (tatweel removal).
    """
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = strip_html(text)
    text = strip_filler_markers(text)
    text = collapse_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


def contains_expected_script(text: str, language: str) -> bool:
    """Check if text contains characters of the language's expected script.

    Unknown languages fall back to "contains any alphabetic character".
    """
    pattern = LANGUAGE_SCRIPTS.get(language)
    if pattern is None:
        return any(c.isalpha() for c in text)
    return bool(pattern.search(text))


def expected_script_ratio(text: str, language: str) -> float:
    """Fraction of alphabetic characters in the language's expected script.

    Code-switching to English is common in hi/ml/zh ASR data, so Latin
    characters also count as expected for non-English languages.
    """
    pattern = LANGUAGE_SCRIPTS.get(language)
    if pattern is None:
        return 1.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    latin = LANGUAGE_SCRIPTS["en"]
    matched = sum(
        1
        for c in alpha_chars
        if pattern.match(c) or (language != "en" and latin.match(c))
    )
    return matched / len(alpha_chars)


def devanagari_to_western(text: str) -> str:
    """Convert Devanagari digits (०१२३४५६७८९) to Western (0123456789)."""
    table = str.maketrans(DEVANAGARI_DIGITS, WESTERN_DIGITS)
    return text.translate(table)


__all__ = [
    "LANGUAGE_NAMES",
    "LANGUAGE_SCRIPTS",
    "contains_expected_script",
    "devanagari_to_western",
    "expected_script_ratio",
    "preprocess_text_generic",
]
