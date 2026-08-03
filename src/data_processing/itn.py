"""Rule-based ITN (Inverse Text Normalization) verification.

Post-LLM sanity checks for number format consistency and correctness.
These are fast, local checks that flag potential issues in the LLM output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Arabic-Indic digits: ٠١٢٣٤٥٦٧٨٩
ARABIC_INDIC_DIGITS = set("٠١٢٣٤٥٦٧٨٩")
WESTERN_DIGITS = set("0123456789")

# Patterns for number detection
WESTERN_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")
ARABIC_INDIC_NUMBER_RE = re.compile(r"[٠-٩][٠-٩,.]*[٠-٩]|[٠-٩]")

# Spoken number words (common ones that should be converted)
SPOKEN_NUMBERS = re.compile(
    r"\b(صفر|واحد|اثنين|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"أحد عشر|اثنا عشر|عشرون|ثلاثون|أربعون|خمسون|ستون|سبعون|ثمانون|تسعون|"
    r"مئة|مئتان|ثلاثمئة|أربعمئة|خمسمئة|ستمئة|سبعمئة|ثمانمئة|تسعمئة|"
    r"ألف|ألفان|آلاف|مليون|مليونا|ملايين|مليار)\b"
)

# Percentage patterns
PERCENTAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|بالمئة|في المئة)")

# Currency patterns (common Arabic currencies)
CURRENCY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:ريال|ريالات|درهم|دراهم|دينار|دنانير|جنيه|جنيهات|"
    r"دولار|دولارات|يورو|يورو)\b"
)

# Number-noun agreement patterns
# Numerals 3-10 require plural noun (تمييز جمع)
_NUMERAL_3_10_RE = re.compile(
    r"\b(ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"ثلاث|أربع|خمس|ست|سبع|ثمان|تسع)\b"
)
# Common singular nouns that should be plural after 3-10
_SINGULAR_NOUN_AFTER_3_10_RE = re.compile(
    r"\b(?:ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة|"
    r"ثلاث|أربع|خمس|ست|سبع|ثمان|تسع)\s+"
    r"(مرة|يوم|سنة|شهر|أسبوع|ساعة|دقيقة|ثانية|"
    r"ريال|درهم|دينار|جنيه|دولار|متر|كيلومتر|"
    r"كتاب|طالب|معلم|رجل|امرأة|بيت|سيارة)\b"
)
# Numerals 11-99 require singular accusative (تمييز منصوب)
_NUMERAL_11_99_RE = re.compile(
    r"\b(أحد عشر|اثنا عشر|ثلاثة عشر|أربعة عشر|خمسة عشر|"
    r"ستة عشر|سبعة عشر|ثمانية عشر|تسعة عشر|"
    r"عشرون|ثلاثون|أربعون|خمسون|ستون|سبعون|ثمانون|تسعون|"
    r"واحد وعشرون|اثنان وعشرون)\b"
)


@dataclass
class ITNCheckResult:
    """Result of ITN verification checks."""

    is_valid: bool
    issues: list[str]
    has_mixed_digits: bool = False
    has_unconverted_numbers: bool = False
    suspicious_large_numbers: list[str] | None = None


def check_digit_consistency(text: str) -> tuple[bool, str | None]:
    """Check if text mixes Arabic-Indic and Western digits.

    Returns (is_consistent, issue_description).
    """
    has_western = bool(WESTERN_DIGITS.intersection(set(text)))
    has_arabic_indic = bool(ARABIC_INDIC_DIGITS.intersection(set(text)))

    if has_western and has_arabic_indic:
        return False, "Mixed Arabic-Indic and Western digits in same text"
    return True, None


def check_unconverted_numbers(text: str) -> tuple[bool, list[str]]:
    """Check for spoken number words that should have been converted to digits.

    Returns (all_converted, list_of_unconverted_words).
    """
    matches = SPOKEN_NUMBERS.findall(text)
    # Filter out ordinal/idiomatic uses that should stay as words
    # (e.g., "الواحد" as "the One" referring to God, or "أول" patterns)
    unconverted = []
    for match in matches:
        # Skip if it's part of an ordinal construction
        # This is a heuristic — the LLM should handle most cases
        unconverted.append(match)

    return len(unconverted) == 0, unconverted


def check_large_numbers(text: str) -> list[str]:
    """Flag very large numbers that might be conversion errors.

    Numbers > 1 billion are suspicious in typical speech transcripts.
    """
    suspicious = []
    for match in WESTERN_NUMBER_RE.finditer(text):
        num_str = match.group().replace(",", "").replace(".", "")
        try:
            value = int(num_str)
            if value > 1_000_000_000:
                suspicious.append(match.group())
        except ValueError:
            continue
    return suspicious


def check_percentage_format(text: str) -> tuple[bool, str | None]:
    """Check percentage format consistency."""
    matches = PERCENTAGE_RE.findall(text)
    if not matches:
        return True, None
    # If we have percentages, they should use Western digits
    for match in matches:
        if any(c in ARABIC_INDIC_DIGITS for c in match):
            return False, f"Percentage uses Arabic-Indic digits: {match}"
    return True, None


def check_number_noun_agreement(text: str) -> list[str]:
    """Check Arabic number-noun agreement rules (تمييز العدد).

    WARNING: advisory/diagnostic only — NEVER use this to drive corrections.
    Speakers often use colloquial agreement ("خمس مرة"); the transcript must
    keep the noun exactly as spoken to match the audio.

    Rules from Arabic grammar:
    - Numerals 3-10: noun must be plural (broken or sound plural)
    - Numerals 11-99: noun must be singular accusative (تمييز منصوب)
    - Numerals 100+: noun must be singular genitive (مضاف إليه مجرور)

    Returns a list of potential agreement violations.
    """
    issues: list[str] = []

    # Check: singular noun after 3-10 (should be plural)
    for match in _SINGULAR_NOUN_AFTER_3_10_RE.finditer(text):
        numeral_and_noun = match.group(0)
        noun = match.group(1)
        issues.append(
            f"Number-noun agreement: singular '{noun}' after 3-10 numeral "
            f"(should be plural) in '{numeral_and_noun}'"
        )

    return issues


def verify_itn(text: str) -> ITNCheckResult:
    """Run all ITN verification checks on a text.

    This is the main entry point for post-LLM ITN validation.
    """
    issues: list[str] = []

    # Check digit consistency
    consistent, issue = check_digit_consistency(text)
    has_mixed = not consistent
    if issue:
        issues.append(issue)

    # Check for unconverted spoken numbers
    all_converted, unconverted = check_unconverted_numbers(text)
    if not all_converted and len(unconverted) > 2:
        # Only flag if there are multiple unconverted numbers
        issues.append(f"Unconverted number words: {', '.join(unconverted[:5])}")

    # Check for suspicious large numbers
    suspicious = check_large_numbers(text)
    if suspicious:
        issues.append(f"Suspicious large numbers: {', '.join(suspicious[:3])}")

    # Check percentage format
    pct_ok, pct_issue = check_percentage_format(text)
    if pct_issue:
        issues.append(pct_issue)

    # Check number-noun agreement
    agreement_issues = check_number_noun_agreement(text)
    issues.extend(agreement_issues)

    return ITNCheckResult(
        is_valid=len(issues) == 0,
        issues=issues,
        has_mixed_digits=has_mixed,
        has_unconverted_numbers=not all_converted,
        suspicious_large_numbers=suspicious if suspicious else None,
    )


def normalize_digits_to_western(text: str) -> str:
    """Convert all Arabic-Indic digits to Western digits.

    Covers both the Arabic-Indic block (٠-٩, U+0660) and the Extended
    Arabic-Indic block (۰-۹, U+06F0, used in Persian/Urdu-influenced text).
    Use this to enforce digit consistency when mixed digits are detected.
    """
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


__all__ = [
    "ITNCheckResult",
    "check_digit_consistency",
    "check_large_numbers",
    "check_number_noun_agreement",
    "check_percentage_format",
    "check_unconverted_numbers",
    "normalize_digits_to_western",
    "verify_itn",
]
