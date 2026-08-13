"""Rule-based ITN (Inverse Text Normalization) verification.

Two surfaces, deliberately separate:

* :func:`enforce_digit_charset` — deterministic, applied to every piece of
  LLM output before it is stored. No judgement involved.
* :func:`itn_soft_flags` — the correction-safe subset of the checks below,
  fed to the LLM validator as advisory hints. Excludes anything the LLM
  must not "fix" (see the function docstring). :func:`numeric_soft_flags`
  is the script-independent part of it, shared with the non-Arabic
  validator.

:func:`verify_itn` is the full diagnostic superset used by offline tooling
and tests; it includes advisory-only checks that must NEVER drive the
correction loop, so it is not wired into the pipeline.
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

# Date/time output patterns. Minutes and seconds are exactly two digits so
# that scores ("3:1") and ratios do not register as clock times, and the
# day/month/year form accepts "/" only. Hyphen-separated digit groups in
# speech are overwhelmingly reference numbers rather than dates: sampling the
# real manifests turned up "قرار 22-16-2015" (a decree number), "19-0-11"
# (a pharmacy group) and a bare spoken digit run "13-14-15" — every single
# hyphen match, and not one of them a date.
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
_DMY_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_YMD_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

# "H:MM" on its own is not evidence of a clock reading. Chapter-and-verse
# citations share the shape and carry field values a clock cannot hold, so
# "भजन 42:21" (Psalm 42:21, found in the Hindi manifest) would otherwise be
# reported as an impossible hour, and "البقرة 2:70" as an impossible minute.
# A time is therefore only claimed when the speaker named one nearby. An
# absent cue costs a missed flag, which is the safe direction here: these
# hints reach an LLM whose corrections are applied to the data.
_TIME_CUE_RE = re.compile(
    r"ساعة|ساعات|صباح|مساء|ظهر|عصر|منتصف الليل|دقيقة|ثانية|توقيت"
    r"|o'clock|[ap]\.?m\.?\b|hour|minute|second"
    r"|点|點|时|時|上午|下午|中午|凌晨"
    r"|बजे|सुबह|शाम|दोपहर|रात|मिनट|सेकंड"
    r"|മണി|രാവിലെ|വൈകുന്നേരം|ഉച്ച|രാത്രി|മിനിറ്റ്",
    re.IGNORECASE,
)

# How far from the digits a time cue may sit and still be about them.
_TIME_CUE_WINDOW = 24

# Longest month length per month number. Deliberately leap-generous (Feb=29)
# and calendar-agnostic: Arabic speech carries Hijri dates as often as
# Gregorian, and we must never assume which one we are looking at.
_LONGEST_MONTH = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

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


def _valid_month_day(month: str, day: str) -> bool:
    """Whether a month/day pair could exist in any calendar we might see."""
    month_num, day_num = int(month), int(day)
    if not 1 <= month_num <= 12 or day_num < 1:
        return False
    return day_num <= _LONGEST_MONTH[month_num]


def check_datetime_sanity(text: str) -> list[str]:
    """Date and time field values that cannot exist in any calendar.

    Script-independent — only digits, separators, and a small multilingual
    set of time cue words are inspected — so this is shared by the Arabic and
    generic validators.

    Deliberately narrow, in two ways. First, only the impossible is flagged:
    by the time we see the output the spoken form is gone, so whether "3:30"
    should have been "15:30", or whether 03/04 meant March 4th or April 3rd,
    is unknowable here and belongs to the LLM, which still has the original.
    An hour of 47 or a month of 19 needs no such context.

    Second, a colon or hyphen between digits is not taken as a date or time on
    its own. Speech transcripts are full of citations, decree numbers, scores,
    and ID codes wearing the same shape, and every match found in the real
    manifests was one of those rather than a date — so the shape must be
    corroborated by a time cue or a slash-separated date before we claim it.
    Flagging only what survives both filters is what keeps this safe to feed
    the correction loop.
    """
    issues: list[str] = []
    western = normalize_digits_to_western(text)

    for match in _TIME_RE.finditer(western):
        hour, minute, second = match.groups()
        if not (
            int(hour) > 23
            or int(minute) > 59
            or (second is not None and int(second) > 59)
        ):
            continue
        # Offsets are valid in `western` because normalize_digits_to_western
        # is a 1:1 character translation.
        start = max(0, match.start() - _TIME_CUE_WINDOW)
        if _TIME_CUE_RE.search(western[start : match.end() + _TIME_CUE_WINDOW]):
            issues.append(f"Impossible time value: {match.group()}")

    for match in _YMD_DATE_RE.finditer(western):
        _year, month, day = match.groups()
        if not _valid_month_day(month, day):
            issues.append(f"Impossible date value: {match.group()}")

    for match in _DMY_DATE_RE.finditer(western):
        first, second, _year = match.groups()
        # Field order is ambiguous (21/03 vs 03/21), so flag only when
        # NEITHER reading works — then it is broken under every convention.
        if not (
            _valid_month_day(second, first) or _valid_month_day(first, second)
        ):
            issues.append(f"Impossible date value: {match.group()}")

    return list(dict.fromkeys(issues))


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
    """Full diagnostic ITN report — offline tooling and tests only.

    Superset of :func:`itn_soft_flags`: it also reports number-noun
    agreement, which is advisory-only and must never reach the correction
    loop. Use :func:`itn_soft_flags` inside the pipeline.
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
    _pct_ok, pct_issue = check_percentage_format(text)
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


def enforce_digit_charset(text: str) -> str:
    """Deterministic charset normalization for Arabic LLM output.

    Western digits are the target convention (cleaner prompt rule 2), and
    the Urdu full stop (۔) is not Arabic punctuation. Applied to EVERY
    stored text — cleaner output and validator/corrector rewrites alike.
    The correction loop rebuilds records straight from LLM text, so without
    this the reject-then-corrected records would ship a charset the cleaner
    output never contains.
    """
    return normalize_digits_to_western(text).replace("۔", ".")


def _digit_soup(text: str) -> str:
    """All digits in order, separators dropped — for presence comparisons."""
    return "".join(ch for ch in normalize_digits_to_western(text) if ch.isdigit())


def _number_tokens(text: str) -> list[str]:
    """Digit runs with separators stripped, in order of appearance."""
    western = normalize_digits_to_western(text)
    tokens = []
    for match in WESTERN_NUMBER_RE.finditer(western):
        token = match.group().replace(",", "").replace(".", "")
        if token:
            tokens.append(token)
    return tokens


def check_dropped_numbers(text: str, original_text: str) -> list[str]:
    """Numbers written as digits in the source that vanished from the output.

    Presence is tested against the output's digit "soup" (all digits, in
    order, separators removed) so the legitimate reformats all pass:
    re-splitting, re-joining, thousands separators, decimal points, and
    de-duplicated stutters. What survives is genuine loss ("2019" → no
    digits) or corruption ("2019" → "2109") — both fidelity failures.
    """
    if not original_text:
        return []
    soup = _digit_soup(text)
    dropped = [
        token for token in dict.fromkeys(_number_tokens(original_text))
        if token not in soup
    ]
    if not dropped:
        return []
    return [
        f"Numbers present in the original are missing from the output: "
        f"{', '.join(dropped[:3])}"
    ]


def numeric_soft_flags(text: str, original_text: str = "") -> list[str]:
    """Script-independent ITN hints, shared by every language.

    Only digits and their separators are inspected, so these read the same in
    Arabic, Chinese, Devanagari, Malayalam, and Latin output. The non-Arabic
    validator ran no ITN verification at all before this, which left
    ``itn_enabled`` in multilingual_cleaning.yaml asserting something nothing
    checked.
    """
    flags: list[str] = []

    suspicious = check_large_numbers(text)
    if suspicious:
        flags.append(
            f"Suspiciously large numbers (likely mis-assembled digit "
            f"readout): {', '.join(suspicious[:3])}"
        )

    flags.extend(check_datetime_sanity(text))
    flags.extend(check_dropped_numbers(text, original_text))
    return flags


def itn_soft_flags(text: str, original_text: str = "") -> list[str]:
    """Advisory ITN hints for the Arabic LLM validator. Safe to act on.

    Every flag here describes something the LLM may legitimately repair,
    because cleaner prompt rule 0 lists "digits for the same spoken number"
    among the allowed edits. Two checks are deliberately left out:

    * number-noun agreement — rule 2 says the counted noun stays exactly as
      spoken, so acting on it would corrupt the transcript (the LLM can and
      does return a corrected_text, which the caller applies).
    * digit consistency / percentage charset — :func:`enforce_digit_charset`
      already fixes those deterministically, so a flag would be either dead
      or a report of our own bug, which the LLM cannot fix anyway.
    """
    flags = numeric_soft_flags(text, original_text)

    # Arabic spoken-number words. Threshold >2: isolated cardinals are
    # usually ordinals or idioms, which rule 2 keeps as words. Three or more
    # suggests ITN was skipped.
    all_converted, unconverted = check_unconverted_numbers(text)
    if not all_converted and len(unconverted) > 2:
        flags.append(
            f"Spoken numbers left as words (expected Western digits): "
            f"{', '.join(unconverted[:5])}"
        )

    return flags


__all__ = [
    "ITNCheckResult",
    "check_datetime_sanity",
    "check_digit_consistency",
    "check_dropped_numbers",
    "check_large_numbers",
    "check_number_noun_agreement",
    "check_percentage_format",
    "check_unconverted_numbers",
    "enforce_digit_charset",
    "itn_soft_flags",
    "normalize_digits_to_western",
    "numeric_soft_flags",
    "verify_itn",
]
