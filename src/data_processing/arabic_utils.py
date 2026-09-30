"""Arabic text pre-processing utilities.

Fast, local operations applied BEFORE the LLM call to reduce token waste.
These do NOT perform the actual cleaning — that is the LLM's job.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic Unicode ranges
ARABIC_LETTER_RANGE = re.compile(r"[\u0621-\u064A\u0660-\u0669\u0671-\u06D5]")
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
TATWEEL = "\u0640"  # Kashida

# Arabic-Indic digits: ٠١٢٣٤٥٦٧٨٩
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"

# Control characters and common artifacts
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
HTML_TAGS = re.compile(r"<[^>]+>")
REPEATED_CHARS = re.compile(r"(.)\1{4,}")  # 5+ repeated chars
#: Same, but digits are excluded: repeated digits are spoken content.
REPEATED_NON_DIGITS = re.compile(r"(\D)\1{4,}", re.UNICODE)
FILLER_MARKERS = re.compile(
    r"\[(?:noise|music|laughter|applause|silence|inaudible|crosstalk)\]",
    re.IGNORECASE,
)

# Whitespace normalization
MULTIPLE_SPACES = re.compile(r"[ \t]+")
MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

# Dialect markers (heuristic)
GULF_MARKERS = re.compile(
    r"\b(يا|وش|ليش|كيف|مو|ما\s+هو|الله\s+ي|الحين|وين|شنو|شلون|يبي|اشوي)\b"
)
LEVANTINE_MARKERS = re.compile(
    r"\b(شو|ليش|كيف|هلق|وين|عم\s+ب|بدي|عنجد|كتير|هيك|منيح|إشي)\b"
)
EGYPTIAN_MARKERS = re.compile(
    r"\b(ايه|ليه|ازاي|دلوقتي|فين|عايز|خالص|اوي|كده|بتاع|حاجة)\b"
)
MAGHREBI_MARKERS = re.compile(
    r"\b(واش|علاش|كيفاش|دابا|فين|بغيت|بزاف|هاد|ديال)\b"
)
IRAQI_MARKERS = re.compile(
    r"\b(شلون|اكو|ماكو|هذني|يريد|گام|شنو|مو|جان|هاي|هاذا)\b"
)


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def remove_tatweel(text: str) -> str:
    """Strip kashida/tatweel characters (U+0640)."""
    return text.replace(TATWEEL, "")


def strip_control_chars(text: str) -> str:
    """Remove non-printable control characters."""
    return CONTROL_CHARS.sub("", text)


def strip_html(text: str) -> str:
    """Remove HTML tags."""
    return HTML_TAGS.sub("", text)


def strip_filler_markers(text: str) -> str:
    """Remove ASR filler markers like [noise], [music], etc."""
    return FILLER_MARKERS.sub("", text)


def collapse_repeated_chars(text: str) -> str:
    """Collapse runs of 5+ identical NON-DIGIT characters to 3.

    Letter/punctuation runs are elongation or ASR stutter artifacts. Digit
    runs are CONTENT: "111111" is six spoken digits (an ID, a phone number,
    a code) and collapsing it silently changes what the speaker said.
    """
    return REPEATED_NON_DIGITS.sub(r"\1\1\1", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs and trim."""
    text = MULTIPLE_SPACES.sub(" ", text)
    text = MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Apply all fast pre-processing steps in order.

    This is the main entry point for pre-LLM text normalization.
    """
    text = normalize_unicode(text)
    text = remove_tatweel(text)
    text = strip_control_chars(text)
    text = strip_html(text)
    text = strip_filler_markers(text)
    text = collapse_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic script characters."""
    return bool(ARABIC_LETTER_RANGE.search(text))


def arabic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Arabic script."""
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    arabic_count = sum(1 for c in alpha_chars if ARABIC_LETTER_RANGE.match(c))
    return arabic_count / len(alpha_chars)


def has_diacritics(text: str) -> bool:
    """Check if text already contains Arabic diacritical marks (tashkeel)."""
    return bool(ARABIC_DIACRITICS.search(text))


def count_diacritics(text: str) -> int:
    """Count the number of diacritical marks in text."""
    return len(ARABIC_DIACRITICS.findall(text))


def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks (tashkeel) from text."""
    return ARABIC_DIACRITICS.sub("", text)


def diacritics_ratio(text: str) -> float:
    """Ratio of diacritical marks to Arabic letters.

    A ratio > 0.5 suggests over-vocalization (full tashkeel), which is
    undesirable for ASR training targets. Critical diacritics only
    (shadda + tanween + disambiguation) typically yield ratio < 0.15.
    """
    arabic_letters = len(ARABIC_LETTER_RANGE.findall(text))
    if arabic_letters == 0:
        return 0.0
    return count_diacritics(text) / arabic_letters


# Critical diacritics: shadda (ّ) and tanween (ٌ ٍ ً)
SHADDA = "\u0651"
TANWEEN = set("\u064B\u064C\u064D")  # fathatan, dammatan, kasratan
CRITICAL_DIACRITICS_RE = re.compile(r"[\u0651\u064B-\u064D]")


def count_critical_diacritics(text: str) -> int:
    """Count shadda and tanween marks (the critical diacritics we restore)."""
    return len(CRITICAL_DIACRITICS_RE.findall(text))


def classify_dialect(text: str) -> str:
    """Heuristic dialect classification based on lexical markers.

    Returns one of: gulf, levantine, egyptian, maghrebi, iraqi, msa, unknown
    """
    scores = {
        "gulf": len(GULF_MARKERS.findall(text)),
        "levantine": len(LEVANTINE_MARKERS.findall(text)),
        "egyptian": len(EGYPTIAN_MARKERS.findall(text)),
        "maghrebi": len(MAGHREBI_MARKERS.findall(text)),
        "iraqi": len(IRAQI_MARKERS.findall(text)),
    }
    max_score = max(scores.values())
    if max_score == 0:
        # No strong dialect markers — likely MSA or unknown
        return "msa" if contains_arabic(text) else "unknown"

    # Return the dialect with highest score
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best


def arabic_indic_to_western(text: str) -> str:
    """Convert Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western (0123456789)."""
    table = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)
    return text.translate(table)


def western_to_arabic_indic(text: str) -> str:
    """Convert Western digits (0123456789) to Arabic-Indic (٠١٢٣٤٥٦٧٨٩)."""
    table = str.maketrans(WESTERN_DIGITS, ARABIC_INDIC_DIGITS)
    return text.translate(table)


__all__ = [
    "arabic_indic_to_western",
    "arabic_ratio",
    "classify_dialect",
    "contains_arabic",
    "count_critical_diacritics",
    "count_diacritics",
    "diacritics_ratio",
    "has_diacritics",
    "normalize_unicode",
    "normalize_whitespace",
    "preprocess_text",
    "remove_tatweel",
    "strip_control_chars",
    "strip_diacritics",
    "strip_filler_markers",
    "strip_html",
    "western_to_arabic_indic",
]
