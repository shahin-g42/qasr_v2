"""Explicit rendering conventions and literal diacritic audits, not acoustic judgments."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .core import ARABIC_VOWEL_MARKS, PROTECTED_TOKEN_RE, format_signature

STYLE_VERSION = "arabic-editorial-v1-salawat-symbol"
_MARKS = "[\\u064b-\\u0652\\u0670\\u0640]*"
_PHRASE = r"\s+".join(
    "".join(re.escape(c) + _MARKS for c in word) for word in ["صلى", "الله", "عليه", "وسلم"]
)
SALAWAT = re.compile(r"(?<!\w)" + _PHRASE + r"(?!\w)")


def ideal_style(text):
    """Only explicit NFC/whitespace conventions and the user-requested exact phrase."""
    # Unicode composition must not rewrite identifiers or numeric source strings.
    parts, end = [], 0
    for match in PROTECTED_TOKEN_RE.finditer(text):
        parts.extend((unicodedata.normalize("NFC", text[end : match.start()]), match.group()))
        end = match.end()
    parts.append(unicodedata.normalize("NFC", text[end:]))
    return SALAWAT.sub("ﷺ", " ".join("".join(parts).split()))


def apply_ideal_style(proposal, adjustments):
    before = proposal["proposed_ideal_text"]
    if before is None:
        return proposal
    after = ideal_style(before)
    if before != after:
        proposal["proposed_ideal_text"] = after
        adjustments.append(
            {
                "type": "editorial_style",
                "rule": STYLE_VERSION,
                "before": before,
                "after": after,
                "reason": "Explicit rendering convention: NFC, whitespace and صلى الله عليه وسلم → ﷺ. No new honorific is added.",
            }
        )
    return proposal


def _marks(text):
    bases, marks = [], {}
    for char in unicodedata.normalize("NFC", ideal_style(text)):
        if char in ARABIC_VOWEL_MARKS:
            if bases:
                marks.setdefault(len(bases) - 1, Counter())[char] += 1
        elif char != "ـ" and (char.isalnum() or char == "_"):
            bases.append(char)
    return bases, marks


def rendering_audit(before, after):
    """Detect mark loss at aligned letters; mark presence is not proof of correctness."""
    left, lm = _marks(before)
    right, rm = _marks(after)
    aligned = (
        format_signature(ideal_style(before)) == format_signature(ideal_style(after))
        and left == right
    )
    removed, added = Counter(), Counter()
    if aligned:
        for index in set(lm) | set(rm):
            removed.update(lm.get(index, Counter()) - rm.get(index, Counter()))
            added.update(rm.get(index, Counter()) - lm.get(index, Counter()))
    return {
        "literal_changed": before != after,
        "same_words_after_explicit_style": aligned,
        "removed_marks": dict(removed),
        "added_marks": dict(added),
        "removed_shadda": removed["ّ"],
        "note": "Literal mark comparison after the explicit ﷺ convention; not a pronunciation or semantic judgment.",
    }
