"""Deterministic, language-aware normalization applied to LLM *output*.

The cleaner prompt asks for Arabic commas and critical-only diacritics, but an
LLM is a probability distribution, not a rule engine. Measured on 43,500 real
cleaned records it still emitted tatweel in 0.87%, Latin commas in 0.03%, and
over-vocalized text in 4.3% -- spiking to 49.2% on one corpus. Asking again
does not fix that. Enforcing it afterwards does, and costs zero GPU.

Historically ``preprocess_text()`` ran on the cleaner's *input* and nothing ran
on its output, so every defect the LLM introduced shipped straight to the
manifest. This module is the missing second half.

Every transform here is a pure function of the text. None consult a model, so
the same input always yields the same output and the whole module is unit
testable.

Two things this module deliberately does NOT do:

*   It never changes a *letter*. Normalizing ة -> ه or ى -> ي would silently
    rewrite transcripts and break verbatim fidelity. Only keyboard-tradition
    variants that are unambiguously the same letter (Persian ی/ک -> Arabic ي/ك)
    are folded.
*   It never deletes a diacritic by turning it into a space. Combining marks
    are removed, not replaced -- replacing them inserts a word boundary and
    shatters the token, which is exactly the bug that inflates Arabic WER.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum

LOGGER = logging.getLogger("data_processing.normalize")


class DiacriticPolicy(str, Enum):
    """How much vocalization survives into the training target."""

    #: Keep shadda + tanween only. Matches the pipeline's stated policy
    #: ("Target: 5-15% of letters carry marks") and what the model actually
    #: learned to emit. Default.
    CRITICAL_ONLY = "critical_only"
    #: Keep every mark the source carried. Choose this only if you also intend
    #: to train and evaluate on fully vocalized text.
    PRESERVE_ALL = "preserve_all"
    #: Remove all marks. Simplest, but discards genuinely disambiguating
    #: shadda/tanween and loses information the speaker produced.
    STRIP_ALL = "strip_all"


# --- character classes -------------------------------------------------------

# Arabic tashkeel. Critical = shadda + the three tanween; the rest are the
# ordinary short-vowel marks that full vocalization adds.
ARABIC_SHADDA = "\u0651"
ARABIC_TANWEEN = "\u064b\u064c\u064d"
ARABIC_CRITICAL = set(ARABIC_SHADDA + ARABIC_TANWEEN)
ARABIC_OTHER_MARKS = set("\u064e\u064f\u0650\u0652\u0653\u0670\u0654\u0655\u0656\u0671")
ARABIC_TATWEEL = "\u0640"
ARABIC_LETTER = re.compile(r"[\u0621-\u064a\u0671-\u06d3]")

ARABIC_INDIC_DIGITS = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
PERSIAN_DIGITS = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
DEVANAGARI_DIGITS = "\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f"
FULLWIDTH_DIGITS = "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"
WESTERN_DIGITS = "0123456789"

# Zero-width characters that are noise in every script...
ZW_NOISE = set("\u200b\u200e\u200f\u202a\u202b\u202c\u202d\u202e\ufeff\u2060")
# ...except these two, which are semantically meaningful in Indic scripts
# (they control consonant-cluster rendering) and must survive.
ZW_MEANINGFUL = set("\u200c\u200d")

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

#: CJK punctuation. Treated as "CJK-ish" so whitespace beside it collapses
#: the same way it does beside a Han character.
CJK_PUNCT = set(
    "\u3000\u3001\u3002\u3008\u3009\u300a\u300b\u300c\u300d\u300e"
    "\u300f\u3010\u3011\u3014\u3015\u2014\u2018\u2019\u201c\u201d"
    "\u2026\u00b7\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f\uff5e"
)
HTML_TAG = re.compile(r"<[^>]{0,200}>")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Single-letter Arabic prefixes that must attach to the following word.
# "و تَدَرَّجْنَا" is an orthographic error; "وتَدَرَّجْنَا" is correct.
ARABIC_PREFIX_SPLIT = re.compile(r"(^|\s)([\u0648\u0641\u0644\u0628\u0643])(\s+)(?=[\u0621-\u064a])")

LATIN_TO_ARABIC_PUNCT = {",": "\u060c", "?": "\u061f", ";": "\u061b"}
LATIN_TO_CJK_PUNCT = {
    ",": "\uff0c", ".": "\u3002", "?": "\uff1f", "!": "\uff01",
    ";": "\uff1b", ":": "\uff1a", "(": "\uff08", ")": "\uff09",
}
SMART_QUOTES: dict[str, str] = dict.fromkeys(
    ("\u201c", "\u201d", "\u201e", "\u201f"), chr(34)
)
SMART_QUOTES.update(
    dict.fromkeys(("\u2018", "\u2019", "\u201a", "\u2039", "\u203a"), chr(39))
)

#: Languages whose script uses meaningful zero-width joiners.
INDIC_LANGS = frozenset({"hi", "ml"})


def _translate_digits(text: str, source: str) -> str:
    """Map a digit alphabet onto Western digits."""
    return text.translate(str.maketrans(source, WESTERN_DIGITS))


def _has_cjk(text: str) -> bool:
    return bool(CJK.search(text))


def normalize(
    text: str,
    lang: str,
    diacritic_policy: DiacriticPolicy | str = DiacriticPolicy.CRITICAL_ONLY,
) -> tuple[str, list[str]]:
    """Normalize one transcript deterministically.

    Returns ``(normalized_text, applied_rules)``. The rule list records which
    transforms actually changed something, so the sidecar can show *why* a
    transcript differs from its source instead of leaving that to guesswork.
    """
    policy = DiacriticPolicy(diacritic_policy)
    applied: list[str] = []
    if not text:
        return "", applied

    original = text

    # NFC first: combining marks must be in canonical order before anything
    # counts or strips them, otherwise the same visual text yields two keys.
    text = unicodedata.normalize("NFC", text)
    if text != original:
        applied.append("nfc")

    before = text
    text = HTML_TAG.sub(" ", text)
    text = CONTROL.sub(" ", text)
    keep_zw = ZW_MEANINGFUL if lang in INDIC_LANGS else set()
    text = "".join(c for c in text if c not in ZW_NOISE and (c not in ZW_MEANINGFUL or c in keep_zw))
    if text != before:
        applied.append("strip_artifacts")

    # Straighten quotes so tokenization is consistent across sources.
    before = text
    text = "".join(SMART_QUOTES.get(c, c) for c in text)
    if text != before:
        applied.append("straighten_quotes")

    # --- digits: every language in this project uses Western digits ---------
    before = text
    text = _translate_digits(text, FULLWIDTH_DIGITS)
    if lang == "ar":
        text = _translate_digits(text, ARABIC_INDIC_DIGITS)
        text = _translate_digits(text, PERSIAN_DIGITS)
    elif lang == "hi":
        text = _translate_digits(text, DEVANAGARI_DIGITS)
    if text != before:
        applied.append("western_digits")

    # --- language-specific --------------------------------------------------
    extra: list[str] = []
    if lang == "ar":
        text, extra = _normalize_arabic(text, policy)
    elif lang == "zh":
        text, extra = _normalize_cjk(text)
    elif lang == "hi":
        text, extra = _normalize_devanagari(text)
    applied.extend(extra)

    # Collapse whitespace last, so spacing fixes above are folded in too.
    before = text
    text = " ".join(text.split())
    if text != before:
        applied.append("collapse_whitespace")

    return text, applied


def _normalize_arabic(text: str, policy: DiacriticPolicy) -> tuple[str, list[str]]:
    applied: list[str] = []

    before = text
    text = text.replace(ARABIC_TATWEEL, "")
    if text != before:
        applied.append("remove_tatweel")

    # Persian/Urdu keyboard variants of two Arabic letters. These are the same
    # letter, not a different word, so folding them is safe.
    before = text
    text = text.replace("\u06cc", "\u064a").replace("\u06a9", "\u0643")
    if text != before:
        applied.append("fold_persian_letters")

    before = text
    text = "".join(LATIN_TO_ARABIC_PUNCT.get(c, c) for c in text)
    if text != before:
        applied.append("arabic_punctuation")

    before = text
    if policy is DiacriticPolicy.CRITICAL_ONLY:
        text = "".join(c for c in text if c not in ARABIC_OTHER_MARKS)
    elif policy is DiacriticPolicy.STRIP_ALL:
        text = "".join(c for c in text if c not in ARABIC_OTHER_MARKS and c not in ARABIC_CRITICAL)
    if text != before:
        applied.append(f"diacritics:{policy.value}")

    # Re-attach split single-letter prefixes. Done AFTER mark stripping so the
    # lookahead sees plain letters.
    before = text
    text = ARABIC_PREFIX_SPLIT.sub(lambda m: m.group(1) + m.group(2), text)
    if text != before:
        applied.append("join_prefixes")

    return text, applied


def _normalize_cjk(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []

    def is_cjkish(char: str | None) -> bool:
        """True for Han characters and for CJK punctuation."""
        return char is not None and (bool(CJK.match(char)) or char in CJK_PUNCT)

    def nearest_non_space(index: int, step: int) -> str | None:
        j = index
        while 0 <= j < len(text) and text[j].isspace():
            j += step
        return text[j] if 0 <= j < len(text) else None

    # Punctuation FIRST. Convert while the neighbour may still be separated by
    # whitespace, otherwise "你好 , 世界" keeps its Latin comma: the character
    # immediately before it is a space, not a Han character.
    def fix_punct(match: re.Match[str]) -> str:
        char = match.group(0)
        if char not in LATIN_TO_CJK_PUNCT:
            return char
        before = nearest_non_space(match.start() - 1, -1)
        after = nearest_non_space(match.end(), 1)
        return LATIN_TO_CJK_PUNCT[char] if (is_cjkish(before) or is_cjkish(after)) else char

    before_text = text
    text = re.sub(r"[,.?!;:()]", fix_punct, text)
    if text != before_text:
        applied.append("cjk_punctuation")

    # Then drop whitespace runs that sit between two CJK-ish characters. CJK
    # carries no word spaces, and a space inside a Han run makes the scorer
    # split one utterance into many "words" -- which is why zh WER can exceed
    # 100%. Spaces beside Latin or digits are kept: that is code-switching.
    out: list[str] = []
    last_kept: str | None = None
    i = 0
    n = len(text)
    dropped = False
    while i < n:
        char = text[i]
        if char.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            nxt = text[j] if j < n else None
            if is_cjkish(last_kept) and is_cjkish(nxt):
                dropped = True
            else:
                out.append(" ")
                last_kept = " "
            i = j
        else:
            out.append(char)
            last_kept = char
            i += 1
    if dropped:
        applied.append("remove_cjk_spaces")
        text = "".join(out)

    return text, applied


def _normalize_devanagari(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    # A Latin pipe used as a sentence terminator is always meant to be danda.
    before = text
    text = text.replace("|", "\u0964")
    if text != before:
        applied.append("danda")
    return text, applied


def dedup_key(text: str, lang: str) -> str:
    """Canonical identity of a transcript, for duplicate detection.

    Strips everything that is not a letter or digit, casefolds, and removes all
    whitespace. Two recordings of the same sentence -- different punctuation,
    different vocalization, different spacing -- collapse to one key.

    This is the check the pipeline never had. Its absence is why a TTS
    preference dataset containing 7,851 distinct sentences entered training as
    156,540 records and was reported as 198 hours of Chinese.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    if lang == "ar":
        folded = "".join(c for c in folded if c not in ARABIC_OTHER_MARKS and c not in ARABIC_CRITICAL)
        folded = folded.replace(ARABIC_TATWEEL, "")
    return "".join(c for c in folded if unicodedata.category(c)[0] in ("L", "N"))


__all__ = [
    "ARABIC_CRITICAL",
    "DiacriticPolicy",
    "dedup_key",
    "normalize",
]
