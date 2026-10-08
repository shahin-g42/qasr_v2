"""Arabic diacritization pass: marks only, letters provably untouched.

Run after correction on every Arabic record. test3 (2026-10-09) showed why it
is a pass of its own: asked as one rule inside the correction prompt, the
model left 69% of Arabic records with no diacritics and stripped most of the
marks from fully vocalized sources (se_v1 0.76 -> 0.14 marks per letter).

The guard is what makes it safe: with every diacritic removed, the output
must equal the input character for character (letters, hamza seats, spaces,
punctuation, digits, Latin words), every mark must sit on an Arabic letter,
and no letter may carry more than two marks (shadda + one vowel). An output
that fails is retried alone, then the record keeps its undiacritized text --
a missing mark is recoverable, a changed word is not.
"""

from __future__ import annotations

import re
import unicodedata

from ..prompts import _RULE_DIACRITICS

# fathatan dammatan kasratan fatha damma kasra shadda sukun, superscript alef
MARKS = "ًٌٍَُِّْٰ"
_MARK_RE = re.compile(f"[{MARKS}]")
_ARABIC_LETTER_RE = re.compile("[ء-غف-يٱ-ۓە]")

POLICIES = ("critical", "full", "none")

_HARD_RULES = """\
You add Arabic diacritics (tashkeel) to the final transcripts of an Arabic \
speech-recognition training corpus. Your ONLY job is diacritics.

HARD RULES (an answer that breaks one is discarded):
- Return each text with EXACTLY the same letters, spaces, punctuation, \
digits and Latin-script words, in the same order.
- You may only insert or remove these marks: fatha, damma, kasra, sukun, \
shadda, fathatan, dammatan, kasratan, superscript alef.
- Never change, add, remove or reorder a letter -- not a hamza seat, not \
ta marbuta, not alef maqsura. Never correct spelling or grammar. Never \
convert dialect to Modern Standard Arabic.
- Marks go on Arabic letters only, never on digits, Latin words or punctuation.
- Diacritize dialectal words by their dialectal pronunciation (Egyptian \
"بِيِعْمِل", Gulf "شْلُون"), never with MSA case endings they do not have.
"""

_POLICY_TEXT = {
    "critical": "Policy: CRITICAL diacritics only.\n" + _RULE_DIACRITICS.split("\n", 1)[1] + "\n"
    "Every text should carry the marks this policy calls for: in practice most "
    "sentences have at least a shadda or a tanween; do not leave a text bare "
    "unless it truly has no ambiguous word, gemination or tanween.\n",
    "full": """\
Policy: FULL diacritization.
- Vocalize every Arabic word completely: short vowel or sukun on every \
letter that takes one, shadda on every geminated consonant, tanween where \
the word is indefinite and inflected.
- Case endings only where the speech is Modern Standard Arabic; use the \
pausal form (sukun / no case vowel) at the end of a phrase or sentence.
- Dialectal speech: vocalize the dialect's pronunciation word by word.
- Leave the long vowels (ا و ي) as letters; mark the consonant before them.
""",
}

_OUTPUT = """\
Output STRICT JSON only (no markdown fences, no commentary): an array with \
EXACTLY one object per item, in order:
[{"i":0,"t":"<the diacritized text>"},{"i":1,"t":"..."}]
"""


def strip_marks(text: str) -> str:
    return _MARK_RE.sub("", unicodedata.normalize("NFC", text))


def density(text: str) -> float:
    """Diacritic marks per Arabic letter."""
    letters = len(_ARABIC_LETTER_RE.findall(text))
    return len(_MARK_RE.findall(text)) / letters if letters else 0.0


def check(src: str, out: str) -> str | None:
    """Why ``out`` is not a marks-only edit of ``src``, or None when it is."""
    out = unicodedata.normalize("NFC", out)
    if strip_marks(out) != strip_marks(src):
        return "letters_changed"
    run = 0
    prev = ""
    for ch in out:
        if ch in MARKS:
            if not _ARABIC_LETTER_RE.fullmatch(prev):
                return "mark_not_on_letter"
            run += 1
            if run > 2:
                return "stacked_marks"
        else:
            prev, run = ch, 0
    return None


def system_prompt(policy: str) -> str:
    return "\n".join((_HARD_RULES, _POLICY_TEXT[policy], _OUTPUT))


def messages(texts: list[str], policy: str) -> list[dict]:
    items = "\n".join(f"{i}. <<<{t}>>>" for i, t in enumerate(texts))
    return [{"role": "system", "content": system_prompt(policy)},
            {"role": "user", "content": f"Diacritize these {len(texts)} texts. Return the JSON array, one object "
                                        f'per text, "i" from 0 to {len(texts) - 1}.\n\n{items}'}]
