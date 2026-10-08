"""Arabic diacritization pass: marks only, letters provably untouched.

Run after correction on every Arabic record. test3 (2026-10-09) showed why it
is a pass of its own: asked as one rule inside the correction prompt, the
model left 69% of Arabic records with no diacritics and stripped most of the
marks from fully vocalized sources (se_v1 0.76 -> 0.14 marks per letter).

The guard is what makes it safe: with every diacritic removed, the output
must equal the input character for character (letters, hamza seats, spaces,
punctuation, digits, Latin words), every mark must sit on an Arabic letter,
no letter may carry more than two marks (shadda + one vowel), and no shadda
or tanween the input had may disappear. Before the pass, the original's
shadda/tanween are carried onto unchanged words (transfer_marks). An output
that fails is retried alone, then the record keeps its undiacritized text --
a missing mark is recoverable, a changed word is not.
"""

from __future__ import annotations

import re
import unicodedata

# fathatan dammatan kasratan fatha damma kasra shadda sukun, superscript alef
MARKS = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670"
# Never removed once present (user decision 2026-10-09): shadda and tanween.
KEEP = "\u0651\u064b\u064c\u064d"
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
    # User decision 2026-10-09: critical-only, but shadda and tanween ALWAYS.
    "critical": """\
Policy: CRITICAL diacritics -- not full vocalization.
ALWAYS mark, on every word where it applies:
- Shadda on EVERY doubled consonant ("المدرّسة", "يُعلّم", "مرّة"), \
including the sun letter after al- ("الشّمس", "النّاس").
- Tanween on EVERY word pronounced with it: fathatan ("شكرًا", "أيضًا", \
"مدرسةً"), dammatan ("كتابٌ"), kasratan ("في بيتٍ").
Add a short vowel (fatha, damma, kasra, sukun) ONLY where the word is \
otherwise ambiguous ("عَلِم" vs "عَلَّم", "كَتَب" vs "كُتُب") and the kasra \
of the feminine "you" ("أنتِ", "لكِ", "عندكِ").
Keep every mark already in the text. Do not vocalize the remaining letters.
""",
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


def _marks_by_letter(text: str) -> list[set[str]]:
    """The marks on each base character of ``text`` (aligned with strip_marks(text))."""
    out: list[set[str]] = []
    for ch in unicodedata.normalize("NFC", text):
        if ch in MARKS:
            if out:
                out[-1].add(ch)
        else:
            out.append(set())
    return out


def check(src: str, out: str) -> str | None:
    """Why ``out`` is not a marks-only edit of ``src``, or None when it is."""
    out = unicodedata.normalize("NFC", out)
    if strip_marks(out) != strip_marks(src):
        return "letters_changed"
    for had, has in zip(_marks_by_letter(src), _marks_by_letter(out), strict=True):
        if (had & set(KEEP)) - has:
            return "dropped_shadda_or_tanween"
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


def _core(word: str) -> str:
    return "".join(_ARABIC_LETTER_RE.findall(strip_marks(word)))


def transfer_marks(org: str, text: str, marks: str = KEEP) -> str:
    """Carry ``marks`` from the original onto the corrected text, word by word.

    Only for words whose Arabic letters are identical in both (aligned with
    difflib over the word sequences), so a mark can only land on the same
    letter of the same word. Marks the corrected text already has are kept.
    """
    import difflib

    if not any(m in org for m in marks):
        return text
    ow, tw = org.split(), text.split()
    oc, tc = [_core(w) for w in ow], [_core(w) for w in tw]
    for a, b, size in difflib.SequenceMatcher(a=oc, b=tc, autojunk=False).get_matching_blocks():
        for k in range(size):
            o, t = ow[a + k], tw[b + k]
            if not oc[a + k] or not any(m in o for m in marks):
                continue
            # marks on each Arabic letter of the original word, in order
            o_marks, cur = [], None
            for ch in unicodedata.normalize("NFC", o):
                if _ARABIC_LETTER_RE.fullmatch(ch):
                    cur = set()
                    o_marks.append(cur)
                elif ch in marks and cur is not None:
                    cur.add(ch)
            rebuilt, li = [], -1
            chars = unicodedata.normalize("NFC", t)
            for idx, ch in enumerate(chars):
                rebuilt.append(ch)
                if _ARABIC_LETTER_RE.fullmatch(ch):
                    li += 1
                    present = set()
                    j = idx + 1
                    while j < len(chars) and chars[j] in MARKS:
                        present.add(chars[j])
                        j += 1
                    # shadda first (canonical order), then the tanween
                    rebuilt.extend(sorted(o_marks[li] - present, key=lambda m: m != "\u0651"))
            tw[b + k] = unicodedata.normalize("NFC", "".join(rebuilt))
    return " ".join(tw)


def system_prompt(policy: str) -> str:
    return "\n".join((_HARD_RULES, _POLICY_TEXT[policy], _OUTPUT))


def messages(texts: list[str], policy: str) -> list[dict]:
    items = "\n".join(f"{i}. <<<{t}>>>" for i, t in enumerate(texts))
    return [{"role": "system", "content": system_prompt(policy)},
            {"role": "user", "content": f"Diacritize these {len(texts)} texts. Return the JSON array, one object "
                                        f'per text, "i" from 0 to {len(texts) - 1}.\n\n{items}'}]
