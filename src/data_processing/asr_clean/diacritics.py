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
# Marks each policy WRITES. test4 showed the model cannot vocalize "only the
# critical marks": records came out either fully vocalized (20-59 short vowels
# per 100 letters) or bare. So the model now vocalizes fully, as pronounced,
# and the policy is applied in code by keeping only these marks.
POLICY_MARKS = {
    "critical": KEEP + "\u0670",  # shadda, tanween, dagger alef
    "full": MARKS,
}
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

_VOCALIZE = """\
Vocalize every Arabic word completely, the way it was pronounced:
- short vowel or sukun on every letter that takes one, shadda on EVERY \
doubled consonant (including the sun letter after al-: "الشَّمْس", \
"النَّاس"), tanween wherever the word is pronounced with it;
- Modern Standard Arabic gets its case endings, with the pausal form \
(sukun / no case vowel) at the end of a phrase;
- dialectal speech is vocalized by its own pronunciation word by word, and \
gets only the tanween dialect speakers actually say -- mostly the adverbial \
fathatan ("شُكْرًا", "طَبْعًا", "أَبَدًا"); never an MSA case tanween the dialect \
does not pronounce;
- keep every mark already in the text; leave long vowels (ا و ي) as letters.
"""

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


def _split(text: str) -> tuple[str, list[set[str]]]:
    """``(base characters, marks on each base character)`` of NFC text."""
    base, marks = [], []
    for ch in unicodedata.normalize("NFC", text):
        if ch in MARKS:
            if marks:
                marks[-1].add(ch)
        else:
            base.append(ch)
            marks.append(set())
    return "".join(base), marks


def _join(base: str, marks: list[set[str]]) -> str:
    return unicodedata.normalize("NFC", "".join(b + "".join(sorted(m)) for b, m in zip(base, marks, strict=True)))


def project(src: str, out: str, policy: str, min_match: float = 0.9) -> tuple[str, str | None]:
    """Policy marks from the model's ``out`` placed on ``src``'s own letters.

    The result's letters are ``src``'s, always: the model's marks are copied
    onto the letters they align with (difflib over base characters), so an
    output that also "fixed" a hamza or added a comma still contributes its
    vowels instead of being thrown away (test4: 6% of outputs failed the strict
    check). Shadda and tanween already in ``src`` are always kept; marks go on
    Arabic letters only, at most one vowel/tanween plus a shadda per letter.
    Returns ``(text, None)`` or ``(src, reason)`` when the output does not
    align well enough to trust.
    """
    import difflib

    allowed = set(POLICY_MARKS[policy])
    s_base, s_marks = _split(src)
    o_base, o_marks = _split(out)
    sm = difflib.SequenceMatcher(a=s_base, b=o_base, autojunk=False)
    if sm.ratio() < min_match:
        return src, "misaligned"
    result = [m & set(KEEP) for m in s_marks]
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            if _ARABIC_LETTER_RE.fullmatch(s_base[a + k]):
                result[a + k] |= o_marks[b + k] & allowed
    for m in result:  # one vowel-type mark per letter (+ shadda)
        vowels = [c for c in m if c not in ("\u0651", "\u0670")]
        if len(vowels) > 1:
            keep = next((c for c in vowels if c in KEEP), sorted(vowels)[0])
            m.difference_update(c for c in vowels if c != keep)
    return _join(s_base, result), None


def system_prompt(policy: str) -> str:
    del policy  # the model always vocalizes fully; the policy filter is applied in code
    return "\n".join((_HARD_RULES, _VOCALIZE, _OUTPUT))


def messages(texts: list[str], policy: str) -> list[dict]:
    items = "\n".join(f"{i}. <<<{t}>>>" for i, t in enumerate(texts))
    return [{"role": "system", "content": system_prompt(policy)},
            {"role": "user", "content": f"Diacritize these {len(texts)} texts. Return the JSON array, one object "
                                        f'per text, "i" from 0 to {len(texts) - 1}.\n\n{items}'}]
