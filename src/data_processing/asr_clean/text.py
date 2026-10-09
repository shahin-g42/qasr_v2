"""Text and audio-header utilities shared by every asr_clean stage (stdlib only)."""

from __future__ import annotations

import re
import struct
import unicodedata
import wave

LANGUAGE_NAMES = {"ar": "Arabic", "en": "English", "zh": "Chinese", "hi": "Hindi", "ml": "Malayalam"}

# Dominant script a clean transcript of each language must be written in.
# Code-switched words in another script are fine as a minority.
EXPECTED_SCRIPT = {"ar": "ARABIC", "en": "LATIN", "zh": "CJK", "hi": "DEVANAGARI", "ml": "MALAYALAM"}

NO_SPACE_LANGS = {"zh"}
INDIC_LANGS = {"hi", "ml"}

# q3asr SFT rows store "language Arabic<asr_text>..." as the label.
_ENVELOPE_RE = re.compile(r"^\s*language\s+[^<]*<asr_text>", re.IGNORECASE)
_ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_AR_FOLDS = str.maketrans({"ة": "ه", "ى": "ي"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def strip_envelope(text: str) -> str:
    return _ENVELOPE_RE.sub("", text or "").strip()


def local_path(path: str) -> str:
    """The training loader's remap (qasr.data): /vast lives under /lustrefs/taiga/vast40."""
    return "/lustrefs/taiga/vast40" + path[len("/vast"):] if path.startswith("/vast/") else path


def normalize(text: str, lang: str) -> str:
    """Comparison form: NFKC, casefold, digits folded, punctuation/symbols out.

    Combining marks stay attached (Indic vowel signs are letters of the word);
    Arabic additionally drops diacritics/tatweel and folds alef forms so a
    diacritized and a bare spelling of the same word compare equal.
    """
    if lang in ("ml", "hi"):
        from .conventions import canonicalize

        text = canonicalize(text or "", lang)  # one encoding per sound (chillu, nta, danda)
    text = unicodedata.normalize("NFKC", text or "").casefold().translate(_DIGITS)
    if lang == "hi":
        # nukta and chandrabindu are spelling variants of the same word ("आजादी"
        # / "आज़ादी", "जहां" / "जहाँ"): test4's review showed them blocking good fixes
        text = text.replace("\u093c", "").replace("\u0901", "\u0902")
    if lang == "ar":
        # Standard Arabic comparison folds: alef forms, ta marbuta/ha, alef
        # maqsura/ya -- spelling fixes of the SAME word must not count as edits.
        text = _ARABIC_DIACRITICS.sub("", text).translate(_ALEF_FORMS).translate(_AR_FOLDS)
    # Zero-width (non-)joiners shape Indic/Arabic letters inside a word: drop,
    # never turn into a space (that split words). Other controls become spaces.
    text = "".join("" if unicodedata.category(c) == "Cf" else " " if unicodedata.category(c)[0] in "PSZC" else c
                   for c in text)
    return " ".join(text.split())


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


try:  # ~100x faster; identical results
    from rapidfuzz.distance import Levenshtein as _lev

    def edit_distance(a, b) -> int:
        return _lev.distance(a, b)
except ImportError:
    pass


def cer(ref: str, hyp: str, lang: str) -> float:
    """Character error rate of ``hyp`` against ``ref`` on the comparison form."""
    r, h = normalize(ref, lang).replace(" ", ""), normalize(hyp, lang).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def added_letters(src: str, out: str, lang: str) -> int:
    """Letters ``out`` inserts or substitutes relative to ``src`` (deletions are free).

    Compared on the normalized form without spaces or digits, so punctuation,
    casing, spacing, Arabic diacritics and spoken numbers becoming digits
    (a deletion of the number word) cost nothing; a changed, inflected or
    invented word does.
    """
    a = re.sub(r"\d", "", normalize(src, lang)).replace(" ", "")
    b = re.sub(r"\d", "", normalize(out, lang)).replace(" ", "")
    if a == b:
        return 0
    try:
        from rapidfuzz.distance import Levenshtein

        return sum(op.tag in ("insert", "replace") for op in Levenshtein.editops(a, b))
    except ImportError:
        pass
    # DP with the cheapest alignment, then count non-deletion steps on one optimal path.
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, added = n, m, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            added += a[i - 1] != b[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            added += 1
            j -= 1
    return added


def dominant_script(text: str, min_letters: int = 5) -> str | None:
    counts: dict[str, int] = {}
    for c in text:
        if unicodedata.category(c)[0] == "L":
            name = unicodedata.name(c, "").split(" ")[0]
            counts[name] = counts.get(name, 0) + 1
    if not counts or sum(counts.values()) < min_letters:
        return None
    return max(counts, key=counts.get)


_SCRIPT_BLOCK = {"hi": ("\u0900", "\u097f"), "ml": ("\u0d00", "\u0d7f")}


def mark_ratio(text: str, lang: str | None = None) -> float:
    """Share of letters that are combining marks (Indic vowel signs, viramas).

    With ``lang``, only that language's own script counts: Latin acronyms in a
    Hindi sentence ("PDP", "GST") have no vowel signs and must not dilute it.
    """
    block = _SCRIPT_BLOCK.get(lang or "")
    letters = [c for c in text if unicodedata.category(c)[0] in "LM" and (not block or block[0] <= c <= block[1])]
    return sum(unicodedata.category(c) in ("Mn", "Mc") for c in letters) / max(len(letters), 1)


# ---------------------------------------------------------------- duration --

def header_duration(path: str) -> float | None:
    """Duration from the file header only (no decoding)."""
    try:
        import soundfile

        info = soundfile.info(path)
        if info.samplerate and info.frames:
            return info.frames / info.samplerate
    except Exception:
        pass
    try:
        if path.endswith(".wav"):
            with wave.open(path) as w:
                return w.getnframes() / float(w.getframerate())
        if path.endswith(".flac"):
            with open(path, "rb") as fh:
                if fh.read(4) != b"fLaC":
                    return None
                fh.read(4)
                info = fh.read(34)
            rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
            total = ((info[13] & 0x0F) << 32) | struct.unpack(">I", info[14:18])[0]
            return total / rate if rate and total else None
    except (OSError, wave.Error, EOFError, IndexError, struct.error):
        return None
    return None
