"""Prompts, response parsing and output guards for the ORIGINAL + ASR corrector.

The formatting rules are the corpus's established ones, imported verbatim
from the text-only cleaner (``data_processing.prompts`` for Arabic,
``generic_prompts`` for the rest) so this pass writes the same conventions as
every earlier pass: Western digits, exact-number ITN, critical-only Arabic
diacritics, dialect and code-switching preserved. What is new is the
adjudication between two transcripts of the same audio.
"""

from __future__ import annotations

import json
import re

from ..prompts import _RULE_DIALECT_PRESERVE
from .conventions import conventions
from .text import EXPECTED_SCRIPT, INDIC_LANGS, LANGUAGE_NAMES, cer, dominant_script, mark_ratio

# v2: compact output keys, frozen words in the format lane
# v3: Arabic diacritics moved to a dedicated, letter-preserving pass (diacritics.py)
# v4: complete per-language ITN/orthography (conventions.py), deterministic canon
# v5: ORIGINAL wins ties; en small numbers as words; script/acronym rules
PROMPT_VERSION = "asr-clean-v5"

_ADJUDICATION = """\
You are the final editor of {article} {language} speech-recognition training corpus. \
For each numbered item you get two independent transcripts of the SAME audio \
clip. You cannot hear the audio: decide from the two transcripts.

- ORIGINAL: the existing label. Usually right about rare words, names and \
numbers, but it may be wrong: misheard or missing words, a truncated end, \
garbled non-words, encoding junk, bracketed noise tags, stripped \
diacritics/vowel signs, or text in the wrong script.
- ASR: a fresh transcript from a strong speech recognizer trained on this \
domain. Usually right on clear speech, but it can mishear rare words and \
names, split or merge words, drop diacritics, loop (repeat a phrase \
several times), hallucinate words on noise or silence, romanize \
(e.g. pinyin), or translate.

Procedure for every item:
1. Align the two word by word. Where they agree, that wording is \
near-certain: keep it.
2. Where they disagree, choose per span the reading the speaker most \
plausibly said: grammatical and coherent in context, consistent with the \
dialect of the rest of the utterance, phonetically close to the other \
reading. Prefer ORIGINAL for names, rare terms and numbers when it is \
plausible. Prefer ASR where ORIGINAL is garbled, has non-words, is cut off \
while ASR continues coherently, misses words ASR clearly heard, or lost \
its diacritics/vowel signs. TIE-BREAK: when both readings are real, \
plausible words (a name spelled two ways, two similar-sounding words), keep \
the ORIGINAL -- the recognizer was trained on these labels, so on a close \
call it is the weaker witness.
3. Never output both alternatives of a span. Never add words that appear \
in neither transcript (orthographic fixes of the same word are fine). \
Never translate, paraphrase, or "improve" the wording.
4. Drop ASR artifacts: repeated loops absent from ORIGINAL, romanized or \
translated text, filler hallucinations on noise.
5. Keep what was spoken: fillers, hesitations, repetitions and false \
starts that the transcripts support; an utterance that genuinely stops \
mid-sentence stays incomplete. Recited scripture (Quran, hadith) and \
quoted poetry keep their exact canonical wording. Remove non-speech \
annotations ([music], <noise>, (laughter)), speaker labels and timestamps.
6. If both transcripts are unusable (both garbled, or describing \
different speech so that no reading is defensible), drop the item.
7. Then write the chosen words in publication-quality form following the \
formatting rules below.
"""

_INDIC_MARKS = """\
- Script integrity: every vowel sign (matra), virama, chillu and anusvara \
must be present. If one transcript is missing them (letters without their \
vowel signs), take the spelling from the other.
"""

# Compact keys on purpose: at ~50 output tokens per item, the old verbose
# object ("text"/"choice"/"drop": false) was ~40% scaffolding, and output
# tokens are what the corrector nodes are bound by (test3, 2026-10-09).
_OUTPUT = """\
Output STRICT JSON only (no markdown fences, no commentary, no spaces \
between keys): an array with EXACTLY one object per item, in order:
[{"i":0,"t":"<final transcript>","s":"o"},{"i":1,"t":"...","s":"a"},...]
"i" = the item number. "t" = the final transcript only. "s" = which \
transcript the final words mostly follow: "o" ORIGINAL, "a" ASR, "m" both. \
Only for an unusable item write {"i":N,"d":1} instead (step 6).
"""  # not passed through str.format: single braces

# The text-only cleaner also emitted a dialect tag; this output has no such
# field, and the instruction only invites an extra key.
_DIALECT = _RULE_DIALECT_PRESERVE.replace("\n   - Tag the detected dialect accurately.", "")
_NO_DIACRITICS = """\
5. DIACRITICS: do NOT add diacritics -- a dedicated pass adds them after you. \
Write the letters exactly; marks already present may be kept or dropped."""
_ARABIC_EXTRA = "\n\n".join((_DIALECT, _NO_DIACRITICS))


def system_prompt(lang: str) -> str:
    language = LANGUAGE_NAMES.get(lang, lang)
    parts = [_ADJUDICATION.format(language=language, article="an" if language[0] in "AEIOU" else "a")]
    parts.append(f"Formatting rules ({language}):\n" + conventions(lang))
    if lang == "ar":
        parts.append(_ARABIC_EXTRA)
    if lang in INDIC_LANGS:
        parts.append(_INDIC_MARKS)
    parts.append(_OUTPUT)
    return "\n\n".join(parts)


_AGREED = """\
In ALL of these items the two transcripts already agree on the words, so the \
words are certain. Change ONLY punctuation, casing, spacing, spoken numbers \
to digits (ITN) and diacritics. Do NOT change, inflect, pluralize, add, drop \
or "correct" any word, its grammar or its spelling, and never rewrite a \
quotation or verse.

"""


def user_prompt(items: list[dict], lang: str, lane: str = "adjudicate") -> str:
    language = LANGUAGE_NAMES.get(lang, lang)
    blocks = [
        f"{i}.\n   ORIGINAL: <<<{it['org_text']}>>>\n   ASR:      <<<{it['asr_text']}>>>"
        for i, it in enumerate(items)
    ]
    return ((_AGREED if lane == "format" else "")
            + f"Edit these {len(items)} {language} items. Return the JSON array, one object per item, "
            f'"i" from 0 to {len(items) - 1}.\n\n' + "\n\n".join(blocks))


def messages(items: list[dict], lang: str, lane: str = "adjudicate") -> list[dict]:
    return [{"role": "system", "content": system_prompt(lang)},
            {"role": "user", "content": user_prompt(items, lang, lane)}]


# ------------------------------------------------------------------ parsing --

_CHOICES = {"o": "original", "a": "asr", "m": "merged",
            "original": "original", "asr": "asr", "merged": "merged"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_response(content: str, n: int) -> dict[int, dict]:
    """``{i: {"text", "choice", "drop"}}`` for every well-formed item; others are absent.

    Strict about indices: 0-based, in range, no duplicates. A missing or
    malformed item is simply not returned, so the caller retries it rather
    than writing a guess.
    """
    content = _FENCE_RE.sub("", content.strip())
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    if not isinstance(data, list):
        return out
    for obj in data:
        if not isinstance(obj, dict):
            continue
        i = obj.get("i")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n or i in out:
            continue
        # Compact keys (t/s/d), with the verbose ones still accepted.
        drop = obj.get("d") in (1, True) or obj.get("drop") is True
        text = obj.get("t", obj.get("text"))
        if not drop and (not isinstance(text, str) or not text.strip()):
            continue
        raw_choice = obj.get("s", obj.get("choice"))
        choice = _CHOICES.get(raw_choice, "unknown")
        out[i] = {"text": (text or "").strip(), "choice": choice, "drop": drop}
    return out


# ------------------------------------------------------------------- guards --

_LEAK_RE = re.compile(r"<<<|>>>|ORIGINAL:|ASR:|<asr_text>|<\|im_")


def guard(text: str, org: str, asr: str, lang: str, max_divergence: float = 0.5) -> str | None:
    """Reason to reject an LLM output, or None when it is acceptable.

    The model may only choose between, merge, and format the two inputs; an
    output far from both is a hallucination or translation, whatever the
    prompt said.
    """
    if _LEAK_RE.search(text):
        return "prompt_leak"
    expected = EXPECTED_SCRIPT.get(lang)
    script = dominant_script(text)
    # Judged against the inputs too: code-switched Chinese ("你怎么学data
    # analysis的") has more Latin letters than Han characters, legitimately.
    if expected and script and script != expected and script not in (dominant_script(org), dominant_script(asr)):
        return "wrong_script"
    if min(cer(org, text, lang) if org else 1.0, cer(asr, text, lang) if asr else 1.0) > max_divergence:
        return "divergent"
    if len(text) > 2 * max(len(org), len(asr)) + 20:
        return "expanded"
    # Vowel signs/viramas lost relative to the better-spelled input (test3: 31
    # Hindi and 8 Malayalam outputs dropped >20% of them, e.g. "बताया" ->
    # "बताय"). The better input is the reference: the ASR learned stripped
    # spellings from stripped training rows, so following it is no excuse.
    if lang in INDIC_LANGS:
        best = max(mark_ratio(org, lang), mark_ratio(asr, lang))
        if best > 0.15 and mark_ratio(text, lang) < 0.8 * best:
            return "stripped_marks"
    return None
