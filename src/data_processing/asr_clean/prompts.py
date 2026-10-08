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

from ..generic_prompts import _conventions_for
from ..prompts import (
    _RULE_DIACRITICS,
    _RULE_DIALECT_PRESERVE,
    _RULE_ITN,
    _RULE_PUNCTUATION,
)
from .text import EXPECTED_SCRIPT, INDIC_LANGS, LANGUAGE_NAMES, cer, dominant_script, mark_ratio

PROMPT_VERSION = "asr-clean-v1"

_ADJUDICATION = """\
You are the final editor of a {language} speech-recognition training corpus. \
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
its diacritics/vowel signs.
3. Never output both alternatives of a span. Never add words that appear \
in neither transcript (orthographic fixes of the same word are fine). \
Never translate, paraphrase, or "improve" the wording.
4. Drop ASR artifacts: repeated loops absent from ORIGINAL, romanized or \
translated text, filler hallucinations on noise.
5. Keep what was spoken: fillers, hesitations, repetitions and false \
starts that the transcripts support; an utterance that genuinely stops \
mid-sentence stays incomplete.
6. If both transcripts are unusable (both garbled, or describing \
different speech so that no reading is defensible), set "drop": true.
7. Then write the chosen words in publication-quality form following the \
formatting rules below.
"""

_INDIC_MARKS = """\
- Script integrity: every vowel sign (matra), virama, chillu and anusvara \
must be present. If one transcript is missing them (letters without their \
vowel signs), take the spelling from the other.
"""

_OUTPUT = """\
Output STRICT JSON only (no markdown fences, no commentary): an array with \
EXACTLY one object per item, in order, using each item's number as "i":
[{{"i": 0, "text": "...", "choice": "original|asr|merged", "drop": false}}, ...]
"choice" says which transcript the final words mostly follow ("merged" when \
both contributed). "text" is the final transcript only.
"""

_ARABIC_FORMAT = "\n\n".join(
    ("Formatting rules (Arabic):", _RULE_ITN, _RULE_PUNCTUATION, _RULE_DIALECT_PRESERVE, _RULE_DIACRITICS)
)


def system_prompt(lang: str) -> str:
    language = LANGUAGE_NAMES.get(lang, lang)
    parts = [_ADJUDICATION.format(language=language)]
    if lang == "ar":
        parts.append(_ARABIC_FORMAT)
    else:
        parts.append(
            "Formatting rules:\n"
            "- Punctuation from syntactic and prosodic boundaries, not mechanically.\n"
            "- Inverse text normalization (ITN): numbers, times, dates, currencies and "
            "percentages in written form, converting exactly what was spoken.\n"
            "- Preserve accent, register, colloquialisms and code-switching exactly as "
            "spoken; code-switched words keep the script they were spoken in.\n"
            + _conventions_for(lang)
        )
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
        drop = obj.get("drop") is True
        text = obj.get("text")
        if not drop and (not isinstance(text, str) or not text.strip()):
            continue
        choice = obj.get("choice") if obj.get("choice") in ("original", "asr", "merged") else "unknown"
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
    if lang in INDIC_LANGS and mark_ratio(text) < 0.05 and max(mark_ratio(org), mark_ratio(asr)) > 0.15:
        return "stripped_marks"
    return None
