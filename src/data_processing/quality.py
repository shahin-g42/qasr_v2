"""Hard quality gates: arithmetic only, no model judgement.

Every check here is a computation over the text and the duration. None of them
consult an LLM, because the one property the previous pipeline handed to LLM
adjudication -- diacritics ratio -- is a count divided by a count. The LLM that
over-vocalized a transcript predictably declined to flag its own output, so
4.3% of records shipped in violation of the pipeline's own threshold, reaching
49.2% on one corpus. An objective quantity gets an objective gate.

Verdicts:

``PASS``
    Keep as-is.
``FIXABLE``
    Deterministically repairable -- run ``normalize()`` and gate again.
``REJECT``
    No deterministic repair exists. Drop the sample and record why.

Speech-rate banding is the highest-value check here and the one nothing else in
the pipeline does: it is the only cheap way to catch audio/text misalignment,
which otherwise trains the model to hallucinate.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .normalize import ARABIC_CRITICAL, ARABIC_OTHER_MARKS, ARABIC_TATWEEL

LOGGER = logging.getLogger("data_processing.quality")


class Verdict(str, Enum):
    PASS = "pass"
    FIXABLE = "fixable"
    REJECT = "reject"


#: Languages measured in words/second rather than characters/second.
SPACE_DELIMITED = frozenset({"ar", "en", "hi", "ml"})

CJK_RANGE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

#: Bracketed non-speech markers. The cleaner prompt requires their removal, so
#: surviving one means the cleaner failed and the sample is not salvageable by
#: normalization alone.
NOISE_MARKER = re.compile(r"\[[^\]\n]{1,40}\]")
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
SPEAKER_LABEL = re.compile(r"(?:^|\s)(?:SPEAKER|SPK|MALE|FEMALE|M|F)\s*[:#]?\s*\d*\s*[:|]", re.IGNORECASE)
TIMESTAMP_TAG = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")

REPLACEMENT_CHAR = "\ufffd"


@dataclass(slots=True)
class QualityConfig:
    """Thresholds for the hard gates. All are inclusive bounds."""

    #: Master switch for every duration-derived gate -- the ``min_duration`` /
    #: ``max_duration`` band, the missing-duration rejection and the speech-rate
    #: band. ``False`` (the default) makes :func:`gate` a text-and-richness check
    #: that never drops a clip for how long it is or how fast it is spoken;
    #: duration is still *measured* (it feeds the manifest and the
    #: ``units_per_sec`` ranking term in :func:`_score`), it just never rejects.
    #: Flip to ``True`` to re-arm the duration band and speech-rate gates with no
    #: other change.
    gate_duration: bool = False

    min_duration: float = 0.5
    max_duration: float = 30.0
    min_text_chars: int = 2
    max_text_chars: int = 2000

    #: Expected-script ratio floor. Code-switching legitimately lowers this, so
    #: the check counts Latin as expected for every language.
    min_script_ratio: float = 0.80

    #: Arabic vocalization ceiling. Was 0.4 as an LLM-adjudicated soft flag;
    #: now a hard gate, because nothing about it required judgement.
    max_diacritics_ratio: float = 0.40

    #: Longest run of one identical character before the sample is rejected.
    #: Catches ASR decode loops ("اههههههههه") and stuck-key artifacts.
    max_char_run: int = 4

    #: Highest share any single word may occupy. Catches "the the the the".
    max_word_fraction: float = 0.50

    #: Speech-rate bands in units/second: words for space-delimited scripts,
    #: characters for CJK. Deliberately generous -- the goal is to catch
    #: misalignment (a 3 s clip carrying 200 characters), not to police pace.
    rate_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "ar": (0.4, 7.0),
            "en": (0.4, 7.0),
            "hi": (0.4, 7.0),
            "ml": (0.4, 7.0),
            "zh": (0.8, 12.0),
        }
    )

    #: Content-richness floor. A transcript can clear every correctness gate and
    #: still be worthless for training -- "yes", "OK", one repeated word. The
    #: richness score (see :func:`_richness`) measures lexical variety + content
    #: volume + length; this drops the trivial tail beyond what ``too_short``
    #: catches. 0.0 disables the gate -- the default -- so a correctness-only run
    #: behaves exactly as it did before richness existed.
    min_richness: float = 0.0

    #: Content units (words, or Han characters for CJK) at which the richness
    #: volume term saturates. A transcript this long earns full volume credit;
    #: shorter ones scale down proportionally.
    richness_full_units: int = 24

    #: Character count at which the richness length term saturates. Below it the
    #: length bonus ramps from 0; at or above it the transcript is long enough to
    #: carry context and earns the full bonus.
    richness_min_chars: int = 24


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of gating one sample."""

    verdict: Verdict
    reasons: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    #: 0.0-1.0 composite used to break ties inside a duplicate group, and by
    #: the distributor when a cap forces it to choose which samples to keep.
    quality: float = 1.0
    #: 0.0-1.0 content richness (lexical variety + volume + length). Kept beside
    #: ``quality`` rather than folded into it so the build's LLM-triage threshold
    #: stays a pure correctness signal; :attr:`composite` is the fold.
    richness: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict is not Verdict.REJECT

    @property
    def composite(self) -> float:
        """``quality x richness``: the Stage-2 best-first selection key.

        Ranking pool candidates by quality *and* richness is what fills each 100k
        batch with the fullest, cleanest transcripts first. Multiplying means a
        sample must score on both axes to rank high -- a pristine but trivial
        clip and a rich but misaligned one both sink.
        """
        return round(self.quality * self.richness, 4)


def _script_ratio(text: str, lang: str) -> float:
    """Fraction of alphabetic characters belonging to the expected script(s).

    Latin counts as expected in every language, because code-switched loans are
    verbatim-correct and must not be penalized as contamination.
    """
    ranges = {
        "ar": r"[\u0621-\u064a\u0671-\u06d3]",
        "zh": r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        "hi": r"[\u0900-\u097f]",
        "ml": r"[\u0d00-\u0d7f]",
        "en": r"[A-Za-z]",
    }
    pattern = ranges.get(lang)
    if pattern is None:
        return 1.0
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    expected = re.compile(pattern)
    latin = re.compile(r"[A-Za-z]")
    hits = sum(1 for c in letters if expected.match(c) or latin.match(c))
    return hits / len(letters)


def _diacritics_ratio(text: str) -> float | None:
    """Marks per Arabic letter, or None for a script without marks."""
    letters = sum(1 for c in text if re.match(r"[\u0621-\u064a\u0671-\u06d3]", c))
    if not letters:
        return None
    marks = sum(1 for c in text if c in ARABIC_CRITICAL or c in ARABIC_OTHER_MARKS or c == ARABIC_TATWEEL)
    return marks / letters


def _max_char_run(text: str) -> int:
    best = run = 1
    prev = None
    for char in text:
        if char == prev and not char.isspace():
            run += 1
            best = max(best, run)
        else:
            run = 1
        prev = char
    return best if text else 0


def _unit_count(text: str, lang: str) -> int:
    """Scoring units: words for space-delimited scripts, characters for CJK."""
    if lang in SPACE_DELIMITED:
        return len(text.split())
    return sum(1 for c in text if CJK_RANGE.match(c)) or len(text)


def _content_units(text: str, lang: str) -> list[str]:
    """Token list for richness: casefolded words, or Han characters for CJK.

    Mirrors :func:`_unit_count` but returns the tokens themselves so type-token
    ratio can count the distinct ones. Punctuation-only tokens are dropped so
    they neither inflate the volume term nor deflate lexical variety.
    """
    if lang in SPACE_DELIMITED:
        return [w for w in text.casefold().split() if any(c.isalnum() for c in w)]
    han = [c for c in text if CJK_RANGE.match(c)]
    if han:
        return han
    return [w for w in text.casefold().split() if any(c.isalnum() for c in w)]


def _richness(text: str, lang: str, cfg: QualityConfig) -> float:
    """0.0-1.0 content-richness score, computed from text alone.

    Three weighted terms:

    * lexical type-token ratio (0.40) -- distinct units / total units. A
      transcript that repeats one word scores zero here; this is the axis the zh
      failure (7,851 transcripts entering as 156,540 records) lived on. Guarded
      by ``distinct >= 2`` so a single token -- trivially ratio 1.0 -- cannot
      masquerade as varied.
    * distinct content-unit count (0.40) -- distinct units toward
      ``richness_full_units``, so a two-word answer cannot out-score a full
      sentence no matter how unique its two words are.
    * length-in-band bonus (0.20) -- characters toward ``richness_min_chars``, a
      smooth version of the ``too_short`` gate.

    Needs no duration, so Stage 1 can score and rank external samples whose
    audio has not been fetched yet.
    """
    units = _content_units(text, lang)
    n = len(units)
    if n == 0:
        return 0.0
    distinct = len(set(units))
    ttr = (distinct / n) if distinct >= 2 else 0.0
    count = min(distinct / cfg.richness_full_units, 1.0) if cfg.richness_full_units > 0 else 1.0
    n_chars = len(text.strip())
    band = min(n_chars / cfg.richness_min_chars, 1.0) if cfg.richness_min_chars > 0 else 1.0
    return round(0.40 * ttr + 0.40 * count + 0.20 * band, 4)


def gate(
    text: str,
    lang: str,
    duration: float | None,
    config: QualityConfig | None = None,
    *,
    require_duration: bool = True,
) -> GateResult:
    """Apply every hard gate to one sample.

    The duration-derived gates -- the ``min_duration``/``max_duration`` band, the
    missing-duration rejection and the speech-rate band -- fire only when
    ``config.gate_duration`` is True. The default (False) makes this a
    text-and-richness gate that never rejects on duration, while still measuring
    it for the manifest and for the ``units_per_sec`` ranking term.

    ``require_duration=False`` runs the gate in **text-only** mode: the duration
    band and speech-rate checks are skipped while everything else -- script
    purity, contamination, repetition, richness -- still applies. Stage 1 uses it
    for external sources whose duration is unknown until the audio is fetched, so
    a metadata-only pass can gate and rank on text without dropping every row. It
    is the per-call refinement of the missing-duration case, and only has an effect
    while ``config.gate_duration`` is armed.
    """
    cfg = config or QualityConfig()
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    have_duration = duration is not None and duration > 0

    # Duration-derived rejections are opt-in via ``cfg.gate_duration``. With the
    # default (False) a clip is never dropped for a missing or out-of-band length
    # -- the gate is text-and-richness only. ``require_duration`` is the per-call
    # refinement of the missing-duration case and only bites while gating is armed.
    if cfg.gate_duration:
        if require_duration and not have_duration:
            return GateResult(Verdict.REJECT, ("duration_missing_or_nonpositive",))
        if have_duration and (duration < cfg.min_duration or duration > cfg.max_duration):
            return GateResult(
                Verdict.REJECT,
                (f"duration_out_of_band:{duration:.2f}s",),
                {"duration": duration},
            )

    if not text or not text.strip():
        return GateResult(
            Verdict.REJECT, ("empty_text",), {"duration": duration} if have_duration else {}
        )

    # --- unrecoverable contamination ---------------------------------------
    if REPLACEMENT_CHAR in text:
        reasons.append("replacement_char")
    if NOISE_MARKER.search(text):
        reasons.append("noise_marker")
    if URL.search(text):
        reasons.append("url")
    if SPEAKER_LABEL.search(text) or TIMESTAMP_TAG.search(text):
        reasons.append("speaker_label_or_timestamp")
    if not any(unicodedata.category(c).startswith("L") for c in text):
        reasons.append("no_letters")
    if reasons:
        return GateResult(
            Verdict.REJECT, tuple(reasons), {"duration": duration} if have_duration else {}
        )

    # --- length -------------------------------------------------------------
    n_chars = len(text)
    metrics["chars"] = float(n_chars)
    if n_chars < cfg.min_text_chars:
        reasons.append(f"too_short:{n_chars}ch")
    if n_chars > cfg.max_text_chars:
        reasons.append(f"too_long:{n_chars}ch")

    # --- script purity ------------------------------------------------------
    ratio = _script_ratio(text, lang)
    metrics["script_ratio"] = round(ratio, 4)
    if ratio < cfg.min_script_ratio:
        reasons.append(f"script_ratio_low:{ratio:.2f}")

    # --- vocalization (hard, arithmetic) ------------------------------------
    d_ratio = _diacritics_ratio(text) if lang == "ar" else None
    if d_ratio is not None:
        metrics["diacritics_ratio"] = round(d_ratio, 4)
        if d_ratio > cfg.max_diacritics_ratio:
            reasons.append(f"over_vocalized:{d_ratio:.2f}")

    # --- degenerate repetition ----------------------------------------------
    run = _max_char_run(text)
    metrics["max_char_run"] = float(run)
    if run > cfg.max_char_run:
        reasons.append(f"char_run:{run}")

    units = _unit_count(text, lang)
    metrics["units"] = float(units)
    if lang in SPACE_DELIMITED and units:
        counts = Counter(text.casefold().split())
        top = counts.most_common(1)[0][1] / units
        metrics["top_word_fraction"] = round(top, 4)
        if units > 5 and top > cfg.max_word_fraction:
            reasons.append(f"word_repeat:{top:.2f}")

    # --- content richness ----------------------------------------------------
    richness = _richness(text, lang, cfg)
    metrics["richness"] = richness
    if cfg.min_richness > 0.0 and richness < cfg.min_richness:
        reasons.append(f"low_richness:{richness:.2f}")

    # --- speech rate: the alignment check -----------------------------------
    # The rate is always *measured* when a duration is known -- it feeds the
    # ranking term in ``_score`` -- but it only *rejects* while ``cfg.gate_duration``
    # is armed. In text-only mode (no duration) ``units_per_sec`` is left out of the
    # metrics entirely, which is what tells ``_score`` to drop the rate term rather
    # than read the sample as infinitely slow.
    if have_duration:
        rate = units / duration
        metrics["units_per_sec"] = round(rate, 3)
        if cfg.gate_duration:
            low, high = cfg.rate_bands.get(lang, (0.0, float("inf")))
            if rate < low:
                reasons.append(f"speech_rate_low:{rate:.2f}")
            elif rate > high:
                reasons.append(f"speech_rate_high:{rate:.2f}")

    if reasons:
        return GateResult(Verdict.REJECT, tuple(reasons), metrics, quality=0.0, richness=richness)

    return GateResult(
        Verdict.PASS, (), metrics, quality=_score(metrics, cfg, lang), richness=richness
    )


def _score(metrics: dict[str, float], cfg: QualityConfig, lang: str) -> float:
    """Composite 0.0-1.0 quality score for tie-breaking and distribution.

    Rewards a script ratio near 1.0 and a speech rate near the middle of the
    expected band. Deliberately simple: it only has to rank samples against
    each other, not estimate transcription accuracy.
    """
    score = min(metrics.get("script_ratio", 1.0), 1.0)

    # The rate term applies only when a duration was known. In text-only mode
    # ``units_per_sec`` is absent; defaulting it to 0.0 would read as "infinitely
    # slow" and multiply the score to zero, so its absence means "no opinion".
    if "units_per_sec" in metrics:
        low, high = cfg.rate_bands.get(lang, (0.0, 1.0))
        rate = metrics["units_per_sec"]
        if high > low:
            mid = (low + high) / 2
            span = (high - low) / 2
            score *= max(0.0, 1.0 - abs(rate - mid) / span)

    length = metrics.get("chars", 0.0)
    if length < 20:
        score *= 0.85          # very short clips carry little context
    return round(score, 4)


__all__ = ["GateResult", "QualityConfig", "Verdict", "gate"]
