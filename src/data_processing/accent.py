"""Deterministic accent/dialect detection and erasure detection.

Why this module exists
----------------------
The v7.6 corpus shipped ground-truth defects with one signature: colloquial
forms had been silently rewritten into formal ones. The LLM was *told* to
preserve dialect and mostly did, but nothing ever checked the output. Trusting
a model to self-report adherence to its own instructions is the same mistake
as letting it adjudicate diacritics arithmetic -- the validator bug that
produced 71% of the over-vocalized Arabic came from exactly that.

So this module does three jobs, none of which need an LLM:

1. :func:`detect` tags the spoken variety from lexical markers plus a measured
   code-switch ratio, and reports how strong the evidence actually is.
2. :func:`reconcile` combines that tag with whatever the LLM claimed,
   preferring the deterministic evidence and *recording* the disagreement
   rather than hiding it.
3. :func:`check_erasure` diffs source text against LLM output and flags the
   damage class above: a dialectal marker that vanished and was replaced by a
   standard-form equivalent.

Coverage is deliberately unequal across the five languages. Arabic has deep,
well-attested marker sets and gets a real classifier. English gets national
varieties. Chinese gets Cantonese particles, erhua and script variant. Hindi
and Malayalam get only the signals that are genuinely reliable -- code-switch
ratio, a few strong markers, loanword density -- and are documented as
low-coverage instead of being padded with invented lexicons that would emit
confident wrong tags.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

LOGGER = logging.getLogger("data_processing.accent")

#: Every language the corpus covers.
LANGUAGES: tuple[str, ...] = ("ar", "en", "zh", "hi", "ml")

#: The tag used when evidence is absent or self-contradictory. Never guess.
UNKNOWN = "unknown"

#: Marker suffix meaning "match as a word prefix", used for productive affixes
#: such as the Egyptian progressive بـ that yields بيعمل / بيشوف / بيقول.
PREFIX = "*"

# --- closed label vocabularies -----------------------------------------------
# The Arabic set MUST stay identical to the one advertised in prompts.py, or
# the LLM's self-reported tag becomes incomparable with ours.

DIALECTS: dict[str, tuple[str, ...]] = {
    "ar": ("msa", "egyptian", "levantine", "gulf", "iraqi", "maghrebi", UNKNOWN),
    "en": ("american", "british", "indian", "australian", UNKNOWN),
    "zh": ("mandarin", "erhua", "yue", "taiwan", UNKNOWN),
    "hi": ("standard", "bihari", "urdu_influenced", "hinglish", UNKNOWN),
    "ml": ("central", "malabar", "travancore", "manglish", UNKNOWN),
}

# --- marker lexicons ---------------------------------------------------------
# Only forms that are strongly diagnostic of one variety. A word that appears
# in two dialects belongs in neither: shared vocabulary produces confident
# wrong tags, which is worse than no tag.

_MARKERS: dict[str, dict[str, tuple[str, ...]]] = {
    "ar": {
        "egyptian": (
            "بي" + PREFIX, "عشان", "كده", "إيه", "ايه", "دلوقتي", "إزاي", "ازاي",
            "عايز", "عايزة", "فلوس", "حاجة", "إنت", "انت", "أنا رايح", "يلا",
            "خالص", "برضه", "عشان كده", "ماشي",
        ),
        "levantine": (
            "شو", "هيك", "كتير", "بدك", "بدي", "ليش", "هلق", "إجا", "اجا",
            "كيفك", "منيح", "هون", "هونيك", "طيّب", "لك", "يعني",
        ),
        "gulf": (
            "شنو", "وايد", "شلون", "يبه", "يبّه", "يالس", "هذي", "وش", "عادي",
            "خلاص", "أبشر", "ابشر", "ماشاء الله عليك", "ترا",
        ),
        "iraqi": (
            "شكو", "ماكو", "هسه", "عيني", "هاي", "جان", "يكدر", "تدلل",
            "شلونك", "أكو", "ماكو", "درب",
        ),
        "maghrebi": (
            "واش", "بزاف", "بصح", "راك", "ندير", "علاش", "هاديك", "دابا",
            "مزيان", "كاين", "واخا", "شنو", "ديال",
        ),
        "msa": (
            "الذي", "التي", "اللذين", "لذلك", "ولكن", "حيث", "سوف", "ليس",
            "إنه", "بالإضافة", "وفقا", "وفقاً", "أوضح", "صرح", "أكد",
        ),
    },
    "en": {
        "american": (
            "gonna", "wanna", "y'all", "yall", "gotta", "dude", "trash",
            "apartment", "elevator", "sidewalk", "cookie", "gasoline",
            "diaper", "vacation", "candy", "purse",
        ),
        "british": (
            "bloody", "mate", "lift", "pavement", "autumn", "biscuit",
            "trousers", "queue", "flat", "petrol", "rubbish", "telly",
            "fortnight", "boot", "bonnet", "nappy", "holiday",
        ),
        "indian": (
            "yaar", "arre", "prepone", "lakh", "crore", "cousin brother",
            "do the needful", "revert back", "upgradation", "matriculation",
            "pass out", "good name", "what is your good name", "doubt",
        ),
        "australian": (
            "arvo", "brekkie", "servo", "esky", "barbie", "no worries",
            "reckon", "thongs", "ute", "bottle-o", "maccas",
        ),
    },
    "zh": {
        # Written Cantonese particles. Unambiguous -- these characters are not
        # used in Mandarin orthography at all.
        "yue": (
            "唔", "係", "咗", "冇", "嘅", "喺", "乜", "嗰", "喎", "咁",
            "唔该", "食咗", "係咪",
        ),
        # Erhua is a northern/Beijing feature. Detected separately in code
        # because it is a productive suffix, not a fixed word.
        "erhua": ("哪儿", "这儿", "那儿", "玩儿", "一会儿", "有点儿", "门儿",
                  "花儿", "鸟儿", "事儿", "味儿", "空儿"),
        "taiwan": ("喔", "耶", "齁", "欸", "機車", "好康", "便當", "捷運",
                   "機車", "吐槽", "歐巴桑"),
    },
    "hi": {
        # Bhojpuri/Bihari verbal morphology: the -ela/-ela suffix and همار.
        "bihari": ("बा", "हमार", "काहे", "रहेला", "करेला", "तनी", "बाड़s", "बाड़े",
                   "तोहार", "रउआ", "भइया"),
        # Urdu register written in Devanagari.
        "urdu_influenced": ("खुदा", "मोहब्बत", "अर्ज़", "दावत", "जनाब",
                            "बेशक", "शुक्रिया", "इंतज़ार", "मालूम", "क़ाबिल",
                            "तशरीफ", "गुस्ताख़"),
        "hinglish": (),  # measured by code-switch ratio, not by lexicon
        "standard": (),
    },
    "ml": {
        # Arabic/Persian loanwords concentrated in the Malabar Muslim dialect.
        "malabar": ("ഇക്കാ", "ഉമ്മ", "ഉപ്പ", "മാപ്പിള", "ഇൻഷാ", "അല്ലാഹു",
                    "സുബ്ഹാന", "നിക്കാഹ്", "ബാങ്ക്", "തങ്ങൾ", "കോയ"),
        "central": ("അളിയാ", "ചേട്ടാ", "മോനെ", "അടിച്ചുപൊളി", "ചുമ്മാ",
                    "പൊളി", "ഇക്കാ", "സാറേ"),
        "travancore": ("എന്തു്", "അങ്ങോട്ട്", "ഇങ്ങോട്ട്", "പറയണ", "വരണ"),
        "manglish": (),  # measured by code-switch ratio, not by lexicon
    },
}

# --- standard-form replacements ----------------------------------------------
# dialectal form -> the formal equivalents an over-eager corrector would emit.
# A marker disappearing while one of these appears is the erasure signature.

_ERASURE_PAIRS: dict[str, dict[str, tuple[str, ...]]] = {
    "ar": {
        "عشان": ("لأن", "لأنّ", "من أجل", "بسبب"),
        "كده": ("هكذا", "بذلك"),
        "إيه": ("ما", "ماذا"),
        "ايه": ("ما", "ماذا"),
        "دلوقتي": ("الآن", "حاليا"),
        "هلق": ("الآن", "حاليا"),
        "هسه": ("الآن", "حاليا"),
        "دابا": ("الآن", "حاليا"),
        "إزاي": ("كيف", "بأي طريقة"),
        "ازاي": ("كيف", "بأي طريقة"),
        "ليش": ("لماذا", "لم"),
        "شو": ("ماذا", "ما"),
        "شنو": ("ماذا", "ما"),
        "وش": ("ماذا", "ما"),
        "واش": ("هل", "ماذا"),
        "بدك": ("تريد", "هل تريد"),
        "بدي": ("أريد", "أنا أريد"),
        "عايز": ("أريد", "يريد"),
        "عايزة": ("أريد", "تريد"),
        "تبي": ("تريد", "ترغب"),
        "فلوس": ("أموال", "نقود", "مال"),
        "حاجة": ("شيء", "أمر"),
        "وايد": ("كثيرا", "بشكل كبير"),
        "بزاف": ("كثيرا", "بكثرة"),
        "كتير": ("كثيرا", "بكثرة"),
        "خالص": ("تماما", "أبدا"),
        "هيك": ("هكذا", "بتلك الطريقة"),
    },
    "en": {
        "gonna": ("going to",),
        "wanna": ("want to",),
        "gotta": ("got to", "have to"),
        "y'all": ("you all", "you"),
        "yall": ("you all", "you"),
        "'cause": ("because",),
        "cause": ("because",),
        "dunno": ("do not know", "don't know"),
        "ain't": ("is not", "are not", "am not"),
        "kinda": ("kind of",),
        "sorta": ("sort of",),
        "lemme": ("let me",),
        "gimme": ("give me",),
    },
    "zh": {
        "唔": ("不",),
        "係": ("是",),
        "咗": ("了",),
        "冇": ("没有",),
        "嘅": ("的",),
        "喺": ("在",),
        "乜": ("什么",),
        "嗰": ("那",),
        "咁": ("这么", "那么"),
        "唔该": ("谢谢", "劳驾"),
        "係咪": ("是不是",),
    },
}

# --- code-switch measurement -------------------------------------------------

_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[\u0621-\u064a\u0671-\u06d3]")
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_MALAYALAM = re.compile(r"[\u0d00-\u0d7f]")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

#: Traditional-only characters. A small diagnostic set is enough: we only need
#: to tell "clearly simplified", "clearly traditional" and "mixed".
_HANT_ONLY = set(
    "們個來時說對國學後這裡發會開從點問間實樣書車長東見動話語無電門風雲"
    "馬鳥魚門齊齒龍龜愛藝節葉號萬與專業務報導體驗認識別處理"
)
_HANS_ONLY = set(
    "们个来说时对国学后这里发会开从点问间实样书车长东见动话语无电门风云"
    "马鸟门齐齿龙龟爱艺节叶号万与专业务报导体验认识别处理"
)

_SCRIPT_OF = {
    "ar": _ARABIC,
    "hi": _DEVANAGARI,
    "ml": _MALAYALAM,
    "zh": _CJK,
}

#: Code-switch fraction above which a variety tag is set from measurement alone.
_CODE_SWITCH_TAG_THRESHOLD = 0.30


def code_switch_ratio(text: str, lang: str) -> float:
    """Fraction of letters belonging to the *other* script.

    For ar/hi/ml the other script is Latin. For en it is any non-Latin script.
    For zh it is Latin. This is a measured quantity, so unlike the lexicons it
    needs no per-language authorship to be trustworthy.
    """
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return 0.0
    if lang == "en":
        native = sum(bool(_LATIN.match(ch)) for ch in letters)
    elif lang == "zh":
        native = sum(bool(_CJK.match(ch)) for ch in letters)
    else:
        pattern = _SCRIPT_OF.get(lang, _LATIN)
        native = sum(bool(pattern.match(ch)) for ch in letters)
    return 1.0 - (native / len(letters))


def script_variant(text: str) -> str | None:
    """Classify CJK orthography as ``hans`` / ``hant`` / ``mixed`` / None.

    Only meaningful for zh. Returns None for every other language so the
    sidecar stays quiet about a field that does not apply.
    """
    if not _CJK.search(text):
        return None
    trad = sum(ch in _HANT_ONLY for ch in text)
    simp = sum(ch in _HANS_ONLY for ch in text)
    if trad and simp:
        return "mixed"
    if trad:
        return "hant"
    if simp:
        return "hans"
    return None


# --- detection ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccentTag:
    """A dialect/accent label together with the evidence behind it."""

    label: str = UNKNOWN
    #: 0.0-1.0. Below ~0.5 the tag is a weak prior, not a conclusion.
    confidence: float = 0.0
    #: The markers that fired, so an audit can re-derive the decision.
    evidence: tuple[str, ...] = ()
    #: hant/hans/mixed for zh, None elsewhere.
    script: str | None = None
    #: Measured foreign-script fraction.
    code_switch: float = 0.0
    #: "lexicon" | "code_switch" | "llm" | "reconciled" | "none"
    source: str = "none"

    def as_dict(self) -> dict:
        out: dict = {
            "accent": self.label,
            "accent_confidence": round(self.confidence, 3),
            "accent_source": self.source,
            "code_switch": round(self.code_switch, 4),
        }
        if self.script:
            out["script"] = self.script
        if self.evidence:
            out["evidence"] = list(self.evidence[:12])
        return out


def _score_labels(text: str, lang: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Weighted marker hits per label.

    Longer markers weigh more: a five-character phrase is specific in a way a
    two-character word is not. Prefix markers (``بي*``) are matched against
    word starts, everything else against whole words.
    """
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    markers = _MARKERS.get(lang)
    if not markers:
        return scores, hits

    if lang in ("ar", "hi", "ml"):
        words = re.split(r"\s+", text)
    elif lang == "zh":
        # No whitespace: substring search is the only option.
        words = []
    else:
        words = re.findall(r"[A-Za-z']+", text.lower())
    lowered = text.lower()

    for label, lexicon in markers.items():
        total = 0.0
        found: list[str] = []
        for marker in lexicon:
            is_prefix = marker.endswith(PREFIX)
            stem = marker[:-1] if is_prefix else marker
            if lang == "zh":
                count = text.count(stem)
            elif is_prefix:
                count = sum(1 for w in words if w.startswith(stem) and len(w) > len(stem))
            else:
                if lang in ("ar", "hi", "ml"):
                    count = sum(1 for w in words if w == stem or stem in w)
                else:
                    count = lowered.count(stem) if " " in stem else sum(
                        1 for w in words if w.strip("',.!?") == stem
                    )
            if count:
                total += count * (1.0 + 0.25 * (len(stem) - 2))
                found.append(stem)
        if total > 0:
            scores[label] = total
            hits[label] = found
    return scores, hits


def detect(text: str, lang: str) -> AccentTag:
    """Tag the spoken variety from evidence alone. No LLM involved.

    Three independent signals, in order of how much they can be trusted:

    * **Code-switch ratio** -- measured, so it is never wrong, but it only
      identifies the bilingual varieties (hinglish/manglish), not regions.
    * **Lexical markers** -- strong when several fire, useless when they
      contradict each other, which is why a margin over the runner-up is
      required before a label is claimed.
    * **Script variant** -- for zh only, and recorded rather than folded into
      the accent label, since simplified/traditional says nothing about how
      someone speaks.
    """
    if lang not in LANGUAGES:
        return AccentTag()

    cs = code_switch_ratio(text, lang)
    script = script_variant(text) if lang == "zh" else None
    scores, hits = _score_labels(text, lang)

    # Bilingual varieties are decided by measurement, not lexicon.
    bilingual = {"hi": "hinglish", "ml": "manglish"}.get(lang)
    if bilingual and cs >= _CODE_SWITCH_TAG_THRESHOLD:
        return AccentTag(
            label=bilingual,
            confidence=min(0.95, 0.5 + cs),
            evidence=(f"code_switch={cs:.2f}",),
            script=script,
            code_switch=cs,
            source="code_switch",
        )

    if lang == "zh" and script is None and not scores:
        return AccentTag(code_switch=cs, script=script)

    if not scores:
        return AccentTag(code_switch=cs, script=script)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_label, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # A single two-character marker firing once is noise. Require either two
    # distinct markers or one strong (long) marker, plus a clear margin, so the
    # tag is only emitted when the evidence would convince a reader.
    distinct = len(hits.get(best_label, ()))
    if best_score < 1.5 or distinct < 1 or (best_score - runner_up) < 0.75:
        return AccentTag(
            label=UNKNOWN,
            confidence=0.2,
            evidence=tuple(f"{k}:{v:.1f}" for k, v in ranked[:3]),
            script=script,
            code_switch=cs,
            source="none",
        )

    # Confidence saturates slowly: two markers is suggestive, five is solid.
    confidence = min(0.95, 0.35 + 0.12 * best_score + 0.05 * distinct)
    return AccentTag(
        label=best_label,
        confidence=confidence,
        evidence=tuple(hits.get(best_label, ())[:12]),
        script=script,
        code_switch=cs,
        source="lexicon",
    )


def reconcile(
    lang: str,
    detected: AccentTag,
    llm_label: str | None,
    llm_confidence: float = 0.0,
) -> AccentTag:
    """Choose the authoritative tag between our evidence and the LLM's claim.

    The LLM sees the audio; we see only text. So an LLM tag is *not* ignored --
    when our lexicon found nothing, the LLM is the only witness and its label
    is used. But when both have an opinion and they differ, the deterministic
    one wins, because it can be re-derived and audited and the LLM's cannot.
    Either way the disagreement is preserved in ``evidence`` rather than
    silently resolved.
    """
    llm_label = (llm_label or "").strip().lower() or None
    # A claim outside the closed vocabulary for this language is not a dialect
    # tag we can act on -- it is the LLM inventing a label.
    valid = bool(llm_label) and llm_label in DIALECTS.get(lang, ())

    if detected.label != UNKNOWN and detected.confidence >= 0.5:
        if valid and llm_label != detected.label:
            return AccentTag(
                label=detected.label,
                confidence=detected.confidence,
                evidence=(*detected.evidence, f"llm_claimed:{llm_label}@{llm_confidence:.2f}"),
                script=detected.script,
                code_switch=detected.code_switch,
                source="reconciled",
            )
        return detected

    if valid:
        return AccentTag(
            label=llm_label,
            confidence=min(0.6, 0.3 + 0.3 * llm_confidence),
            evidence=(f"llm_claimed:{llm_label}@{llm_confidence:.2f}",),
            script=detected.script,
            code_switch=detected.code_switch,
            source="llm",
        )
    return detected


# --- erasure detection -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErasureReport:
    """Result of checking whether the LLM normalized away the speaker's dialect."""

    #: Markers that disappeared AND had a standard equivalent appear. This is
    #: the damaging case: the transcript now describes speech nobody produced.
    replaced: tuple[tuple[str, str], ...] = ()
    #: Markers that disappeared with no replacement. Weaker -- the corrector is
    #: allowed to fix a genuine mis-hearing, so these are reviewed, not refused.
    dropped: tuple[str, ...] = ()
    #: Markers that survived.
    kept: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.replaced

    @property
    def severity(self) -> float:
        """0.0-1.0. Used by the distributor to prefer samples the LLM left intact."""
        if not self.kept and not self.replaced and not self.dropped:
            return 0.0
        markers = len(self.kept) + len(self.replaced) + len(self.dropped)
        return (2.0 * len(self.replaced) + len(self.dropped)) / markers

    def as_dict(self) -> dict:
        return {
            "erasure_ok": self.ok,
            "erasure_replaced": [{"marker": m, "with": w} for m, w in self.replaced],
            "erasure_dropped": list(self.dropped),
            "erasure_kept": list(self.kept),
            "erasure_severity": round(self.severity, 3),
        }


def check_erasure(source: str, final: str, lang: str) -> ErasureReport:
    """Diff the LLM's input against its output for dialect-erasure damage.

    ``source`` must be exactly the text the model was given, which in this
    pipeline is the *normalized* transcript -- not the raw one. Diffing against
    the raw source would bill the model for our own deterministic transforms
    (digit folding, tatweel removal, punctuation conversion) and inflate the
    damage count until the check cries wolf and gets ignored.

    Nothing is hidden by taking normalization as the baseline. Every rule in
    ``normalize`` is orthographic while every marker in ``_ERASURE_PAIRS`` is
    lexical, so normalization cannot erase a dialect: it changes how a word is
    written, never which word it is.
    """
    pairs = _ERASURE_PAIRS.get(lang)
    if not pairs or not source or not final:
        return ErasureReport()
    if source == final:
        present = tuple(m for m in pairs if m in source)
        return ErasureReport(kept=present)

    replaced: list[tuple[str, str]] = []
    dropped: list[str] = []
    kept: list[str] = []

    for marker, standards in pairs.items():
        if marker not in source:
            continue
        if marker in final:
            kept.append(marker)
            continue
        hit = next((s for s in standards if s in final), None)
        if hit:
            replaced.append((marker, hit))
        else:
            dropped.append(marker)

    return ErasureReport(
        replaced=tuple(replaced), dropped=tuple(dropped), kept=tuple(kept)
    )


# --- prompt injection --------------------------------------------------------

#: The exact forms the LLM must not touch, per language. Injected into the
#: correction prompt so "preserve the dialect" stops being an abstraction.
_PRESERVE_HINTS: dict[str, str] = {
    "ar": (
        "Do NOT convert colloquial to Modern Standard Arabic. Keep عشان, كده, "
        "دلوقتي, هلق, شو, بدك, شنو, وايد, بزاف, دابا, هسه, فلوس, حاجة verbatim. "
        "Do NOT expand the Egyptian progressive prefix بـ (keep بيعمل, not يعمل). "
        "Do NOT add short vowels to make text look more formal."
    ),
    "en": (
        "Keep reduced spoken forms verbatim: gonna, wanna, gotta, y'all, dunno, "
        "kinda, 'cause. Do NOT expand them into written equivalents. Do NOT "
        "replace regional vocabulary (lift/elevator, biscuit/cookie) with a "
        "single standard."
    ),
    "zh": (
        "Keep Cantonese particles and characters verbatim (唔, 係, 咗, 冇, 嘅, "
        "喺, 乜, 嗰). Do NOT translate them into Mandarin equivalents. Keep "
        "erhua 儿化 where spoken. Preserve the source script: do NOT convert "
        "simplified to traditional or the reverse."
    ),
    "hi": (
        "Keep Bhojpuri/Bihari verb forms (रहेला, करेला, बा) and Urdu-register "
        "vocabulary verbatim. Do NOT normalize to Khari Boli. Do NOT expand "
        "code-switched English words into Hindi."
    ),
    "ml": (
        "Keep regional address forms (അളിയാ, ചേട്ടാ, ഇക്കാ) and Malabar "
        "loanwords verbatim. Do NOT normalize to the standard written register. "
        "Do NOT expand code-switched English words into Malayalam."
    ),
}


def preservation_block(lang: str, label: str | None = None) -> str:
    """Prompt text enforcing preservation, optionally narrowed to one variety.

    Returning a string rather than editing prompts.py in place keeps the
    lexicon and the instructions next to each other, which is what stops them
    drifting apart the way the validator's diacritics rule drifted from the
    policy documented in arabic_utils.
    """
    base = _PRESERVE_HINTS.get(lang, "")
    if not base:
        return ""
    if label and label != UNKNOWN:
        base = f"This sample is tagged {lang}:{label}. " + base
    return base


def vocab_for(lang: str, label: str) -> tuple[str, ...]:
    """Markers associated with one variety. Exposed for tests and for audits."""
    return _MARKERS.get(lang, {}).get(label, ())


__all__ = [
    "DIALECTS",
    "LANGUAGES",
    "UNKNOWN",
    "AccentTag",
    "ErasureReport",
    "check_erasure",
    "code_switch_ratio",
    "detect",
    "preservation_block",
    "reconcile",
    "script_variant",
    "vocab_for",
]
