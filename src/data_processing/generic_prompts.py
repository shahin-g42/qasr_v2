"""Prompt templates for the generic (non-Arabic) multilingual cleaner.

One shared prompt structure for all non-Arabic languages (en, zh, hi, ml, ...),
parameterized by language. Language-specific conventions (punctuation, ITN,
script) are injected as guidance blocks.
"""

from __future__ import annotations

from .accent import preservation_block
from .text_utils import LANGUAGE_NAMES

# Per-language conventions injected into the system prompt.
# Falls back to _DEFAULT_CONVENTIONS for unlisted languages.
_LANGUAGE_CONVENTIONS: dict[str, str] = {
    "en": """\
Language conventions (English):
- Punctuation: standard English (. , ? ! ; : "quotes"). Sentence-case \
capitalization; capitalize proper nouns and "I".
- ITN: spelled-out numbers to digits ("twenty five" -> "25", \
"three thousand" -> "3000"). Keep idiomatic ordinals in words when natural \
("first of all"). Format times ("three thirty pm" -> "3:30 PM"), dates \
("March twenty first twenty twenty four" -> "March 21, 2024"), \
currencies ("fifty dollars" -> "$50"), and percentages ("ten percent" -> "10%").
- Keep contractions as spoken ("don't", "it's") — do not expand them.""",
    "zh": """\
Language conventions (Chinese, Mandarin):
- Punctuation: full-width Chinese punctuation only (\u3002 \uff0c \uff1f \uff01 \uff1b \uff1a \u3001 \
\u201c\u201d). Never use half-width . , ? in Chinese sentence flow.
- ITN: spoken numbers to Arabic numerals ("\u4e09\u5343" -> "3000", \
"\u767e\u5206\u4e4b\u4e94\u5341" -> "50%"), but keep idiomatic/lexicalized number words \
("\u4e00\u4e9b", "\u4e00\u8d77", "\u5341\u5206") untouched.
- Times and dates in numerals, keeping the 年/月/日 markers \
("三点半" -> "3:30", "二〇二四年三月二十一日" -> "2024年3月21日").
- No spaces between Chinese characters; keep a single space around \
embedded Latin words or numbers where standard.
- Preserve erhua (\u513f\u5316) and regional colloquialisms as spoken.""",
    "hi": """\
Language conventions (Hindi):
- Punctuation: danda (\u0964) as sentence terminator, ? for questions, \
standard commas. Keep Devanagari script.
- ITN: spelled-out numbers to Western digits ("\u092a\u091a\u093e\u0938" -> "50", \
"\u0924\u0940\u0928 \u0939\u091c\u093c\u093e\u0930" -> "3000"). Convert Devanagari digits (\u0967\u0968\u0969) to Western (123).
- Times and dates in digits, month name as spoken \
("साढ़े तीन बजे" -> "3:30 बजे", "इक्कीस मार्च" -> "21 मार्च").
- Preserve Hinglish code-switching exactly as spoken — keep English \
words in Latin script, do not translate or transliterate them.""",
    "ml": """\
Language conventions (Malayalam):
- Punctuation: standard (. , ? !). Keep Malayalam script.
- ITN: spelled-out numbers to Western digits ("\u0d05\u0d3e\u0d2f\u0d3f\u0d30\u0d02" -> "1000", \
"\u0d05\u0d2e\u0d4d\u0d2a\u0d24\u0d4d" -> "50").
- Times and dates in digits, month name as spoken \
("3:30", "2024 മാർച്ച് 21").
- Preserve Manglish code-switching exactly as spoken — keep English \
words in Latin script, do not translate or transliterate them.""",
}

_DEFAULT_CONVENTIONS = """\
Language conventions:
- Use the standard punctuation of the language.
- Convert spelled-out numbers to digits where natural; keep idiomatic \
number expressions in words.
- Preserve code-switching exactly as spoken."""

# Date/time ITN carries the same hazard in every language: the spoken form is
# the only source of truth, so whatever the speaker left out cannot be filled
# in from context. Appended to every conventions block rather than restated
# per language.
_DATETIME_GUARDRAIL = """\
- Dates and times: digits for the fields actually spoken, month name and \
clock reading kept as spoken. Never infer an absent year, century, or \
AM/PM, never reorder date fields, and never convert between calendar \
systems."""


def _conventions_for(language: str) -> str:
    """Per-language conventions plus the shared date/time guardrail."""
    base = _LANGUAGE_CONVENTIONS.get(language, _DEFAULT_CONVENTIONS)
    return f"{base}\n{_DATETIME_GUARDRAIL}"


GENERIC_CLEANER_USER_TEMPLATE = """\
Process this {language_name} ASR transcript with the highest linguistic \
quality. Preserve the speaker's natural voice:
<<<{transcript}>>>
"""

GENERIC_CLEANER_BATCH_USER_TEMPLATE = """\
Process each {language_name} ASR transcript below with the highest linguistic \nquality. Preserve each speaker's accent, dialect, and natural voice. Return \na JSON array with one result object per transcript, in the same order, using \neach transcript's number (0-based) as "i".

{numbered_transcripts}

Output STRICT JSON array (no markdown fences, no commentary, no thinking):
[{{"i": 0, "text": "...", "dialect": "...", "confidence": 0.0}}, ...]
"""

GENERIC_VALIDATOR_USER_TEMPLATE = """\
Original ASR transcript:
<<<{original}>>>

Processed transcript:
<<<{processed}>>>

Language: {language_name}
Reported changes: {changes}
{auto_flags_section}
Perform a thorough quality review. Check every word and every punctuation \
mark. If anything is wrong, provide the corrected text.
"""

_AUTO_FLAGS_TEMPLATE = """\

Automated heuristic checks flagged the following (these may be false \
positives — adjudicate each one with your linguistic expertise):
{flags}
"""


def build_generic_cleaner_system_prompt(language: str) -> str:
    """Build the cleaner system prompt for a non-Arabic language."""
    language_name = LANGUAGE_NAMES.get(language, language)
    conventions = _conventions_for(language)

    return (
        f"You are a world-class {language_name} computational linguist "
        "specializing in ASR transcript post-processing for speech-to-text "
        "training data.\n\n"
        f"Your task: transform raw ASR output into publication-quality "
        f"{language_name} text that is natural, elegant, and linguistically "
        "precise — while faithfully preserving the speaker's voice.\n\n"
        "Rules:\n"
        "0. VERBATIM FIDELITY (overrides all rules below): NEVER substitute, "
        "translate, paraphrase, reorder, add, or drop spoken words — this is "
        "ASR training data, the text must match the audio word-for-word. "
        "Repeated words the speaker said stay. Only orthography, punctuation, "
        "digits, and artifact removal are allowed edits.\n"
        "1. CLEANING: Remove all artifacts: HTML tags, encoding errors, "
        "repeated characters, filler markers ([noise], [music], [inaudible]). "
        "Fix obvious ASR misrecognitions (homophones, split/merged words). "
        "Remove speaker labels, timestamps, and metadata. Remove stuttering "
        "artifacts but preserve meaningful repetitions (emphasis).\n"
        "2. INVERSE TEXT NORMALIZATION (ITN): Convert spoken-form numbers, "
        "times, dates, currencies, and percentages to written form per the "
        "language conventions below.\n"
        "3. PUNCTUATION: Add punctuation based on syntactic and prosodic "
        "boundaries — meaning units and breath groups, not mechanically. "
        "Ensure complete sentences are properly terminated; if the audio "
        "cuts off mid-sentence, keep the transcript incomplete and end it "
        "with \"...\" — NEVER invent words to finish it.\n"
        "4. LANGUAGE FIDELITY: NEVER translate. Preserve the speaker's "
        "register, colloquialisms, and code-switching exactly as spoken. "
        "Spoken fillers and disfluencies (\"uh\", \"uhm\", \"hm\", false "
        "starts, repetitions) exist in the audio — KEEP them. "
        "Promote uncommon/ad-hoc native-script transliterations of foreign "
        "words back to their original script (same spoken word); "
        "established loanwords keep their conventional native spelling.\n\n"
        f"{conventions}\n\n"
        "Quality standards:\n"
        "- Formatting (spelling, punctuation, casing) is publication-quality; "
        "the WORDS are the speaker's, however casual or disfluent.\n"
        "- Preserve the speaker's voice, register, and naturalness.\n"
        "- Fidelity to the spoken words ALWAYS outweighs elegance.\n"
        "- When uncertain about a correction, preserve the original form.\n\n"
        "Output STRICT JSON (no markdown fences, no commentary, no thinking):\n"
        '{"text": "...", "confidence": 0.0-1.0, '
        '"changes": ["clean", "itn", "punct"]}\n'
    )


def build_generic_validator_system_prompt(language: str) -> str:
    """Build the validator system prompt for a non-Arabic language."""
    language_name = LANGUAGE_NAMES.get(language, language)
    conventions = _conventions_for(language)

    return (
        f"You are a senior {language_name} linguistics QA reviewer "
        "specializing in ASR training data quality. You review processed "
        "transcripts with extreme attention to detail.\n\n"
        "Validation checklist:\n"
        "1. SEMANTIC FIDELITY: No meaning was altered, added, or "
        "hallucinated. The processed text conveys exactly what the speaker "
        "said. Nothing was translated.\n"
        "2. ITN CORRECTNESS: All number/time/currency conversions are "
        "accurate and follow the language conventions below. An impossible "
        "date or time field (hour above 23, month above 12) is mis-assembled "
        "ITN — correct it.\n"
        "3. PUNCTUATION QUALITY: Punctuation is syntactically valid, uses "
        "the correct character set for the language, and reflects natural "
        "prosodic boundaries.\n"
        "4. NATURALNESS: The formatting reads naturally — judge the "
        "formatting, NOT the speaker. Disfluent, casual, or profane speech "
        "faithfully transcribed is natural.\n"
        "5. ORTHOGRAPHY: Correct spelling, casing, and script conventions. "
        "Code-switched words remain in their original script.\n\n"
        f"{conventions}\n\n"
        "Adjudication principles (apply BEFORE flagging issues):\n"
        "- ITN conversion to digits is the EXPECTED output format, NOT a "
        "fidelity violation. A spoken number left in words is at most a "
        "minor issue — correct it, do not invalidate.\n"
        "- Promoting an uncommon native-script transliteration of a foreign "
        "word back to its original script is an allowed orthography edit of "
        "the SAME spoken word — NOT a fidelity violation. Established "
        "loanwords keep their conventional native spelling.\n"
        "- The speaker's exact words are ground truth. If the speaker "
        "uttered a grammatical error or awkward phrasing, the transcript "
        "must KEEP it — matching the audio outweighs grammatical "
        "correctness and naturalness.\n"
        "- Fillers and disfluencies (\"uhm\", \"hm\", repetitions, false "
        "starts) in the transcript are CORRECT — they exist in the audio. "
        "Never flag them.\n"
        "- A transcript that ends mid-sentence reflects audio segmentation, "
        "NOT an error. Never require completion or reject a truncated final "
        "utterance.\n"
        "- Content (profanity, adult topics) is never a validity "
        "criterion.\n\n"
        "If you find issues, provide a corrected version that fixes them "
        "while maintaining the speaker's voice and naturalness. The "
        "corrected_text must reuse the ORIGINAL transcript's exact spoken "
        "words — never introduce substitutions or paraphrases of your own.\n\n"
        "Minor stylistic imperfections (punctuation placement, casing) do "
        'NOT make a transcript invalid — reserve "valid": false for real '
        "fidelity or correctness failures.\n\n"
        "Keep each issue to ONE short phrase (max 10 words) — e.g. "
        '"hallucinated word X", "wrong number conversion". Do NOT write explanations.\n\n'
        "Output STRICT JSON (no markdown fences, no commentary, no thinking):\n"
        '{"valid": true|false, "issues": ["issue1", ...], '
        '"corrected_text": "..." or null, "quality_score": 0.0-1.0}\n'
    )


def build_generic_cleaner_messages(
    transcript: str, language: str
) -> list[dict[str, str]]:
    """Build chat messages for a single-transcript cleaning request."""
    language_name = LANGUAGE_NAMES.get(language, language)
    return [
        {
            "role": "system",
            "content": build_generic_cleaner_system_prompt(language),
        },
        {
            "role": "user",
            "content": GENERIC_CLEANER_USER_TEMPLATE.format(
                language_name=language_name, transcript=transcript
            ),
        },
    ]


def build_generic_batch_cleaner_messages(
    transcripts: list[str],
    language: str,
    accent_label: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for a batch cleaning request.

    Mirrors the Arabic ``build_batch_cleaner_messages``: a numbered ``<<<t>>>``
    list whose 0-based numbers are the ``i`` the batch corrector's parser
    reads back, so the rich-prompt path and the result contract cannot drift
    apart. ``accent_label`` appends the dialect-specific preservation block so
    a (language, accent)-grouped batch gets instructions that fit it.
    """
    language_name = LANGUAGE_NAMES.get(language, language)
    numbered = "\n".join(f"{i}. <<<{t}>>>" for i, t in enumerate(transcripts))
    system = build_generic_cleaner_system_prompt(language)
    block = preservation_block(language, accent_label)
    if block:
        system = system.rstrip() + "\n\n" + block
    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": GENERIC_CLEANER_BATCH_USER_TEMPLATE.format(
                language_name=language_name, numbered_transcripts=numbered
            ),
        },
    ]


def build_generic_validator_messages(
    original: str,
    processed: str,
    language: str,
    changes: list[str],
    auto_flags: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for a validation request.

    Optional heuristic auto_flags are embedded as advisory hints for
    the LLM to confirm or dismiss — they are not verdicts.
    """
    language_name = LANGUAGE_NAMES.get(language, language)
    if auto_flags:
        flags_text = "\n".join(f"- {flag}" for flag in auto_flags)
        auto_flags_section = _AUTO_FLAGS_TEMPLATE.format(flags=flags_text)
    else:
        auto_flags_section = ""

    return [
        {
            "role": "system",
            "content": build_generic_validator_system_prompt(language),
        },
        {
            "role": "user",
            "content": GENERIC_VALIDATOR_USER_TEMPLATE.format(
                original=original,
                processed=processed,
                language_name=language_name,
                changes=", ".join(changes) if changes else "none",
                auto_flags_section=auto_flags_section,
            ),
        },
    ]


__all__ = [
    "GENERIC_CLEANER_BATCH_USER_TEMPLATE",
    "GENERIC_CLEANER_USER_TEMPLATE",
    "GENERIC_VALIDATOR_USER_TEMPLATE",
    "build_generic_batch_cleaner_messages",
    "build_generic_cleaner_messages",
    "build_generic_cleaner_system_prompt",
    "build_generic_validator_messages",
    "build_generic_validator_system_prompt",
]
