"""GenericValidatorAgent: quality check for non-Arabic cleaned transcripts.

One shared implementation for all non-Arabic languages (en, zh, hi, ml, ...),
parameterized by language. Same architecture as the Arabic ValidatorAgent:
hard failures reject immediately, soft heuristic flags are passed to the
LLM validator as advisory hints, and the LLM verdict is final.

Inherits validate() and validate_with_retry() from ValidatorAgent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import PipelineConfig
from .generic_prompts import build_generic_validator_messages
from .itn import numeric_soft_flags
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord
from .text_utils import (
    check_codeswitch_preservation,
    contains_expected_script,
    devanagari_to_western,
    expected_script_ratio,
)
from .validator import ValidationResult, ValidatorAgent

if TYPE_CHECKING:
    from .corrector import CorrectorAgent


LOGGER = logging.getLogger("data_processing.generic_validator")

# Brahmic script blocks — LLMs occasionally emit cross-script homoglyphs
# (e.g. Gurmukhi ੀ U+0A40 for Devanagari ी U+0940 inside नहीं), which read
# fine visually but corrupt training text. Detected against the ORIGINAL:
# only characters the cleaner introduced count.
_INDIC_BLOCKS = {
    "devanagari": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "gurmukhi": ("\u0a00", "\u0a7f"),
    "gujarati": ("\u0a80", "\u0aff"),
    "oriya": ("\u0b00", "\u0b7f"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "kannada": ("\u0c80", "\u0cff"),
    "malayalam": ("\u0d00", "\u0d7f"),
}


def _foreign_indic_chars(text: str, original: str) -> dict[str, set[str]]:
    """Indic-script chars present in text but absent from original, by block."""
    introduced: dict[str, set[str]] = {}
    original_chars = set(original)
    for ch in set(text) - original_chars:
        for name, (lo, hi) in _INDIC_BLOCKS.items():
            if lo <= ch <= hi:
                introduced.setdefault(name, set()).add(ch)
                break
    return introduced


class GenericValidatorAgent(ValidatorAgent):
    """Validates non-Arabic cleaned transcripts (language-parameterized)."""

    def __init__(
        self,
        client: VLLMClient,
        max_retries: int = 2,
        config: PipelineConfig | None = None,
        language: str = "en",
        corrector: CorrectorAgent | None = None,
    ) -> None:
        super().__init__(
            client, max_retries=max_retries, config=config, corrector=corrector
        )
        self.language = language

    def _hard_checks(self, record: CleanedRecord) -> list[str]:
        """Objective failures that reject without LLM involvement."""
        issues = []

        # Cleaned text must not be empty
        if not record.text or not record.text.strip():
            issues.append("CRITICAL: Cleaned text is empty")
            return issues

        # Cleaned text must contain the expected script
        if not contains_expected_script(record.text, self.language):
            issues.append(
                f"CRITICAL: Cleaned text contains no {self.language} "
                "script characters"
            )

        # Hallucination-scale length changes
        original_len = len(record.original_text)
        cleaned_len = len(record.text)
        if original_len > 20 and cleaned_len < original_len * 0.3:
            issues.append(
                f"Text drastically shortened: {original_len} -> {cleaned_len} chars"
            )
        if original_len > 0 and cleaned_len > original_len * 3:
            issues.append(
                f"Text drastically expanded: {original_len} -> {cleaned_len} chars"
            )

        # Cross-script homoglyph injection: the cleaner must never introduce
        # characters from a different Indic script (e.g. Gurmukhi matra
        # inside Hindi text). Legit code-switching survives because chars
        # already present in the original are excluded.
        expected_block = {"hi": "devanagari", "ml": "malayalam"}.get(self.language)
        for block, chars in _foreign_indic_chars(
            record.text, record.original_text
        ).items():
            if block != expected_block:
                issues.append(
                    f"CRITICAL: Cleaner introduced {block} script "
                    f"character(s): {' '.join(sorted(chars))}"
                )

        return issues

    def _soft_flags(self, record: CleanedRecord) -> list[str]:
        """Weak heuristic signals passed to the LLM validator as hints.

        These never reject a record on their own — the LLM adjudicates.
        """
        flags = []

        # Expected script ratio. Code-switching does NOT dip this, because
        # expected_script_ratio() counts Latin as expected for every
        # non-English language — a low value means a third script appeared.
        ratio = expected_script_ratio(record.text, self.language)
        if ratio < 0.8:
            flags.append(
                f"Expected-script ratio is {ratio:.2f} (expected > 0.80)"
            )

        # Cleaner's own confidence
        if record.confidence < 0.3:
            flags.append(f"Cleaner reported low confidence: {record.confidence:.2f}")

        # Code-switching: Hinglish and Manglish carry English in Latin script,
        # and Chinese transcripts embed Latin brand/technical terms. Skipped
        # for English, where every token is Latin and a lost word is an
        # ordinary omission for the LLM to judge, not a script violation.
        if self.language != "en":
            flags.extend(
                check_codeswitch_preservation(record.text, record.original_text)
            )

        # ITN verification. Only the script-independent (digit-only) checks
        # apply here — the Arabic set inspects Arabic number words.
        if self.itn_enabled:
            flags.extend(numeric_soft_flags(record.text, record.original_text))

        return flags

    def _normalize_corrected(self, text: str) -> str:
        """Mirror GenericCleanerAgent's post-processing, not the Arabic one.

        The inherited implementation enforces the Arabic charset, which would
        rewrite digits in every other language's output.
        """
        if self.language == "hi":
            return devanagari_to_western(text)
        return text

    async def _llm_validate(
        self, record: CleanedRecord, auto_flags: list[str] | None = None
    ) -> ValidationResult:
        """Use the LLM to validate semantic correctness (language-aware)."""
        messages = build_generic_validator_messages(
            original=record.original_text,
            processed=record.text,
            language=self.language,
            changes=record.changes,
            auto_flags=auto_flags,
        )

        result = await self.client.chat_completion_json(
            messages, thinking=self.thinking
        )

        # Fail CLOSED: a parseable-but-schema-broken verdict (missing
        # "valid" key) must never auto-accept — it goes through the
        # retry/correction loop and is rejected if it cannot settle.
        is_valid = result.get("valid", False)
        issues = result.get("issues", [])
        corrected_text = result.get("corrected_text")
        quality_score = float(result.get("quality_score", 0.0))

        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        return ValidationResult(
            is_valid=bool(is_valid),
            issues=issues,
            corrected_text=corrected_text if corrected_text else None,
            quality_score=quality_score,
            llm_validated=True,
        )


__all__ = ["GenericValidatorAgent"]
