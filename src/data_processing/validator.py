"""ValidatorAgent: quality check for cleaned transcripts.

Runs on a sample of processed records to verify quality.
Combines rule-based checks with LLM-based semantic validation.
Includes dialect preservation and diacritics correctness verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .arabic_utils import (
    arabic_ratio,
    classify_dialect,
    contains_arabic,
    count_critical_diacritics,
    diacritics_ratio,
    has_diacritics,
)
from .config import PipelineConfig
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord
from .prompts import build_validator_messages

if TYPE_CHECKING:
    from .corrector import CorrectorAgent


LOGGER = logging.getLogger("data_processing.validator")

# Dialect markers that must be preserved (shared across dialects)
_DIALECT_MARKER_WORDS = {
    "gulf": {"وش", "ليش", "مو", "الحين", "وين", "شنو", "شلون", "يبي", "اشوي"},
    "levantine": {"شو", "هلق", "بدي", "عنجد", "كتير", "هيك", "منيح", "إشي"},
    "egyptian": {"ايه", "ليه", "ازاي", "دلوقتي", "عايز", "خالص", "كده", "بتاع", "حاجة"},
    "maghrebi": {"واش", "علاش", "كيفاش", "دابا", "بغيت", "بزاف", "هاد", "ديال"},
    "iraqi": {"شلون", "اكو", "ماكو", "هذني", "يريد", "گام", "شنو", "مو"},
}

# Over-vocalization threshold: if diacritics/letters ratio exceeds this,
# the text is likely fully vocalized (undesirable for ASR targets)
OVER_VOCALIZATION_THRESHOLD = 0.4

# Max validator correction rounds before settling (prevents correction
# ping-pong: reject → correct → re-reject → correct → ... loops)
MAX_CORRECTION_ROUNDS = 2

# If still "invalid" after corrections but the LLM quality score is at
# least this, accept: residual issues at this score are stylistic nits,
# not fidelity failures
BORDERLINE_QUALITY_THRESHOLD = 0.7


@dataclass
class ValidationResult:
    """Result of validating a cleaned record."""

    is_valid: bool
    issues: list[str]
    corrected_text: str | None = None
    quality_score: float = 0.0
    llm_validated: bool = False


class ValidatorAgent:
    """Validates cleaned transcripts using rule-based and LLM checks."""

    def __init__(
        self,
        client: VLLMClient,
        max_retries: int = 2,
        config: PipelineConfig | None = None,
        corrector: CorrectorAgent | None = None,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self.config = config
        self.language = "ar"
        self.corrector = corrector
        self.preserve_dialects = config.preserve_dialects if config else True
        self.restore_diacritics = config.restore_diacritics if config else True
        self.thinking = config.validator_thinking if config else False

    async def validate(self, record: CleanedRecord) -> ValidationResult:
        """Validate a cleaned record.

        Hard failures (empty, no Arabic, hallucination-scale length change)
        reject immediately. Soft heuristic flags are passed to the LLM
        validator as advisory hints — the LLM makes the final call.
        """
        # Hard failures: objective, mechanical — no LLM needed
        hard_issues = self._hard_checks(record)
        if hard_issues:
            return ValidationResult(is_valid=False, issues=hard_issues)

        # Soft flags: weak heuristics — advisory only, LLM adjudicates
        soft_flags = self._soft_flags(record)

        try:
            llm_result = await self._llm_validate(record, auto_flags=soft_flags)
            # The LLM verdict is final — heuristics do not override it
            return ValidationResult(
                is_valid=llm_result.is_valid,
                issues=llm_result.issues,
                corrected_text=llm_result.corrected_text,
                quality_score=llm_result.quality_score,
                llm_validated=True,
            )
        except Exception as exc:
            LOGGER.warning("LLM validation failed: %s", exc)
            # LLM unavailable: fall back to conservative heuristic verdict
            return ValidationResult(
                is_valid=len(soft_flags) == 0,
                issues=soft_flags,
                llm_validated=False,
            )

    def _hard_checks(self, record: CleanedRecord) -> list[str]:
        """Objective failures that reject without LLM involvement."""
        issues = []

        # Cleaned text must not be empty
        if not record.text or not record.text.strip():
            issues.append("CRITICAL: Cleaned text is empty")
            return issues

        # Cleaned text must contain Arabic
        if not contains_arabic(record.text):
            issues.append("CRITICAL: Cleaned text contains no Arabic characters")

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

        return issues

    def _soft_flags(self, record: CleanedRecord) -> list[str]:
        """Weak heuristic signals passed to the LLM validator as hints.

        These never reject a record on their own — the LLM adjudicates.
        """
        flags = []

        # Arabic ratio (soft: code-switched speech can legitimately dip)
        ratio = arabic_ratio(record.text)
        if ratio < 0.8:
            flags.append(f"Arabic ratio is {ratio:.2f} (expected > 0.80)")

        # Cleaner's own confidence
        if record.confidence < 0.3:
            flags.append(f"Cleaner reported low confidence: {record.confidence:.2f}")

        # Dialect marker heuristic (advisory — marker lists are incomplete)
        if self.preserve_dialects:
            flags.extend(self._check_dialect_preservation(record))

        # Diacritics heuristics
        flags.extend(self._check_diacritics(record))

        return flags

    def _check_dialect_preservation(self, record: CleanedRecord) -> list[str]:
        """Advisory heuristic: detect potentially lost dialect markers.

        NOTE: marker lists are incomplete and overlap across dialects
        (e.g., شلون/مو are both Gulf and Iraqi). Findings are hints for
        the LLM validator, never rejection grounds on their own.
        """
        issues = []
        original_dialect = classify_dialect(record.original_text)

        # Only check if original has a detectable non-MSA dialect
        if original_dialect in ("msa", "unknown"):
            return issues

        # Check that dialect markers are preserved in cleaned text
        markers = _DIALECT_MARKER_WORDS.get(original_dialect, set())
        original_words = set(record.original_text.split())
        cleaned_words = set(record.text.split())

        # Find markers present in original
        original_markers = original_words & markers
        if not original_markers:
            return issues

        # Check how many are preserved
        lost_markers = original_markers - cleaned_words
        if lost_markers and len(lost_markers) >= len(original_markers) * 0.5:
            issues.append(
                f"Dialect not preserved: {original_dialect} markers lost: "
                f"{', '.join(sorted(lost_markers))}"
            )

        # Cross-check: if the cleaned text lost all dialectal reading
        cleaned_dialect = classify_dialect(record.text)
        if (
            cleaned_dialect != original_dialect
            and cleaned_dialect == "msa"
            and original_dialect != "msa"
        ):
            issues.append(
                f"Possible MSA normalization: original reads as {original_dialect}, "
                f"cleaned reads as MSA"
            )

        return issues

    def _check_diacritics(self, record: CleanedRecord) -> list[str]:
        """Verify diacritics handling per config.

        When restore_diacritics=True:
        - Flag over-vocalization (full tashkeel is undesirable)
        - For longer texts, expect at least some critical diacritics

        When restore_diacritics=False:
        - Flag if diacritics were added when they shouldn't be
        """
        issues = []

        if self.restore_diacritics:
            # Check for over-vocalization
            d_ratio = diacritics_ratio(record.text)
            if d_ratio > OVER_VOCALIZATION_THRESHOLD:
                issues.append(
                    f"Over-vocalized: diacritics ratio {d_ratio:.2f} exceeds "
                    f"threshold {OVER_VOCALIZATION_THRESHOLD} (full tashkeel detected)"
                )

            # For longer texts, expect at least some critical diacritics
            # (shadda/tanween should appear in most Arabic text > 50 chars)
            if len(record.text) > 50 and has_diacritics(record.text):
                critical_count = count_critical_diacritics(record.text)
                # If text has diacritics but zero critical ones, the LLM
                # may have added only non-essential harakat
                if critical_count == 0:
                    issues.append(
                        "Diacritics present but no critical marks "
                        "(shadda/tanween) — likely non-essential vocalization"
                    )
        else:
            # Diacritics should NOT have been added
            if has_diacritics(record.text) and not has_diacritics(record.original_text):
                issues.append(
                    "Diacritics added despite restore_diacritics=False"
                )

        return issues

    async def _llm_validate(
        self, record: CleanedRecord, auto_flags: list[str] | None = None
    ) -> ValidationResult:
        """Use the LLM to validate semantic correctness.

        Heuristic soft flags are included in the prompt as advisory hints
        for the LLM to confirm or dismiss.
        """
        messages = build_validator_messages(
            original=record.original_text,
            processed=record.text,
            dialect=record.dialect,
            changes=record.changes,
            auto_flags=auto_flags,
        )

        result = await self.client.chat_completion_json(
            messages, thinking=self.thinking
        )

        is_valid = result.get("valid", True)
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

    async def validate_with_retry(
        self, record: CleanedRecord
    ) -> tuple[CleanedRecord, bool]:
        """Validate and potentially correct a record.

        Returns (final_record, was_accepted).
        Stage 1 applies the validator's own inline corrections for up to
        MAX_CORRECTION_ROUNDS. Stage 2 (if a corrector is attached and the
        record is still invalid) runs one issue-driven repair targeting the
        flagged issues, then re-validates. Finally settles: accept if valid
        or if the quality score clears the borderline threshold, otherwise
        reject. Never re-validates unchanged text (a corrected_text identical
        to the current text would loop without progress).
        """
        result = await self.validate(record)

        rounds = 0
        while (
            not result.is_valid
            and result.corrected_text
            and result.corrected_text != record.text
            and rounds < MAX_CORRECTION_ROUNDS
        ):
            record = CleanedRecord(
                audio_filepath=record.audio_filepath,
                text=result.corrected_text,
                duration=record.duration,
                original_text=record.original_text,
                dialect=record.dialect,
                confidence=record.confidence * 0.9,  # Reduce confidence
                changes=record.changes + ["corrected"],
            )
            rounds += 1
            result = await self.validate(record)

        # Stage 2: issue-driven repair. The loop above only fires when the
        # validator volunteers a corrected_text; many rejects are "invalid
        # with issues but no correction offered" (rounds=0). Hand those to a
        # dedicated corrector that repairs the SPECIFIC flagged issues, then
        # re-validate once. Excludes hard-check failures (llm_validated is
        # False) — those need re-cleaning from source, not a formatting patch.
        corrected_once = False
        if (
            self.corrector is not None
            and not result.is_valid
            and result.llm_validated
            and result.issues
        ):
            try:
                repaired = await self.corrector.correct(record, result.issues)
            except Exception as exc:
                LOGGER.warning(
                    "Corrector failed for %s: %s", record.audio_filepath, exc
                )
                repaired = None
            if repaired and repaired != record.text:
                record = CleanedRecord(
                    audio_filepath=record.audio_filepath,
                    text=repaired,
                    duration=record.duration,
                    original_text=record.original_text,
                    dialect=record.dialect,
                    confidence=record.confidence * 0.9,  # Reduce confidence
                    changes=record.changes + ["issue_corrected"],
                )
                corrected_once = True
                result = await self.validate(record)

        if result.is_valid:
            return record, True

        # Borderline acceptance: the LLM still lists issues but scores the
        # record decently — residual nits, not fidelity failures. Hard-check
        # rejections never reach here (llm_validated is False for them).
        if (
            result.llm_validated
            and result.quality_score >= BORDERLINE_QUALITY_THRESHOLD
        ):
            LOGGER.debug(
                "Accepting borderline record (score %.2f): %s",
                result.quality_score,
                result.issues,
            )
            record = CleanedRecord(
                audio_filepath=record.audio_filepath,
                text=record.text,
                duration=record.duration,
                original_text=record.original_text,
                dialect=record.dialect,
                confidence=record.confidence,
                changes=record.changes + ["borderline"],
            )
            return record, True

        LOGGER.warning(
            "Record rejected after %d correction rounds%s: %s",
            rounds,
            " + issue-correction" if corrected_once else "",
            result.issues,
        )
        return record, False


__all__ = ["ValidationResult", "ValidatorAgent"]
