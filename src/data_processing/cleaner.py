"""CleanerAgent: full LLM pass for transcript cleaning.

Performs cleaning, ITN, punctuation, dialect preservation, and diacritics
restoration via one focused LLM call per transcript. Records in a batch
are processed concurrently — vLLM's continuous batching handles scheduling.
"""

from __future__ import annotations

import asyncio
import logging

from .arabic_utils import contains_arabic, preprocess_text
from .config import PipelineConfig
from .itn import normalize_digits_to_western
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord, ManifestRecord
from .prompts import build_cleaner_messages


LOGGER = logging.getLogger("data_processing.cleaner")


class CleanerAgent:
    """Full LLM pass: clean + ITN + punctuate + dialect-preserve + diacritics."""

    def __init__(self, client: VLLMClient, config: PipelineConfig | None = None) -> None:
        self.client = client
        self.config = config
        # Extract processing flags from config (defaults: all enabled)
        self.preserve_dialects = config.preserve_dialects if config else True
        self.restore_diacritics = config.restore_diacritics if config else True

    def _preprocess(self, text: str) -> str:
        """Fast local pre-processing hook (overridden per language family)."""
        return preprocess_text(text)

    async def process_single(self, record: ManifestRecord) -> CleanedRecord:
        """Process a single transcript through the LLM."""
        # Pre-process locally to reduce token waste
        preprocessed = preprocess_text(record.text)

        # Short-circuit: trivial/non-Arabic inputs don't need an LLM pass.
        # They will be caught by the validator's hard checks downstream.
        # Distinct tags so rejected files are triageable: "skipped_short"
        # (near-empty text) vs "skipped_wrong_script" (language contamination).
        if len(preprocessed) < 3 or not contains_arabic(preprocessed):
            skip_tag = (
                "skipped_short" if len(preprocessed) < 3 else "skipped_wrong_script"
            )
            return CleanedRecord(
                audio_filepath=record.audio_filepath,
                text=preprocessed,
                duration=record.duration,
                original_text=record.text,
                dialect="unknown",
                confidence=0.0,
                changes=[skip_tag],
            )

        # Build messages and call LLM (adapts prompt to config flags)
        messages = build_cleaner_messages(
            preprocessed,
            preserve_dialects=self.preserve_dialects,
            restore_diacritics=self.restore_diacritics,
        )
        result = await self.client.chat_completion_json(messages)

        # Parse and validate result
        cleaned_text = result.get("text", preprocessed)
        dialect = result.get("dialect", "unknown")
        confidence = float(result.get("confidence", 0.5))
        changes = result.get("changes", [])

        # Post-process: enforce digit/punctuation charset deterministically.
        # Western digits are the target convention — normalize ALL Arabic-Indic
        # digits (not just mixed cases) and the Urdu full stop (۔ → .)
        normalized = normalize_digits_to_western(cleaned_text).replace("۔", ".")
        if normalized != cleaned_text:
            cleaned_text = normalized
            if "itn" not in changes:
                changes.append("itn")

        return CleanedRecord(
            audio_filepath=record.audio_filepath,
            text=cleaned_text,
            duration=record.duration,
            original_text=record.text,
            dialect=dialect,
            confidence=confidence,
            changes=changes,
        )

    async def process_batch(
        self, records: list[ManifestRecord]
    ) -> list[CleanedRecord]:
        """Process a batch of transcripts concurrently.

        Each record gets its own focused LLM call (highest quality — no
        attention splitting across unrelated transcripts). Concurrency
        comes from asyncio.gather; vLLM's continuous batching schedules
        the parallel requests efficiently on the GPU.

        Failed records fall back to their preprocessed original text.
        """
        if not records:
            return []

        results = await asyncio.gather(
            *(self.process_single(r) for r in records),
            return_exceptions=True,
        )

        cleaned_records: list[CleanedRecord] = []
        for record, result in zip(records, results):
            if isinstance(result, BaseException):
                LOGGER.error(
                    "Failed to process record at line %d: %s",
                    record.line_number,
                    result,
                )
                # Fallback: preprocessed original text, flagged for review
                cleaned_records.append(
                    CleanedRecord(
                        audio_filepath=record.audio_filepath,
                        text=self._preprocess(record.text),
                        duration=record.duration,
                        original_text=record.text,
                        dialect="unknown",
                        confidence=0.0,
                        changes=["error_fallback"],
                    )
                )
            else:
                cleaned_records.append(result)

        return cleaned_records


__all__ = ["CleanerAgent"]
