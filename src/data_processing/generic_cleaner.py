"""GenericCleanerAgent: LLM cleaning pass for non-Arabic languages.

One shared implementation for all non-Arabic languages (en, zh, hi, ml, ...):
cleaning, ITN, and punctuation via one focused LLM call per transcript,
parameterized by language. Arabic uses the specialized CleanerAgent
(dialect preservation, diacritics restoration, Arabic ITN).

Inherits the concurrent process_batch machinery from CleanerAgent.
"""

from __future__ import annotations

import logging

from .cleaner import CleanerAgent
from .config import PipelineConfig
from .generic_prompts import build_generic_cleaner_messages
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord, ManifestRecord
from .text_utils import (
    contains_expected_script,
    devanagari_to_western,
    preprocess_text_generic,
)

LOGGER = logging.getLogger("data_processing.generic_cleaner")


class GenericCleanerAgent(CleanerAgent):
    """Language-parameterized LLM pass: clean + ITN + punctuate."""

    def __init__(
        self,
        client: VLLMClient,
        config: PipelineConfig | None = None,
        language: str = "en",
    ) -> None:
        super().__init__(client, config=config)
        self.language = language

    def _preprocess(self, text: str) -> str:
        """Language-agnostic pre-processing (no Arabic-specific steps)."""
        return preprocess_text_generic(text)

    async def process_single(self, record: ManifestRecord) -> CleanedRecord:
        """Process a single transcript through the LLM."""
        # Pre-process locally to reduce token waste
        preprocessed = preprocess_text_generic(record.text)

        # Short-circuit: trivial/wrong-script inputs don't need an LLM pass.
        # They will be caught by the validator's hard checks downstream.
        # Distinct tags so rejected files are triageable: "skipped_short"
        # (near-empty text) vs "skipped_wrong_script" (language contamination).
        if len(preprocessed) < 3 or not contains_expected_script(
            preprocessed, self.language
        ):
            skip_tag = (
                "skipped_short" if len(preprocessed) < 3 else "skipped_wrong_script"
            )
            return CleanedRecord(
                audio_filepath=record.audio_filepath,
                text=preprocessed,
                duration=record.duration,
                original_text=record.text,
                dialect=self.language,
                confidence=0.0,
                changes=[skip_tag],
            )

        # Build messages and call LLM (language-aware prompt)
        messages = build_generic_cleaner_messages(preprocessed, self.language)
        result = await self.client.chat_completion_json(messages)

        # Parse and validate result
        cleaned_text = result.get("text", preprocessed)
        confidence = float(result.get("confidence", 0.5))
        changes = result.get("changes", [])

        # Post-process: deterministic digit normalization for Hindi
        if self.language == "hi":
            normalized = devanagari_to_western(cleaned_text)
            if normalized != cleaned_text:
                cleaned_text = normalized
                if "itn" not in changes:
                    changes.append("itn")

        return CleanedRecord(
            audio_filepath=record.audio_filepath,
            text=cleaned_text,
            duration=record.duration,
            original_text=record.text,
            dialect=self.language,
            confidence=confidence,
            changes=changes,
        )


__all__ = ["GenericCleanerAgent"]
