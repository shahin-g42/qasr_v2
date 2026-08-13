"""CorrectorAgent: issue-driven repair pass for rejected transcripts.

After the validator rejects a record with a list of specific issues, this
agent attempts one focused repair that resolves ONLY those issues while
preserving the speaker's exact words. It rescues the common "invalid but
no correction offered" case, where the validator flags problems without
volunteering a corrected_text (the rounds=0 rejections).

One shared implementation for all languages, parameterized by language —
the issue list and the original/processed texts carry the language-specific
detail; the prompt itself is language-neutral in structure.
"""

from __future__ import annotations

import logging

from .config import PipelineConfig
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord
from .prompts import build_corrector_messages

LOGGER = logging.getLogger("data_processing.corrector")


class CorrectorAgent:
    """Repairs a rejected transcript given the validator's issue list."""

    def __init__(
        self,
        client: VLLMClient,
        config: PipelineConfig | None = None,
        language: str = "ar",
    ) -> None:
        self.client = client
        self.config = config
        self.language = language

    async def correct(
        self, record: CleanedRecord, issues: list[str]
    ) -> str | None:
        """Attempt an issue-driven repair.

        Returns the corrected transcript, or None when the corrector
        declines (no issues, empty/null output, or output unchanged).
        The caller re-validates any returned text — this never accepts
        on its own.
        """
        if not issues:
            return None

        messages = build_corrector_messages(
            original=record.original_text,
            processed=record.text,
            issues=issues,
            language=self.language,
        )
        result = await self.client.chat_completion_json(messages)

        corrected = result.get("corrected_text")
        if not isinstance(corrected, str) or not corrected.strip():
            return None
        corrected = corrected.strip()
        return corrected if corrected != record.text else None


__all__ = ["CorrectorAgent"]
