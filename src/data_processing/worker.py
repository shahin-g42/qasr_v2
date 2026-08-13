"""Worker: reads a shard, cleans records, validates, and writes output.

Each worker is an asyncio task in the orchestrator's event loop.
Workers write to their own shard output file (Lustre-safe).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import signal
import time
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cleaner import CleanerAgent
from .config import PipelineConfig
from .corrector import CorrectorAgent
from .duration_probe import fill_missing_durations
from .generic_cleaner import GenericCleanerAgent
from .generic_validator import GenericValidatorAgent
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord, ManifestRecord, append_to_shard, read_manifest
from .validator import ValidatorAgent

LOGGER = logging.getLogger("data_processing.worker")

# Cleaner outcomes that validation sampling must never wave through.
# The cleaner short-circuited these records without an LLM pass, so `text`
# is only the preprocessed original — at a 5% sample rate 95% of them would
# land in the accepted set having never been checked at all. The cleaner
# defers to "the validator's hard checks downstream", so make that true.
_ALWAYS_VALIDATE_CHANGES = frozenset({"skipped_short", "skipped_wrong_script"})

# The cleaning LLM call failed outright and `text` is the untouched original.
# Rejected deterministically rather than validated: the validator would burn
# another (likely also failing) LLM call on text that was never cleaned,
# whereas reprocess re-cleans it properly from original_text.
_CLEANER_FAILURE_CHANGE = "error_fallback"

# All live Worker instances. Workers share one event loop, so ONE signal
# handler (registered once) fans shutdown out to every live worker via
# weakrefs — each worker registering its own handler would overwrite the
# previous ones and only the last worker would ever see the signal.
_LIVE_WORKERS: weakref.WeakSet[Worker] = weakref.WeakSet()
_SIGNAL_HANDLERS_INSTALLED = False


def _install_signal_handlers() -> None:
    """Register process-wide SIGTERM/SIGINT handlers exactly once."""
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return

    def _request_all_shutdown() -> None:
        LOGGER.info("Shutdown signal received; flushing %d workers", len(_LIVE_WORKERS))
        for worker in list(_LIVE_WORKERS):
            worker.request_shutdown()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # Platform without loop signal support (e.g. Windows) raises
        # NotImplementedError/RuntimeError; shutdown then falls back to
        # Ctrl-C KeyboardInterrupt handling by the orchestrator.
        with suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, _request_all_shutdown)
    _SIGNAL_HANDLERS_INSTALLED = True


def build_agents(
    client: VLLMClient, config: PipelineConfig, language: str
) -> tuple[CleanerAgent, ValidatorAgent]:
    """Build the cleaner/validator pair for a manifest's language.

    Arabic gets the specialized agents (dialect preservation, diacritics,
    Arabic ITN). All other languages share the generic language-aware agents.
    When correction is enabled, the validator gets an issue-driven corrector
    that repairs rejected records before they are dropped.
    """
    corrector = (
        CorrectorAgent(client, config=config, language=language)
        if config.correction_enabled
        else None
    )
    if language == "ar":
        return (
            CleanerAgent(client, config=config),
            ValidatorAgent(client, config=config, corrector=corrector),
        )
    return (
        GenericCleanerAgent(client, config=config, language=language),
        GenericValidatorAgent(
            client, config=config, language=language, corrector=corrector
        ),
    )


@dataclass
class WorkerStats:
    """Statistics for a worker's processing session."""

    worker_id: int
    shard_path: str
    total_records: int = 0
    processed_records: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def records_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.processed_records / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "shard_path": self.shard_path,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "records_per_second": round(self.records_per_second, 2),
        }


@dataclass
class Checkpoint:
    """Checkpoint state for resumable processing."""

    last_line_number: int
    records_processed: int
    timestamp: float
    # Fingerprint of the shard file this checkpoint describes. Resume is
    # keyed by LINE NUMBER, but shards are regenerated on every run: with
    # --skip-processed the records that survive get redistributed, so line N
    # of the new shard is a DIFFERENT record. Resuming on a stale line
    # number would silently skip records that were never processed, so a
    # checkpoint whose fingerprint no longer matches is discarded.
    shard_signature: str = ""

    @classmethod
    def load(cls, path: Path) -> Checkpoint | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_line_number=data["last_line_number"],
                records_processed=data["records_processed"],
                timestamp=data["timestamp"],
                shard_signature=data.get("shard_signature", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_line_number": self.last_line_number,
            "records_processed": self.records_processed,
            "timestamp": self.timestamp,
            "shard_signature": self.shard_signature,
        }
        # Atomic write: a torn checkpoint makes load() return None, which
        # restarts the whole shard and re-appends every record (mass
        # duplication). tmp + os.replace guarantees an all-or-nothing file.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)


def compute_shard_signature(path: Path) -> str:
    """Content fingerprint of a shard file, for checkpoint validation."""
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Worker:
    """Async worker that processes a single shard."""

    def __init__(
        self,
        worker_id: int,
        shard_path: Path,
        output_path: Path,
        rejected_path: Path,
        checkpoint_path: Path,
        config: PipelineConfig,
        language: str = "ar",
        progress_counter: Any = None,
    ) -> None:
        self.worker_id = worker_id
        self.shard_path = shard_path
        self.output_path = output_path
        self.rejected_path = rejected_path
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.language = language
        self.progress_counter = progress_counter
        self._shutdown_requested = False
        # Line number of the LAST record whose batch was fully processed
        # (written to output). Checkpoints must record this — never the
        # last record READ — or a mid-run shutdown would make resume
        # silently skip unprocessed records.
        self._last_processed_line = 0
        self._shard_signature = ""

    def request_shutdown(self) -> None:
        """Signal the worker to shut down gracefully."""
        self._shutdown_requested = True

    async def run(self) -> WorkerStats:
        """Run the worker: read shard, process, write output."""
        stats = WorkerStats(worker_id=self.worker_id, shard_path=str(self.shard_path))

        # Load checkpoint if resuming — but only if it describes THIS shard
        # content. Shards are rewritten every run, so a checkpoint taken
        # against a different partition would skip unprocessed records.
        self._shard_signature = compute_shard_signature(self.shard_path)
        checkpoint = Checkpoint.load(self.checkpoint_path)
        if checkpoint is not None and checkpoint.shard_signature != self._shard_signature:
            LOGGER.warning(
                "Worker %d: discarding checkpoint for %s — it was taken against "
                "different shard content (fingerprint mismatch), so its line "
                "number no longer identifies a record. Processing from the "
                "start; use --skip-processed so records the earlier run "
                "finished are excluded from the regenerated shards.",
                self.worker_id,
                self.shard_path.name,
            )
            checkpoint = None
        skip_until = checkpoint.last_line_number if checkpoint else 0
        self._last_processed_line = skip_until
        stats.processed_records = checkpoint.records_processed if checkpoint else 0

        LOGGER.info(
            "Worker %d starting on %s (resuming from line %d)",
            self.worker_id,
            self.shard_path.name,
            skip_until,
        )

        # Set up signal handlers for graceful shutdown (registered once
        # for the whole process; fans out to every live worker)
        _LIVE_WORKERS.add(self)
        _install_signal_handlers()

        async with VLLMClient(self.config) as client:
            cleaner, validator = build_agents(client, self.config, self.language)

            # Read records from shard
            records = list(read_manifest(self.shard_path))
            stats.total_records = len(records)

            # Filter out already-processed records
            if skip_until > 0:
                records = [r for r in records if r.line_number > skip_until]
                LOGGER.info(
                    "Worker %d: skipping %d already-processed records",
                    self.worker_id,
                    stats.total_records - len(records),
                )

            # Process in batches
            batch: list[ManifestRecord] = []
            for record in records:
                if self._shutdown_requested:
                    LOGGER.info("Worker %d: shutdown requested, flushing", self.worker_id)
                    break

                batch.append(record)
                if len(batch) >= self.config.batch_size:
                    await self._process_batch(
                        batch, cleaner, validator, stats
                    )
                    self._last_processed_line = record.line_number
                    batch = []

                    # Checkpoint periodically
                    if stats.processed_records % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(self._last_processed_line, stats)

            # Process remaining records
            if batch and not self._shutdown_requested:
                await self._process_batch(batch, cleaner, validator, stats)
                self._last_processed_line = batch[-1].line_number

        stats.end_time = time.time()
        self._save_checkpoint(self._last_processed_line, stats)

        LOGGER.info(
            "Worker %d finished: %d processed, %d accepted, %d rejected (%.1f rec/s)",
            self.worker_id,
            stats.processed_records,
            stats.accepted_records,
            stats.rejected_records,
            stats.records_per_second,
        )
        return stats

    async def _process_batch(
        self,
        batch: list[ManifestRecord],
        cleaner: CleanerAgent,
        validator: ValidatorAgent,
        stats: WorkerStats,
    ) -> None:
        """Process a batch of records through cleaner and validator."""
        # Fill in durations the source manifest never carried. Must happen
        # before cleaning, which copies duration into every CleanedRecord.
        if self.config.probe_missing_durations:
            await fill_missing_durations(batch, self.config.duration_probe_workers)

        # Clean the batch
        cleaned_records = await cleaner.process_batch(batch)

        accepted: list[CleanedRecord] = []
        rejected: list[CleanedRecord] = []

        for cleaned in cleaned_records:
            # Never cleaned at all — reject straight to the rejected pile so
            # reprocess re-cleans it instead of it passing as cleaned output.
            if _CLEANER_FAILURE_CHANGE in cleaned.changes:
                cleaned.rejection_reasons = ["cleaner_error_fallback"]
                rejected.append(cleaned)
                continue

            # Decide whether to validate this record
            should_validate = bool(
                _ALWAYS_VALIDATE_CHANGES.intersection(cleaned.changes)
            ) or random.random() < self.config.validation_sample_rate

            if should_validate:
                final_record, was_accepted = await validator.validate_with_retry(cleaned)
                if was_accepted:
                    accepted.append(final_record)
                else:
                    rejected.append(final_record)
            else:
                accepted.append(cleaned)

        # Write accepted records
        if accepted:
            append_to_shard(self.output_path, accepted)
            stats.accepted_records += len(accepted)

        # Write rejected records
        if rejected:
            append_to_shard(self.rejected_path, rejected)
            stats.rejected_records += len(rejected)

        stats.processed_records += len(cleaned_records)

        # Update progress counter if provided
        if self.progress_counter is not None:
            with self.progress_counter.get_lock():
                self.progress_counter.value += len(cleaned_records)

    def _save_checkpoint(self, last_line: int, stats: WorkerStats) -> None:
        """Save checkpoint for resume capability."""
        checkpoint = Checkpoint(
            last_line_number=last_line,
            records_processed=stats.processed_records,
            timestamp=time.time(),
            shard_signature=self._shard_signature,
        )
        checkpoint.save(self.checkpoint_path)


async def run_worker(
    worker_id: int,
    shard_path: Path,
    output_path: Path,
    rejected_path: Path,
    checkpoint_path: Path,
    config: PipelineConfig,
    language: str = "ar",
    progress_counter: Any = None,
) -> WorkerStats:
    """Entry point for running a worker as an asyncio task."""
    worker = Worker(
        worker_id=worker_id,
        shard_path=shard_path,
        output_path=output_path,
        rejected_path=rejected_path,
        checkpoint_path=checkpoint_path,
        config=config,
        language=language,
        progress_counter=progress_counter,
    )
    return await worker.run()


__all__ = [
    "Checkpoint",
    "Worker",
    "WorkerStats",
    "build_agents",
    "compute_shard_signature",
    "run_worker",
]
