"""Worker process: reads a shard, cleans records, validates, and writes output.

Each worker is a separate OS process with its own asyncio event loop.
Workers write to their own shard output file (Lustre-safe).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cleaner import CleanerAgent
from .config import PipelineConfig
from .corrector import CorrectorAgent
from .generic_cleaner import GenericCleanerAgent
from .generic_validator import GenericValidatorAgent
from .llm_client import VLLMClient
from .manifest_io import CleanedRecord, ManifestRecord, append_to_shard, read_manifest
from .validator import ValidatorAgent


LOGGER = logging.getLogger("data_processing.worker")


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
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_line_number": self.last_line_number,
            "records_processed": self.records_processed,
            "timestamp": self.timestamp,
        }
        path.write_text(json.dumps(data), encoding="utf-8")


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

    def request_shutdown(self) -> None:
        """Signal the worker to shut down gracefully."""
        self._shutdown_requested = True

    async def run(self) -> WorkerStats:
        """Run the worker: read shard, process, write output."""
        stats = WorkerStats(worker_id=self.worker_id, shard_path=str(self.shard_path))

        # Load checkpoint if resuming
        checkpoint = Checkpoint.load(self.checkpoint_path)
        skip_until = checkpoint.last_line_number if checkpoint else 0
        stats.processed_records = checkpoint.records_processed if checkpoint else 0

        LOGGER.info(
            "Worker %d starting on %s (resuming from line %d)",
            self.worker_id,
            self.shard_path.name,
            skip_until,
        )

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.request_shutdown)

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
                    batch = []

                    # Checkpoint periodically
                    if stats.processed_records % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(record.line_number, stats)

            # Process remaining records
            if batch and not self._shutdown_requested:
                await self._process_batch(batch, cleaner, validator, stats)

        stats.end_time = time.time()
        self._save_checkpoint(
            records[-1].line_number if records else skip_until, stats
        )

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
        # Clean the batch
        cleaned_records = await cleaner.process_batch(batch)

        accepted: list[CleanedRecord] = []
        rejected: list[CleanedRecord] = []

        for cleaned in cleaned_records:
            # Decide whether to validate this record
            should_validate = (
                random.random() < self.config.validation_sample_rate
            )

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
    """Entry point for running a worker in a subprocess."""
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


__all__ = ["Checkpoint", "Worker", "WorkerStats", "build_agents", "run_worker"]
