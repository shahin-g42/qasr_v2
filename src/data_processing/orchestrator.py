"""Orchestrator: coordinates multiple workers to process manifests.

Handles sharding, worker lifecycle, progress monitoring, and merging.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .config import PipelineConfig
from .manifest_io import (
    collect_processed_keys,
    count_manifest_records,
    merge_shards,
    shard_manifest,
)
from .worker import WorkerStats, run_worker

LOGGER = logging.getLogger("data_processing.orchestrator")


@dataclass
class ManifestJob:
    """A manifest to be processed."""

    path: Path
    language: str
    record_count: int = 0
    shard_paths: list[Path] = field(default_factory=list)
    output_path: Path | None = None
    # Slice-aware output identity, filled in by process_manifest(). Reporting
    # must read these off the job instead of re-deriving names from
    # path.stem — that loses the record-range suffix and names files that
    # do not exist.
    manifest_name: str = ""
    rejected_path: Path | None = None
    # Fault accounting, filled in by process_manifest(): shard indices that
    # exhausted their retries, and the sharded-vs-accounted reconciliation
    # delta (positive = records missing from the outputs). Nonzero either
    # way means the cleaned/rejected files do not cover the shards.
    failed_shards: list[int] = field(default_factory=list)
    missing_records: int = 0


@dataclass
class OrchestratorReport:
    """Summary report from the orchestrator."""

    manifests_processed: int = 0
    total_records: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    worker_stats: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Shards that exhausted their retries ("<manifest>: shards [i, ...]")
    # and the total reconciliation gap across manifests. Both used to be
    # logged and discarded, so a failed shard exited 0 with its whole
    # record slice silently missing from the outputs.
    failed_shards: list[str] = field(default_factory=list)
    missing_records: int = 0
    # The jobs this node actually ran, carrying their resolved output paths
    # so downstream reporting does not have to re-discover them.
    jobs: list[ManifestJob] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """True when any manifest, shard, or record went missing."""
        return bool(self.errors or self.failed_shards or self.missing_records)


class Orchestrator:
    """Coordinates multi-worker processing of manifests."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.progress_counter = multiprocessing.Value("i", 0)

    def discover_manifests(self) -> list[ManifestJob]:
        """Discover manifests from the training config YAML.

        language_filter supports a single code ("ar"), a comma-separated
        list ("en,zh,hi,ml"), or empty/None for all languages.
        """
        config_path = Path(self.config.manifests_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Manifests config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            train_config = yaml.safe_load(handle) or {}

        jobs: list[ManifestJob] = []
        language_filter = self.config.language_filter
        allowed_languages = (
            {lang.strip() for lang in language_filter.split(",") if lang.strip()}
            if language_filter
            else None
        )

        # Process train manifests
        train_manifest = train_config.get("train_manifest", {})
        if isinstance(train_manifest, dict):
            for language, paths in train_manifest.items():
                if allowed_languages and language not in allowed_languages:
                    continue
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    jobs.append(ManifestJob(path=Path(path), language=language))

        # Process eval manifests
        eval_manifest = train_config.get("eval_manifest", {})
        if isinstance(eval_manifest, dict):
            for language, paths in eval_manifest.items():
                if allowed_languages and language not in allowed_languages:
                    continue
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    jobs.append(ManifestJob(path=Path(path), language=language))

        LOGGER.info(
            "Discovered %d %s manifests to process",
            len(jobs),
            language_filter or "all",
        )

        # Optional narrowing to specific manifests (--manifests stems)
        if self.config.manifest_filter:
            wanted = {
                stem.strip()
                for stem in self.config.manifest_filter.split(",")
                if stem.strip()
            }
            jobs = [job for job in jobs if job.path.stem in wanted]
            LOGGER.info(
                "Manifest filter %s matched %d manifests", sorted(wanted), len(jobs)
            )
        return jobs

    def distribute_jobs(self, jobs: list[ManifestJob]) -> list[ManifestJob]:
        """Filter jobs for this node based on node_rank."""
        if self.config.num_nodes <= 1:
            return jobs

        # Round-robin distribution by index
        node_jobs = [
            job for i, job in enumerate(jobs)
            if i % self.config.num_nodes == self.config.node_rank
        ]
        LOGGER.info(
            "Node %d/%d assigned %d of %d manifests",
            self.config.node_rank,
            self.config.num_nodes,
            len(node_jobs),
            len(jobs),
        )
        return node_jobs

    async def process_manifest(self, job: ManifestJob) -> list[WorkerStats]:
        """Process a single manifest (or a record-range slice of it)."""
        manifest_path = job.path
        start, end = self.config.parse_record_range()
        # Slice runs get their own namespace so shards/checkpoints/outputs
        # from different nodes working the same manifest never collide.
        slice_suffix = (
            f"_r{start}_{end if end is not None else 'end'}"
            if self.config.record_range
            else ""
        )
        manifest_name = manifest_path.stem + slice_suffix

        LOGGER.info("Processing manifest: %s (language=%s)", manifest_name, job.language)

        # Set up directories (namespaced by language to avoid stem collisions)
        shard_base = Path(self.config.shard_dir) / job.language / manifest_name
        output_base = Path(self.config.output_dir) / job.language
        checkpoint_base = Path(self.config.checkpoint_dir) / job.language / manifest_name

        shard_base.mkdir(parents=True, exist_ok=True)
        output_base.mkdir(parents=True, exist_ok=True)
        checkpoint_base.mkdir(parents=True, exist_ok=True)

        # Resolve output identity BEFORE any early return, so reporting still
        # covers a resumed run that finds nothing left to process.
        job.manifest_name = manifest_name
        job.output_path = output_base / f"{manifest_name}_cleaned.jsonl"
        job.rejected_path = output_base / f"{manifest_name}_rejected.jsonl"

        # Count records
        job.record_count = count_manifest_records(manifest_path)
        LOGGER.info("  %d records to process", job.record_count)

        if job.record_count == 0:
            LOGGER.warning("  Skipping empty manifest: %s", manifest_name)
            return []

        # Skip records already processed by earlier runs / other slices
        skip_keys = None
        if self.config.skip_processed:
            skip_keys = collect_processed_keys(output_base, manifest_path.stem)

        # Shard the manifest (slice-aware)
        slice_size = (end if end is not None else job.record_count) - start
        num_shards = max(1, min(self.config.workers_per_node, max(slice_size, 1)))
        job.shard_paths = shard_manifest(
            manifest_path,
            num_shards,
            shard_base,
            start=start,
            end=end,
            skip_keys=skip_keys,
        )

        # Progress total = records actually sharded (slice minus already done)
        job.record_count = sum(
            sum(1 for _ in path.open("r", encoding="utf-8"))
            for path in job.shard_paths
        )
        if job.record_count == 0:
            LOGGER.info("  Nothing left to process for %s", manifest_name)
            return []

        # Set up output paths
        rejected_path = job.rejected_path

        # Launch workers
        worker_stats, failed_shards = await self._launch_workers(
            job=job,
            manifest_name=manifest_name,
            shard_base=shard_base,
            output_base=output_base,
            checkpoint_base=checkpoint_base,
            rejected_path=rejected_path,
        )
        job.failed_shards = failed_shards

        # Merge output shards. Glob the output directory rather than deriving
        # names from the CURRENT run's shard count: resuming with a smaller
        # --workers value used to orphan the higher-index shards an earlier
        # run produced, and merge_shards' full overwrite of
        # <name>_cleaned.jsonl dropped those records for good
        # (--skip-processed then hid them from re-cleaning). Rejected records
        # have no shard equivalent — every worker appends to the single
        # shared <name>_rejected.jsonl, which is never merged or overwritten.
        output_shards = sorted(output_base.glob(f"{manifest_name}_shard_*.jsonl"))
        if output_shards:
            merge_shards(output_shards, job.output_path)
            LOGGER.info("  Merged %d shards into %s", len(output_shards), job.output_path)

        # Reconcile: every record sharded must come back accepted or
        # rejected. A permanently failed shard (or a dropped batch)
        # otherwise vanishes without a trace in the outputs.
        accounted = sum(
            stats.accepted_records + stats.rejected_records for stats in worker_stats
        )
        if accounted != job.record_count:
            job.missing_records = job.record_count - accounted
            LOGGER.error(
                "Manifest %s: sharded %d records but workers accounted for %d "
                "(accepted+rejected) — delta %+d (positive = records missing)",
                manifest_name,
                job.record_count,
                accounted,
                job.missing_records,
            )

        return worker_stats

    async def _launch_workers(
        self,
        job: ManifestJob,
        manifest_name: str,
        shard_base: Path,
        output_base: Path,
        checkpoint_base: Path,
        rejected_path: Path,
        max_worker_retries: int = 2,
    ) -> tuple[list[WorkerStats], list[int]]:
        """Launch and monitor worker tasks with fault tolerance.

        If a worker dies (raises an exception), its shard is re-assigned
        up to max_worker_retries times before being marked as failed.
        ``manifest_name`` carries any slice suffix so outputs from
        different nodes working the same manifest never collide.

        Returns (worker stats, permanently failed shard indices). The
        failed indices must reach the caller — they used to be logged and
        discarded here, so a shard that exhausted its retries lost its
        whole record slice with a clean exit.
        """
        num_shards = len(job.shard_paths)

        # Reset progress counter
        with self.progress_counter.get_lock():
            self.progress_counter.value = 0

        # Track retry counts per shard index
        shard_retries: dict[int, int] = dict.fromkeys(range(num_shards), 0)
        failed_shards: list[int] = []

        def _create_worker_task(shard_idx: int) -> asyncio.Task:
            """Create a worker task for the given shard index."""
            shard_path = job.shard_paths[shard_idx]
            output_path = output_base / f"{manifest_name}_shard_{shard_idx:04d}.jsonl"
            checkpoint_path = checkpoint_base / f"shard_{shard_idx:04d}.checkpoint"
            return asyncio.create_task(
                run_worker(
                    worker_id=shard_idx,
                    shard_path=shard_path,
                    output_path=output_path,
                    rejected_path=rejected_path,
                    checkpoint_path=checkpoint_path,
                    config=self.config,
                    language=job.language,
                    progress_counter=self.progress_counter,
                )
            )

        # Create initial worker tasks, mapping task -> shard_idx
        task_to_shard: dict[asyncio.Task, int] = {}
        for i in range(num_shards):
            task = _create_worker_task(i)
            task_to_shard[task] = i

        # Monitor progress with tqdm
        all_stats: list[WorkerStats] = []
        with tqdm(
            total=job.record_count,
            desc=f"Processing {manifest_name}",
            unit="rec",
        ) as pbar:
            last_count = 0
            pending = set(task_to_shard.keys())

            while pending:
                # Wait for any task to complete or timeout for progress update
                done, pending = await asyncio.wait(
                    pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )

                # Collect results from completed tasks
                for task in done:
                    shard_idx = task_to_shard.pop(task)
                    try:
                        stats = task.result()
                        all_stats.append(stats)
                    except Exception as exc:
                        # Fault tolerance: re-assign the shard if retries remain
                        shard_retries[shard_idx] += 1
                        retries_used = shard_retries[shard_idx]
                        if retries_used <= max_worker_retries:
                            LOGGER.warning(
                                "Worker for shard %d failed (attempt %d/%d): %s. "
                                "Re-assigning shard...",
                                shard_idx,
                                retries_used,
                                max_worker_retries + 1,
                                exc,
                            )
                            new_task = _create_worker_task(shard_idx)
                            task_to_shard[new_task] = shard_idx
                            pending.add(new_task)
                        else:
                            LOGGER.error(
                                "Worker for shard %d failed permanently after "
                                "%d attempts: %s",
                                shard_idx,
                                retries_used,
                                exc,
                            )
                            failed_shards.append(shard_idx)

                # Update progress bar
                with self.progress_counter.get_lock():
                    current = self.progress_counter.value
                pbar.update(current - last_count)
                last_count = current

        if failed_shards:
            LOGGER.error(
                "Manifest %s: %d shards failed permanently: %s",
                manifest_name,
                len(failed_shards),
                failed_shards,
            )

        return all_stats, failed_shards

    async def run(self) -> OrchestratorReport:
        """Run the full orchestration pipeline."""
        report = OrchestratorReport()

        # Discover and distribute manifests
        all_jobs = self.discover_manifests()
        node_jobs = self.distribute_jobs(all_jobs)
        report.jobs = node_jobs

        if not node_jobs:
            LOGGER.warning("No manifests assigned to this node")
            return report

        # Process each manifest
        for job in node_jobs:
            try:
                worker_stats = await self.process_manifest(job)
                report.manifests_processed += 1
                report.total_records += job.record_count

                for stats in worker_stats:
                    report.total_accepted += stats.accepted_records
                    report.total_rejected += stats.rejected_records
                    report.worker_stats.append(stats.to_dict())

                if job.failed_shards:
                    report.failed_shards.append(
                        f"{job.manifest_name}: shards {job.failed_shards}"
                    )
                # abs(): a duplicate-records delta on one manifest must not
                # cancel a missing-records delta on another in has_failures.
                report.missing_records += abs(job.missing_records)

            except Exception as exc:
                error_msg = f"Failed to process {job.path}: {exc}"
                LOGGER.error(error_msg)
                report.errors.append(error_msg)

        LOGGER.info(
            "Orchestration complete: %d manifests, %d records, %d accepted, %d rejected",
            report.manifests_processed,
            report.total_records,
            report.total_accepted,
            report.total_rejected,
        )
        return report


__all__ = ["ManifestJob", "Orchestrator", "OrchestratorReport"]
