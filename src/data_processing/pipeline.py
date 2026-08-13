"""CLI entry point for the Arabic transcript processing pipeline.

Usage:
    PYTHONPATH=src python -m data_processing \
        --config configs/data_processing/arabic_cleaning.yaml \
        --node-rank 0 \
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from .config import PipelineConfig
from .llm_client import VLLMClient
from .orchestrator import Orchestrator
from .reporting import (
    ManifestReport,
    generate_manifest_report,
    generate_summary_report,
    save_report,
)

LOGGER = logging.getLogger("data_processing")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Arabic transcript processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_processing/arabic_cleaning.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Node rank (0-7) for distributed processing",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=8,
        help="Total number of nodes",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        default=None,
        help="Comma-separated manifest names to process (default: all)",
    )
    parser.add_argument(
        "--record-range",
        type=str,
        default=None,
        help="Process only records [START:END) of each manifest "
        "(END empty for EOF, e.g. '120000:'). Outputs are namespaced "
        "per range so multiple nodes can slice one huge manifest.",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip records already present in any prior output of the "
        "manifest (cleaned/shards/rejected) before processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override workers_per_node from the YAML config "
        "(asyncio tasks; vLLM queues anything past its max-num-seqs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count records and validate connectivity without processing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accepted for compatibility; workers ALWAYS resume from their "
        "per-shard checkpoints when present",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


async def check_vllm_health(config: PipelineConfig) -> bool:
    """Check if the vLLM server is healthy."""
    async with VLLMClient(config) as client:
        return await client.health_check()


async def dry_run(config: PipelineConfig) -> None:
    """Perform a dry run: count records and check connectivity."""
    LOGGER.info("=== DRY RUN MODE ===")

    # Check vLLM connectivity
    LOGGER.info("Checking vLLM server at %s...", config.vllm_base_url)
    healthy = await check_vllm_health(config)
    if healthy:
        LOGGER.info("  vLLM server is healthy")
    else:
        LOGGER.warning("  vLLM server is NOT reachable (will retry during processing)")

    # Discover manifests
    orchestrator = Orchestrator(config)
    jobs = orchestrator.discover_manifests()
    node_jobs = orchestrator.distribute_jobs(jobs)

    LOGGER.info("Manifests assigned to this node:")
    total_records = 0
    for job in node_jobs:
        from .manifest_io import count_manifest_records

        try:
            count = count_manifest_records(job.path)
            total_records += count
            LOGGER.info("  %s: %d records", job.path.name, count)
        except FileNotFoundError:
            LOGGER.warning("  %s: NOT FOUND", job.path.name)

    LOGGER.info("Total records to process: %d", total_records)

    # Estimate processing time
    # Rough estimate: 500 records/sec/node with 48 workers
    est_seconds = total_records / 500
    est_hours = est_seconds / 3600
    LOGGER.info("Estimated processing time: %.1f hours", est_hours)


async def run_pipeline(config: PipelineConfig) -> None:
    """Run the full processing pipeline."""
    start_time = time.time()

    LOGGER.info("Starting Arabic transcript processing pipeline")
    LOGGER.info("  Node rank: %d/%d", config.node_rank, config.num_nodes)
    LOGGER.info("  Workers per node: %d", config.workers_per_node)
    LOGGER.info("  Batch size: %d", config.batch_size)
    LOGGER.info("  vLLM endpoint: %s", config.vllm_base_url)

    # Wait for vLLM to be healthy
    LOGGER.info("Waiting for vLLM server...")
    async with VLLMClient(config) as client:
        await client.wait_for_health(max_wait_seconds=300)

    # Run orchestrator
    orchestrator = Orchestrator(config)
    report = await orchestrator.run()

    # Generate reports.
    # Report on the jobs the orchestrator actually ran — they carry the
    # resolved, slice-aware output/rejected paths. Re-discovering manifests
    # here would hand back fresh ManifestJob objects with output_path=None,
    # so the loop produced no reports at all.
    output_dir = Path(config.output_dir)
    reports: list[ManifestReport] = []

    for job in report.jobs:
        if not (job.output_path and job.output_path.is_file()):
            continue
        lang_dir = output_dir / job.language
        manifest_report = generate_manifest_report(
            source_path=job.path,
            cleaned_path=job.output_path,
            rejected_path=job.rejected_path,
            processing_start_time=start_time,
        )
        reports.append(manifest_report)
        save_report(
            manifest_report,
            lang_dir / f"{job.manifest_name}_report.json",
        )

    # Generate summary
    if reports:
        generate_summary_report(reports, output_dir / "summary_report.json")

    elapsed = time.time() - start_time
    LOGGER.info("Pipeline complete in %.1f hours", elapsed / 3600)
    LOGGER.info("  Manifests processed: %d", report.manifests_processed)
    LOGGER.info("  Total records: %d", report.total_records)
    LOGGER.info("  Accepted: %d", report.total_accepted)
    LOGGER.info("  Rejected: %d", report.total_rejected)

    if report.errors:
        LOGGER.warning("  Errors: %d", len(report.errors))
        for error in report.errors[:5]:
            LOGGER.warning("    %s", error)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    # Load config
    config_path = Path(args.config)
    if config_path.is_file():
        config = PipelineConfig.from_yaml(config_path)
    else:
        LOGGER.warning("Config file not found: %s, using defaults", config_path)
        config = PipelineConfig()

    # Override with CLI args
    config.node_rank = args.node_rank
    config.num_nodes = args.num_nodes
    if args.manifests:
        config.manifest_filter = args.manifests
    if args.record_range:
        config.record_range = args.record_range
    if args.skip_processed:
        config.skip_processed = True
    if args.workers is not None:
        config.workers_per_node = args.workers

    config.validate()

    # Run
    if args.dry_run:
        asyncio.run(dry_run(config))
    else:
        asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()


__all__ = ["main"]
