"""Reprocess rejected records through the (improved) cleaning pipeline.

Rejected records keep ``original_text`` (the raw source transcript), so they
can be re-run from scratch. This is useful after pipeline improvements —
e.g. records rejected due to LLM-response truncation or overly strict
heuristic vetoes in earlier versions.

Usage:
    PYTHONPATH=src python -m data_processing.reprocess \
        --config configs/data_processing/arabic_cleaning.yaml \
        --rejected-dir /path/to/cleaned_manifests \
        --language ar

Outputs (next to each input file):
    <name>_recovered.jsonl       — records that now pass validation
    <name>_still_rejected.jsonl  — records that fail even after rework

Resumable: already-processed records (present in either output file) are
skipped on re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from .config import PipelineConfig
from .llm_client import VLLMClient
from .manifest_io import ManifestRecord
from .pipeline import setup_logging
from .worker import build_agents

LOGGER = logging.getLogger("data_processing.reprocess")


def _record_key(entry: dict) -> tuple[str, str]:
    """Stable identity for a rejected record (dedupe + resume)."""
    return (
        str(entry.get("audio_filepath", "")),
        str(entry.get("original_text") or entry.get("text") or ""),
    )


def load_rejected_records(path: Path) -> list[ManifestRecord]:
    """Load rejected records, rebuilding source records from original_text.

    Deduplicates by (audio_filepath, original_text) — rejected files are
    append-only and may contain duplicates from resumed runs.
    """
    records: list[ManifestRecord] = []
    seen: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                LOGGER.warning("%s:%d: skipping malformed line", path.name, line_number)
                continue

            key = _record_key(entry)
            if key in seen:
                continue
            seen.add(key)

            # Reprocess from the ORIGINAL text, not the failed cleaned text
            source_text = entry.get("original_text") or entry.get("text") or ""
            records.append(
                ManifestRecord(
                    line_number=line_number,
                    audio_filepath=str(entry.get("audio_filepath", "")),
                    text=source_text,
                    duration=entry.get("duration"),
                    raw=entry,
                )
            )

    return records


def load_done_keys(*paths: Path) -> set[tuple[str, str]]:
    """Collect keys already present in output files (for resume)."""
    done: set[tuple[str, str]] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    done.add(_record_key(json.loads(stripped)))
                except json.JSONDecodeError:
                    continue
    return done


class RejectedReprocessor:
    """Re-runs rejected records through cleaner + validator concurrently."""

    def __init__(
        self,
        config: PipelineConfig,
        language: str = "ar",
        concurrency: int = 512,
    ) -> None:
        self.config = config
        self.language = language
        self.semaphore = asyncio.Semaphore(concurrency)
        self._write_lock = asyncio.Lock()

    async def process_file(self, rejected_path: Path, client: VLLMClient) -> dict:
        """Reprocess one *_rejected.jsonl file. Returns stats dict."""
        stem = rejected_path.stem
        if stem.endswith("_rejected"):
            stem = stem[: -len("_rejected")]
        recovered_path = rejected_path.parent / f"{stem}_recovered.jsonl"
        still_rejected_path = rejected_path.parent / f"{stem}_still_rejected.jsonl"

        records = load_rejected_records(rejected_path)
        done = load_done_keys(recovered_path, still_rejected_path)
        pending = [
            r for r in records
            if (r.audio_filepath, r.text) not in done
        ]

        LOGGER.info(
            "%s: %d unique records (%d already done, %d to process)",
            rejected_path.name,
            len(records),
            len(records) - len(pending),
            len(pending),
        )

        stats = {
            "file": rejected_path.name,
            "unique": len(records),
            "skipped": len(records) - len(pending),
            "recovered": 0,
            "still_rejected": 0,
            "errors": 0,
        }
        if not pending:
            return stats

        cleaner, validator = build_agents(client, self.config, self.language)

        async def _process_one(record: ManifestRecord) -> None:
            async with self.semaphore:
                try:
                    cleaned = await cleaner.process_single(record)
                    final_record, accepted = await validator.validate_with_retry(cleaned)
                except Exception as exc:
                    LOGGER.warning(
                        "Reprocess error for %s: %s", record.audio_filepath, exc
                    )
                    stats["errors"] += 1
                    return

                line = json.dumps(final_record.to_dict(), ensure_ascii=False) + "\n"
                async with self._write_lock:
                    if accepted:
                        with recovered_path.open("a", encoding="utf-8") as handle:
                            handle.write(line)
                        stats["recovered"] += 1
                    else:
                        with still_rejected_path.open("a", encoding="utf-8") as handle:
                            handle.write(line)
                        stats["still_rejected"] += 1

                total_done = stats["recovered"] + stats["still_rejected"]
                if total_done % 500 == 0:
                    LOGGER.info(
                        "%s: %d/%d done (%d recovered, %d still rejected)",
                        rejected_path.name,
                        total_done,
                        len(pending),
                        stats["recovered"],
                        stats["still_rejected"],
                    )

        await asyncio.gather(*(_process_one(r) for r in pending))
        return stats


async def run_reprocess(args: argparse.Namespace) -> None:
    config = PipelineConfig.from_yaml(args.config)
    config.validate()

    rejected_dir = Path(args.rejected_dir)
    if args.files:
        files = [rejected_dir / name.strip() for name in args.files.split(",")]
    else:
        files = sorted(rejected_dir.glob("*_rejected.jsonl"))
    files = [f for f in files if f.is_file()]

    if not files:
        LOGGER.error("No *_rejected.jsonl files found in %s", rejected_dir)
        sys.exit(1)

    LOGGER.info("Reprocessing %d rejected files (language=%s)", len(files), args.language)

    reprocessor = RejectedReprocessor(
        config, language=args.language, concurrency=args.concurrency
    )

    start = time.time()
    all_stats: list[dict] = []
    async with VLLMClient(config) as client:
        await client.wait_for_health(max_wait_seconds=300)
        for path in files:
            all_stats.append(await reprocessor.process_file(path, client))

    # Summary
    elapsed = time.time() - start
    total_recovered = sum(s["recovered"] for s in all_stats)
    total_still = sum(s["still_rejected"] for s in all_stats)
    total_errors = sum(s["errors"] for s in all_stats)
    LOGGER.info("=" * 60)
    LOGGER.info("Reprocessing complete in %.1f min", elapsed / 60)
    for s in all_stats:
        processed = s["recovered"] + s["still_rejected"]
        rate = (s["recovered"] / processed * 100) if processed else 0.0
        LOGGER.info(
            "  %s: %d recovered / %d processed (%.1f%%)",
            s["file"], s["recovered"], processed, rate,
        )
    LOGGER.info(
        "TOTAL: %d recovered, %d still rejected, %d errors",
        total_recovered, total_still, total_errors,
    )
    LOGGER.info(
        "Merge recovered records into training data with:\n"
        "  cat <name>_cleaned.jsonl <name>_recovered.jsonl > <name>_final.jsonl"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess rejected records through the improved pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_processing/arabic_cleaning.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--rejected-dir",
        type=str,
        required=True,
        help="Directory containing *_rejected.jsonl files",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated rejected file names (default: all in dir)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ar",
        help="Language code for agent routing (default: ar)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=512,
        help="Max concurrent in-flight records (default: 512)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    asyncio.run(run_reprocess(args))


if __name__ == "__main__":
    main()


__all__ = ["main", "RejectedReprocessor", "load_rejected_records"]
