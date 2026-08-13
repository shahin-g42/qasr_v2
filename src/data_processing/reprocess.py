"""Reprocess rejected records through the (improved) cleaning pipeline.

Rejected records keep ``original_text`` (the raw source transcript), so they
can be re-run from scratch. This is useful after pipeline improvements —
e.g. records rejected due to LLM-response truncation or overly strict
heuristic vetoes in earlier versions.

Distributed mode (8 nodes, full language coverage):
    PYTHONPATH=src python -m data_processing.reprocess \
        --rejected-dir /path/to/cleaned_manifests \
        --node-rank $SLURM_NODEID --num-nodes 8

  * Language is auto-detected from ``<rejected-dir>/<lang>/`` subdirs
    (flat ``*_rejected.jsonl`` layouts fall back to ``--language``).
  * Arabic uses ``--arabic-config``; every other language uses
    ``--multilingual-config`` (agents are routed per language).
  * Each file is split into ``--slice-size`` record slices; slices are
    dealt round-robin across nodes. Every slice writes its own part file
    (``<name>_recovered_pNNNN.jsonl`` / ``<name>_still_rejected_pNNNN.jsonl``),
    so nodes never append to the same file. The assembly step folds all
    parts back into the parent corpus.
  * Files assigned to a node are processed concurrently against the local
    vLLM server (one shared semaphore keeps exactly ``--concurrency``
    requests in flight).

Single-node usage is unchanged:
    PYTHONPATH=src python -m data_processing.reprocess \
        --config configs/data_processing/arabic_cleaning.yaml \
        --rejected-dir /path/to/cleaned_manifests --language ar

Outputs (next to each input file):
    <name>_recovered[_pNNNN].jsonl       — records that now pass validation
    <name>_still_rejected[_pNNNN].jsonl  — records that fail even after rework

Resumable: records already present in any recovered/still-rejected output
(main or part files) are skipped on re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .llm_client import VLLMClient
from .manifest_io import ManifestRecord
from .pipeline import setup_logging
from .worker import build_agents

LOGGER = logging.getLogger("data_processing.reprocess")

DEFAULT_ARABIC_CONFIG = "configs/data_processing/arabic_cleaning.yaml"
DEFAULT_MULTILINGUAL_CONFIG = "configs/data_processing/multilingual_cleaning.yaml"


@dataclass
class ReprocessSlice:
    """One unit of distributed work: a record range of one rejected file."""

    language: str
    path: Path
    slice_idx: int | None = None   # None = whole file (single-node mode)
    start: int | None = None
    end: int | None = None


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
        if not path or not path.is_file():
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


def discover_rejected_files(
    rejected_dir: Path,
    languages: list[str] | None = None,
    files: str | None = None,
    default_language: str = "ar",
) -> list[tuple[str, Path]]:
    """Find (language, path) pairs to reprocess.

    Language comes from the ``<rejected-dir>/<lang>/`` layout; a flat
    directory falls back to ``default_language``. Round-1
    ``*_rejected.jsonl`` piles only — ``*_still_rejected*`` outputs must
    be named explicitly via ``--files``.
    """
    if files:
        pairs: list[tuple[str, Path]] = []
        for name in files.split(","):
            path = (rejected_dir / name.strip()).resolve()
            if not path.is_file():
                LOGGER.warning("Skipping missing --files entry: %s", name)
                continue
            lang = default_language if path.parent == rejected_dir.resolve() else path.parent.name
            pairs.append((lang, path))
        return sorted(pairs, key=lambda p: str(p[1]))

    lang_dirs = sorted(p for p in rejected_dir.iterdir() if p.is_dir())
    if lang_dirs:
        picked = [d for d in lang_dirs if not languages or d.name in languages]
        return sorted(
            (d.name, f)
            for d in picked
            for f in d.glob("*_rejected.jsonl")
            if f.is_file() and not f.stem.endswith("_still_rejected")
        )
    return sorted(
        (default_language, f)
        for f in rejected_dir.glob("*_rejected.jsonl")
        if f.is_file() and not f.stem.endswith("_still_rejected")
    )


class RejectedReprocessor:
    """Re-runs rejected records through cleaner + validator concurrently."""

    def __init__(
        self,
        config: PipelineConfig,
        language: str = "ar",
        concurrency: int = 512,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.config = config
        self.language = language
        # A shared semaphore lets several reprocessors (files, languages)
        # run concurrently without oversubscribing the vLLM server.
        self.semaphore = semaphore or asyncio.Semaphore(concurrency)
        self._write_lock = asyncio.Lock()

    def _output_paths(
        self, rejected_path: Path, slice_idx: int | None
    ) -> tuple[str, Path, Path]:
        """(stem, recovered_path, still_rejected_path), slice-aware."""
        stem = rejected_path.stem
        if stem.endswith("_rejected"):
            stem = stem[: -len("_rejected")]
        part = "" if slice_idx is None else f"_p{slice_idx:04d}"
        return (
            stem,
            rejected_path.parent / f"{stem}_recovered{part}.jsonl",
            rejected_path.parent / f"{stem}_still_rejected{part}.jsonl",
        )

    def _all_output_paths(self, rejected_path: Path) -> list[Path]:
        """Every recovered/still-rejected output for resume key collection."""
        stem, recovered, still = self._output_paths(rejected_path, None)
        paths = [recovered, still]
        paths.extend(sorted(rejected_path.parent.glob(f"{stem}_recovered_p*.jsonl")))
        paths.extend(
            sorted(rejected_path.parent.glob(f"{stem}_still_rejected_p*.jsonl"))
        )
        return paths

    async def process_file(
        self,
        rejected_path: Path,
        client: VLLMClient,
        slice_idx: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> dict:
        """Reprocess one *_rejected.jsonl file (or one slice of it)."""
        _stem, recovered_path, still_rejected_path = self._output_paths(
            rejected_path, slice_idx
        )
        label = f"{rejected_path.name}[p{slice_idx:04d}]" if slice_idx is not None \
            else rejected_path.name

        records = load_rejected_records(rejected_path)
        if start is not None or end is not None:
            records = records[start:end]
        # Resume scope = this slice's own outputs only; slices partition the
        # pending list by index, so their key sets are disjoint by design.
        done = load_done_keys(recovered_path, still_rejected_path)
        pending = [
            r for r in records
            if (r.audio_filepath, r.text) not in done
        ]

        LOGGER.info(
            "%s: %d unique records (%d already done, %d to process)",
            label,
            len(records),
            len(records) - len(pending),
            len(pending),
        )

        stats = {
            "file": label,
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
                        label,
                        total_done,
                        len(pending),
                        stats["recovered"],
                        stats["still_rejected"],
                    )

        await asyncio.gather(*(_process_one(r) for r in pending))
        return stats


async def run_reprocess(args: argparse.Namespace) -> None:
    languages = (
        [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        if args.languages else None
    )
    pairs = discover_rejected_files(
        Path(args.rejected_dir), languages=languages, files=args.files,
        default_language=args.language,
    )
    if not pairs:
        LOGGER.error("No *_rejected.jsonl files found in %s", args.rejected_dir)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Slice big files, then deal slices round-robin across nodes.         #
    # ------------------------------------------------------------------ #
    slices: list[ReprocessSlice] = []
    for language, path in pairs:
        unique_count = len(load_rejected_records(path))
        if args.num_nodes > 1 and unique_count > args.slice_size:
            num_slices = min(
                -(-unique_count // args.slice_size),   # ceil div
                args.num_nodes * args.slices_per_node,
            )
            step = -(-unique_count // num_slices)
            for i in range(num_slices):
                slices.append(ReprocessSlice(
                    language=language, path=path, slice_idx=i,
                    start=i * step, end=min((i + 1) * step, unique_count),
                ))
        else:
            slices.append(ReprocessSlice(language=language, path=path))

    if args.num_nodes > 1:
        node_slices = [s for i, s in enumerate(slices)
                       if i % args.num_nodes == args.node_rank]
        LOGGER.info(
            "Node %d/%d: %d of %d slices across %d files (languages: %s)",
            args.node_rank, args.num_nodes, len(node_slices), len(slices),
            len({(s.language, s.path) for s in slices}),
            ", ".join(sorted({s.language for s in slices})),
        )
    else:
        node_slices = slices
    if not node_slices:
        LOGGER.info("Nothing assigned to this node.")
        return

    # ------------------------------------------------------------------ #
    # One reprocessor per language (ar: specialized agents; rest: generic #
    # language-aware agents). All share one semaphore so the vLLM server  #
    # sees at most --concurrency in-flight requests.                      #
    # ------------------------------------------------------------------ #
    semaphore = asyncio.Semaphore(args.concurrency)
    reprocessors: dict[str, RejectedReprocessor] = {}
    for slice_ in node_slices:
        if slice_.language in reprocessors:
            continue
        if args.config:
            config = PipelineConfig.from_yaml(args.config)
        elif slice_.language == "ar":
            config = PipelineConfig.from_yaml(args.arabic_config)
        else:
            config = PipelineConfig.from_yaml(args.multilingual_config)
        config.validate()
        reprocessors[slice_.language] = RejectedReprocessor(
            config, language=slice_.language,
            concurrency=args.concurrency, semaphore=semaphore,
        )

    start = time.time()
    endpoints = {
        (r.config.vllm_host, r.config.vllm_port)
        for r in reprocessors.values()
    }
    if len(endpoints) > 1:
        LOGGER.error(
            "Arabic and multilingual configs point at different vLLM "
            "endpoints %s; align vllm_host/vllm_port (one server per node).",
            sorted(endpoints),
        )
        sys.exit(1)
    async with VLLMClient(next(iter(reprocessors.values())).config) as client:
        await client.wait_for_health(max_wait_seconds=300)

        async def _run_slice(slice_: ReprocessSlice) -> dict:
            return await reprocessors[slice_.language].process_file(
                slice_.path, client,
                slice_idx=slice_.slice_idx,
                start=slice_.start, end=slice_.end,
            )

        # Files/slices of a node run concurrently against the shared
        # semaphore — no more sequential per-file draining.
        all_stats = await asyncio.gather(*(_run_slice(s) for s in node_slices))

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
        "Fold recovered records into a training snapshot with:\n"
        "  python3 scripts/assemble_training_manifests.py "
        "--version vNEXT  # folds all *_recovered[_pNNNN] parts back in"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess rejected records through the improved pipeline"
    )
    parser.add_argument(
        "--rejected-dir",
        type=str,
        required=True,
        help="Directory containing *_rejected.jsonl files "
             "(language-subdir layout <dir>/<lang>/ auto-detects language)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Force one pipeline config for ALL files (default: auto — "
             "arabic config for ar, multilingual config for the rest)",
    )
    parser.add_argument("--arabic-config", type=str,
                        default=DEFAULT_ARABIC_CONFIG)
    parser.add_argument("--multilingual-config", type=str,
                        default=DEFAULT_MULTILINGUAL_CONFIG)
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language dirs to include (default: all)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ar",
        help="Language for flat (non-subdir) rejected dirs (default: ar)",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated rejected file names (default: all round-1 "
             "*_rejected.jsonl; still-rejected piles must be named here)",
    )
    parser.add_argument("--node-rank", type=int, default=0,
                        help="This node's rank (default: 0)")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Total nodes sharing the work (default: 1)")
    parser.add_argument(
        "--slice-size",
        type=int,
        default=50_000,
        help="Files larger than this are split into slices so multiple "
             "nodes can share them (default: 50000)",
    )
    parser.add_argument(
        "--slices-per-node",
        type=int,
        default=4,
        help="Max slices of one big file per node; higher = better tail "
             "load balance when few files dominate (default: 4)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=512,
        help="Max concurrent in-flight records per node (default: 512)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    asyncio.run(run_reprocess(args))


if __name__ == "__main__":
    main()


__all__ = ["RejectedReprocessor", "load_rejected_records", "main"]
