"""Manifest I/O: shard reading, writing, and atomic merge.

Handles JSONL manifest files with Lustre-safe write patterns.
Each worker writes to its own shard file — no concurrent writes to the same file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


LOGGER = logging.getLogger("data_processing.manifest")

PROCESSING_VERSION = "0.2.0"


@dataclass
class ManifestRecord:
    """A single record from a JSONL manifest."""

    line_number: int
    audio_filepath: str
    text: str
    duration: float | None
    raw: dict[str, Any]


@dataclass
class CleanedRecord:
    """A cleaned record ready for output."""

    audio_filepath: str
    text: str
    duration: float | None
    original_text: str
    dialect: str
    confidence: float
    changes: list[str]
    processing_version: str = PROCESSING_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to output JSONL schema."""
        result: dict[str, Any] = {
            "audio_filepath": self.audio_filepath,
            "text": self.text,
            "original_text": self.original_text,
            "dialect": self.dialect,
            "confidence": self.confidence,
            "changes": self.changes,
            "processing_version": self.processing_version,
        }
        if self.duration is not None:
            result["duration"] = self.duration
        return result


def read_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    """Stream records from a JSONL manifest.

    Yields ManifestRecord for each valid line. Skips blank lines.
    Raises ValueError for malformed JSON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")

            # Extract text (support both 'text' and 'transcript' keys)
            text = record.get("text") or record.get("transcript")
            if not isinstance(text, str) or not text.strip():
                # Skip records without usable text
                continue

            # Extract audio path (support both 'audio_filepath' and 'wav_path')
            audio = record.get("audio_filepath") or record.get("wav_path")
            if not isinstance(audio, str) or not audio.strip():
                continue

            duration = record.get("duration")
            if duration is not None:
                try:
                    duration = float(duration)
                except (TypeError, ValueError):
                    duration = None

            yield ManifestRecord(
                line_number=line_number,
                audio_filepath=audio,
                text=text.strip(),
                duration=duration,
                raw=record,
            )


def count_manifest_records(path: str | Path) -> int:
    """Count the number of valid records in a manifest without loading all."""
    count = 0
    for _ in read_manifest(path):
        count += 1
    return count


def shard_manifest(
    path: str | Path,
    num_shards: int,
    output_dir: str | Path,
    prefix: str = "shard",
    start: int = 0,
    end: int | None = None,
    skip_keys: set[tuple[str, str]] | None = None,
) -> list[Path]:
    """Split a manifest into N shard files using round-robin distribution.

    ``start``/``end`` select a slice of the valid-record index space
    (end=None means EOF) so one huge manifest can be spread across nodes.
    ``skip_keys`` drops records already processed in earlier runs
    (keyed by (audio_filepath, text)).

    Returns list of shard file paths.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [output_dir / f"{prefix}_{i:04d}.jsonl" for i in range(num_shards)]
    shard_handles = [p.open("w", encoding="utf-8") for p in shard_paths]

    written = 0
    skipped = 0
    try:
        for idx, record in enumerate(read_manifest(path)):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            if skip_keys and (record.audio_filepath, record.text) in skip_keys:
                skipped += 1
                continue
            line = json.dumps(record.raw, ensure_ascii=False)
            shard_handles[written % num_shards].write(line + "\n")
            written += 1
    finally:
        for handle in shard_handles:
            handle.close()

    LOGGER.info(
        "Sharded %s[%d:%s] into %d files in %s (%d records, %d already done)",
        path.name,
        start,
        end if end is not None else "EOF",
        num_shards,
        output_dir,
        written,
        skipped,
    )
    return shard_paths


def collect_processed_keys(output_dir: str | Path, stem: str) -> set[tuple[str, str]]:
    """Collect (audio_filepath, source_text) keys already processed.

    Scans every prior output of the manifest — cleaned, shards, rejected,
    and slice variants (``<stem>*.jsonl``) — so a range run can skip work a
    partial full-manifest run (or another slice) already finished.
    """
    output_dir = Path(output_dir)
    keys: set[tuple[str, str]] = set()
    if not output_dir.is_dir():
        return keys
    for path in sorted(output_dir.glob(f"{stem}*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    audio = record.get("audio_filepath") or record.get("wav_path")
                    source_text = record.get("original_text") or record.get("text")
                    if audio and source_text:
                        keys.add((str(audio), str(source_text).strip()))
        except OSError as exc:
            LOGGER.warning("Could not read %s for dedup: %s", path, exc)
    LOGGER.info("Collected %d already-processed keys for %s*", len(keys), stem)
    return keys


def write_shard_atomic(path: str | Path, records: list[CleanedRecord]) -> None:
    """Atomically write cleaned records to a shard file.

    Writes to a temp file first, then renames (atomic on POSIX).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".jsonl"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                line = json.dumps(record.to_dict(), ensure_ascii=False)
                handle.write(line + "\n")
        os.replace(tmp_path, path)  # Atomic rename
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_to_shard(path: str | Path, records: list[CleanedRecord]) -> None:
    """Append cleaned records to an existing shard file.

    Use this for incremental writes during processing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record.to_dict(), ensure_ascii=False)
            handle.write(line + "\n")


def merge_shards(shard_paths: list[Path], output_path: str | Path) -> int:
    """Merge multiple shard files into a single output manifest.

    Returns the total number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=".tmp_merge_", suffix=".jsonl"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out_handle:
            for shard_path in sorted(shard_paths):
                if not shard_path.is_file():
                    LOGGER.warning("Shard not found, skipping: %s", shard_path)
                    continue
                with shard_path.open("r", encoding="utf-8") as in_handle:
                    for line in in_handle:
                        stripped = line.strip()
                        if stripped:
                            out_handle.write(stripped + "\n")
                            total_records += 1
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    LOGGER.info("Merged %d shards into %s (%d records)", len(shard_paths), output_path, total_records)
    return total_records


def read_cleaned_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a cleaned manifest and yield records as dicts."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


__all__ = [
    "CleanedRecord",
    "ManifestRecord",
    "append_to_shard",
    "collect_processed_keys",
    "count_manifest_records",
    "merge_shards",
    "read_cleaned_manifest",
    "read_manifest",
    "shard_manifest",
    "write_shard_atomic",
]
