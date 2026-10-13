"""Uniform record-line reservoir sampling with a reusable, bounded-size cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from .io import digest, read_json


def sample_record_lines(path, *, limit, seed, cache_directory):
    """Read raw JSONL once; only the retained records need JSON/audio validation.

    Algorithm R chooses each nonblank record line with probability min(1, k/N).
    Cached source identity uses path, size, mtime and ctime, avoiding another full
    read. Source contents are SHA256-hashed during the original streaming pass.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("candidates_per_manifest must be a positive integer")
    path = Path(path).resolve()
    stat = path.stat()
    identity = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    key = {"schema": "qasr-record-reservoir-v1", "source": identity, "limit": limit, "seed": seed}
    cache_directory = Path(cache_directory).expanduser().resolve()
    cache_path = cache_directory / f"{digest(key)}.json"
    if cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("key") != key or digest(cached["records"]) != cached.get("records_sha256"):
            raise ValueError(f"Invalid manifest sample cache: {cache_path}")
        print(f"Using cached record sample: {path}", file=sys.stderr, flush=True)
        return cached["records"], {
            **cached["metadata"],
            "cache_hit": True,
            "cache_path": str(cache_path),
        }

    print(
        f"Sampling record lines once: {path} (keep at most {limit:,})", file=sys.stderr, flush=True
    )
    rng = random.Random(int(digest([seed, str(path)]), 16))
    reservoir, sha = [], hashlib.sha256()
    records_seen, byte_offset = 0, 0
    started = time.monotonic()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            sha.update(line)
            start = byte_offset
            byte_offset += len(line)
            if not line.strip():
                continue
            records_seen += 1
            slot = len(reservoir) if records_seen <= limit else rng.randrange(records_seen)
            if slot < limit:
                # Keep bytes until the pass finishes; never parse discarded records.
                entry = (line_number, start, line)
                if records_seen <= limit:
                    reservoir.append(entry)
                else:
                    reservoir[slot] = entry
            if records_seen % 1000000 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"  {records_seen:,} record lines read; {len(reservoir):,} retained; "
                    f"{records_seen / elapsed:,.0f} records/s",
                    file=sys.stderr,
                    flush=True,
                )
    current = path.stat()
    if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ):
        raise ValueError(f"Manifest changed during sampling: {path}")
    records = [
        {"line_number": number, "byte_offset": offset, "record_json": raw.decode("utf-8")}
        for number, offset, raw in sorted(reservoir)
    ]
    metadata = {
        "method": "uniform_record_reservoir",
        "source_identity": identity,
        "sha256": sha.hexdigest(),
        "records_read": records_seen,
        "candidate_records": len(records),
        "all_records_considered": records_seen <= limit,
        "selection_seed": seed,
        "candidate_limit": limit,
        "cache_validation": "source path, size, mtime_ns and ctime_ns; sampled-record checksum",
    }
    cached = {
        "key": key,
        "records": records,
        "records_sha256": digest(records),
        "metadata": metadata,
    }
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_directory, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(cached, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    print(
        f"  Cached {len(records):,} of {records_seen:,} record lines in {elapsed:.1f}s; "
        "only cached candidates will be parsed",
        file=sys.stderr,
        flush=True,
    )
    return records, {**metadata, "cache_hit": False, "cache_path": str(cache_path)}
