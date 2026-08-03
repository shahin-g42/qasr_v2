"""Backfill missing ``duration`` fields in JSONL manifests.

Generic over files and schemas: every ``*.jsonl`` (or ``--pattern``) under
``--dir`` is scanned recursively. Any entry without a usable duration —
key absent, ``null``, non-numeric, zero or negative — gets its duration
recalculated from the audio header (soundfile.info — no decode, so header
reads stay fast at large scale). The audio path is taken from
``audio_filepath`` or ``wav_path`` (same fallback as the training loader).
Entries that already carry a valid duration are left untouched.

Files are rewritten atomically (tmp + rename) and only when at least one
record was filled. Records whose audio is missing/unreadable are left
unchanged and counted.

Usage (on the cluster, where audio paths resolve):
    python scripts/backfill_durations.py --dir cleaned_manifests --workers 48
    python scripts/backfill_durations.py --dir cleaned_manifests/hi --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

# Same rewrite the training loader applies (src/qasr/data.py)
_VAST = re.compile(r"^/vast")


def _audio_path(entry: dict) -> str:
    """Audio path from audio_filepath or wav_path (training-loader fallback)."""
    for key in ("audio_filepath", "wav_path"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _needs_duration(entry: dict) -> bool:
    """True when the entry has no usable duration (missing/null/invalid)."""
    value = entry.get("duration")
    if value is None:
        return True
    try:
        dur = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(dur) or dur <= 0


def _probe(path: str) -> tuple[str, float | None]:
    """Read duration from the audio header. Returns (path, seconds|None)."""
    resolved = _VAST.sub("/lustrefs/taiga/vast40", path)
    try:
        info = sf.info(resolved)
        if info.frames > 0 and info.samplerate > 0:
            return path, round(info.frames / info.samplerate, 3)
    except Exception:
        pass
    return path, None


def backfill_file(
    path: Path, durations: dict[str, float], dry_run: bool
) -> tuple[int, int]:
    """Rewrite one JSONL with filled durations. Returns (filled, unresolved)."""
    filled = unresolved = 0
    out_lines: list[str] = []
    for line in path.open("r", encoding="utf-8"):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            out_lines.append(line.rstrip("\n"))
            continue
        if _needs_duration(entry):
            dur = durations.get(_audio_path(entry))
            if dur is not None:
                entry["duration"] = dur
                filled += 1
            else:
                unresolved += 1
        out_lines.append(json.dumps(entry, ensure_ascii=False))

    if filled and not dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return filled, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill missing durations")
    parser.add_argument("--dir", required=True, help="Manifest dir (recursed)")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(
        p for p in root.rglob(args.pattern)
        if p.is_file() and not p.name.endswith(".tmp")
    )
    if not files:
        print(f"No files match {args.pattern} under {root}", file=sys.stderr)
        return 1

    # Pass 1: collect audio paths of records missing duration
    missing_paths: set[str] = set()
    per_file_missing: dict[Path, int] = {}
    for path in files:
        count = 0
        for line in path.open("r", encoding="utf-8"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _needs_duration(entry):
                ap = _audio_path(entry)
                if ap:
                    missing_paths.add(ap)
                count += 1
        if count:
            per_file_missing[path] = count
    print(f"{len(files)} files scanned; {len(per_file_missing)} need backfill; "
          f"{len(missing_paths):,} unique audio files to probe")
    if not missing_paths:
        return 0

    # Pass 2: probe headers in parallel (I/O-bound header reads -> threads)
    durations: dict[str, float] = {}
    probed = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ap, dur in pool.map(_probe, sorted(missing_paths)):
            probed += 1
            if dur is not None:
                durations[ap] = dur
            else:
                failed += 1
            if probed % 50000 == 0:
                print(f"  probed {probed:,}/{len(missing_paths):,} "
                      f"({failed:,} unreadable)")
    print(f"Probed {probed:,} files: {len(durations):,} ok, {failed:,} unreadable")

    # Pass 3: rewrite manifests
    total_filled = total_unresolved = 0
    for path in per_file_missing:
        filled, unresolved = backfill_file(path, durations, args.dry_run)
        total_filled += filled
        total_unresolved += unresolved
        if unresolved:
            print(f"  {path}: {filled} filled, {unresolved} UNRESOLVED")
    action = "would fill" if args.dry_run else "filled"
    print(f"Done: {action} {total_filled:,} records; "
          f"{total_unresolved:,} left without duration (audio unreadable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
