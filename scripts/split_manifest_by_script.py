#!/usr/bin/env python3
"""Split records of one script out of a JSONL manifest into a new manifest.

Built for the eval_ml_inworld case: Hindi (Devanagari) records mixed into
the Malayalam manifest. Classification is deterministic by Unicode block —
no LLM involved:

    Malayalam  U+0D00-0D7F
    Devanagari U+0900-097F  (Hindi)

A record is extracted when the extract-script's characters dominate the
record's Indic characters (>50%) with a small absolute floor. Records with
no Indic script at all (Latin-only, empty) stay in the source and are
reported.

The source is rewritten atomically WITHOUT the extracted records; a
one-time ``<source>.orig`` backup is kept. Extracted records are APPENDED
to the target if it already exists (dedup by audio_filepath).

Usage:
    python3 scripts/split_manifest_by_script.py \
        --input  validated_manifests/eval_ml_inworld.json \
        --output validated_manifests/eval_hi_inworld.json \
        --extract devanagari --dry-run

    # then for real:
    python3 scripts/split_manifest_by_script.py \
        --input  validated_manifests/eval_ml_inworld.json \
        --output validated_manifests/eval_hi_inworld.json \
        --extract devanagari
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Indic Unicode blocks (name -> inclusive range)
SCRIPT_BLOCKS: dict[str, tuple[str, str]] = {
    "devanagari": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "gurmukhi": ("\u0a00", "\u0a7f"),
    "gujarati": ("\u0a80", "\u0aff"),
    "oriya": ("\u0b00", "\u0b7f"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "kannada": ("\u0c80", "\u0cff"),
    "malayalam": ("\u0d00", "\u0d7f"),
}

# Minimum extract-script chars before a record can be extracted — guards
# against a stray single character flipping a record's language.
MIN_SCRIPT_CHARS = 3


def script_counts(text: str) -> dict[str, int]:
    """Count characters per Indic block in text."""
    counts: dict[str, int] = {}
    for ch in text:
        for name, (lo, hi) in SCRIPT_BLOCKS.items():
            if lo <= ch <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def classify(text: str, extract: str) -> str:
    """Return 'extract', 'keep', or 'no_indic' for a record's text."""
    counts = script_counts(text)
    total = sum(counts.values())
    if total == 0:
        return "no_indic"
    extract_chars = counts.get(extract, 0)
    if extract_chars >= MIN_SCRIPT_CHARS and extract_chars > total / 2:
        return "extract"
    return "keep"


def iter_records(path: Path):
    """Yield dict records from a JSONL manifest (JSON-array fallback)."""
    with path.open(encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            # Rare: whole-file JSON array
            yield from json.load(fh)
            return
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN: skipping malformed line {line_no}: {exc}",
                      file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Source manifest (rewritten without extracted records)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Target manifest for extracted records (appended)")
    parser.add_argument("--extract", default="devanagari",
                        choices=sorted(SCRIPT_BLOCKS),
                        help="Script to extract (default: devanagari)")
    parser.add_argument("--text-key", default="text",
                        help="JSON key holding the transcript (default: text)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts and examples only; write nothing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    kept: list[dict] = []
    extracted: list[dict] = []
    no_indic: list[dict] = []
    for rec in iter_records(args.input):
        verdict = classify(str(rec.get(args.text_key, "")), args.extract)
        if verdict == "extract":
            extracted.append(rec)
        elif verdict == "no_indic":
            no_indic.append(rec)  # stays in source, reported separately
            kept.append(rec)
        else:
            kept.append(rec)

    total = len(kept) + len(extracted)
    print(f"input records : {total}")
    print(f"  keep        : {len(kept)}  (incl. {len(no_indic)} with no Indic script)")
    print(f"  extract     : {len(extracted)}  ({args.extract})")
    for rec in extracted[:3]:
        print(f"    e.g. {str(rec.get(args.text_key, ''))[:80]}")
    for rec in no_indic[:3]:
        print(f"    no-indic e.g. {str(rec.get(args.text_key, ''))[:80]}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0
    if not extracted:
        print("\nNothing to extract — source left untouched.")
        return 0

    # One-time backup of the source (never overwrite an existing backup)
    backup = args.input.with_suffix(args.input.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(args.input, backup)
        print(f"backup        : {backup}")

    # Append to target, dedup by audio_filepath against existing content
    existing_keys: set[str] = set()
    if args.output.exists():
        for rec in iter_records(args.output):
            existing_keys.add(rec.get("audio_filepath", ""))
    new_records = [r for r in extracted
                   if r.get("audio_filepath", "") not in existing_keys]
    with args.output.open("a", encoding="utf-8") as fh:
        for rec in new_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"target        : {args.output} (+{len(new_records)} records, "
          f"{len(extracted) - len(new_records)} already present)")

    # Atomic rewrite of the source without the extracted records
    tmp = args.input.with_suffix(args.input.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.input)
    print(f"source        : {args.input} rewritten with {len(kept)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
