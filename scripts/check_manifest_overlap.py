#!/usr/bin/env python3
"""Report audio overlap between two sets of manifests.

Written for one question: does the incoming q3asr SFT data duplicate audio
that is already in ``training_manifests/v4.0/``? Both carry the corpus name
``q3asr``, and re-importing 9M already-trained records would silently
double-weight them.

Overlap is keyed on the audio FILE NAME, not the full path, because the two
sides resolve their paths differently (relative source paths vs. absolute
cluster paths, plus the ``/vast`` -> ``/lustrefs/taiga/vast40`` rewrite).
File names in this corpus are UUIDs, so they are safe join keys.

Memory: only the smaller ``--new`` side is held as a set; ``--existing`` is
streamed. A 9M-line manifest costs one pass and no extra RAM.

Usage::

    # incoming SFT files vs. everything already in v4.0
    PYTHONPATH=src python3 scripts/check_manifest_overlap.py \
        --new      /lustrefs/.../q3asr/json/sft/train_filter.jsonl \
                   /lustrefs/.../q3asr/json/sft/eval_filter.jsonl \
        --existing training_manifests/v4.0

    # after transforming, re-check the produced manifests
    PYTHONPATH=src python3 scripts/check_manifest_overlap.py \
        --new      /lustrefs/.../q3asr_sft_manifests \
        --existing training_manifests/v4.0

Exit code is 1 when overlap is found, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

AUDIO_KEYS = ("audio_filepath", "audio", "wav_path")


def expand(paths: list[str]) -> list[Path]:
    """Accept files or directories; directories expand to *.jsonl recursively."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            out.append(path)
        else:
            print(f"WARN: not found, skipping: {path}", file=sys.stderr)
    return out


def iter_audio_names(path: Path) -> Iterator[str]:
    """Yield the basename of every audio reference in a JSONL manifest."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for key in AUDIO_KEYS:
                value = record.get(key)
                if isinstance(value, str) and value:
                    yield value.rsplit("/", 1)[-1]
                    break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--new", nargs="+", required=True, help="Incoming manifests (files or dirs)"
    )
    parser.add_argument(
        "--existing",
        nargs="+",
        required=True,
        help="Manifests already in use (files or dirs)",
    )
    parser.add_argument(
        "--show", type=int, default=10, help="Example overlapping names to print"
    )
    args = parser.parse_args(argv)

    new_files = expand(args.new)
    existing_files = expand(args.existing)
    if not new_files or not existing_files:
        print("ERROR: nothing to compare", file=sys.stderr)
        return 2

    # Hold only the new side in memory.
    new_names: set[str] = set()
    for path in new_files:
        before = len(new_names)
        new_names.update(iter_audio_names(path))
        print(f"  new      {path}  (+{len(new_names) - before:,} unique)")
    print(f"\n{len(new_names):,} unique audio files on the new side\n")

    hits: Counter[str] = Counter()
    examples: list[str] = []
    scanned = 0
    for path in existing_files:
        local = 0
        for name in iter_audio_names(path):
            scanned += 1
            if name in new_names:
                local += 1
                if len(examples) < args.show:
                    examples.append(f"{name}  <- {path}")
        if local:
            hits[str(path)] = local
            print(f"  OVERLAP  {path}: {local:,} record(s)")

    total = sum(hits.values())
    print(f"\nscanned {scanned:,} existing record(s) across {len(existing_files)} file(s)")
    print(f"overlapping records: {total:,}")
    if total:
        pct = 100.0 * total / max(1, len(new_names))
        print(f"that is {pct:.2f}% of the new side's unique audio\n")
        for line in examples:
            print(f"  e.g. {line}")
        print(
            "\nVERDICT: OVERLAP — importing as-is would double-weight this audio. "
            "Drop the overlapping records or skip the affected corpora."
        )
        return 1
    print("\nVERDICT: no overlap — the new data is disjoint from the existing set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
