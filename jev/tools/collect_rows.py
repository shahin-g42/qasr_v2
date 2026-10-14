#!/usr/bin/env python3
"""Merge stage directories or JSONL files into one new stage, with optional filters.

    uv run python tools/collect_rows.py --input runs/stages/verified-w0 runs/stages/verified-w1 \
        --tier silver --output runs/stages/verified-v1
    uv run python tools/collect_rows.py --input runs/stages/plan-v1 \
        --split train --output runs/stages/plan-train-v1

Filters select rows; they never bypass per-row validation on the inputs.
--exclude-file drops listed row ids from the selection (one id per line, '#' starts
a comment); exclusions and all dropped ids are recorded in the stage selection.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jev_contracts import ValidationError
from jev_data.batch import collect_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True,
                        help="stage directories or JSONL files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", nargs="*", choices=["bronze", "silver", "gold"])
    parser.add_argument("--split", nargs="*",
                        choices=["train", "development", "calibration", "test"])
    parser.add_argument("--exclude-file", default=None,
                        help="text file of row ids to drop (one per line; '#' starts a comment)")
    parser.add_argument("--stage", default="collected")
    args = parser.parse_args(argv)
    exclude_ids = None
    if args.exclude_file:
        exclude_ids = {line.split("#", 1)[0].strip()
                       for line in Path(args.exclude_file).read_text(encoding="utf-8").splitlines()}
        exclude_ids.discard("")
        exclude_ids = exclude_ids or None
    try:
        rows = collect_rows(args.input, output=args.output, tiers=args.tier,
                            splits=args.split, exclude_ids=exclude_ids, stage=args.stage)
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"collect_rows: {type(exc).__name__}: operation rejected; check inputs",
              file=sys.stderr)
        return 2
    note = f" (excluded {len(exclude_ids)} ids from {args.exclude_file})" if exclude_ids else ""
    print(f"collected {len(rows)} rows -> {args.output}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
