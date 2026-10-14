#!/usr/bin/env python3
"""Author a whole oracle worlds file into one stage directory.

    uv run python tools/oracle_batch.py --worlds runs/worlds/pilot-v1.jsonl \
        --source oracle-templates --registry configs/data/sources.yaml \
        --output runs/stages/oracle-v1

One world per JSONL line; row identifiers become <prefix><index:06d>.
"""

from __future__ import annotations

import argparse
import sys

from jev_contracts import ValidationError
from jev_data.batch import oracle_batch
from jev_data.io import parse_jsonl, read_bytes
from jev_data.registry import Registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", required=True, help="JSONL file, one world per line")
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default="oracle-")
    parser.add_argument("--split", default="train",
                        choices=["train", "development", "calibration", "test"])
    parser.add_argument("--calibration-subset",
                        choices=["temperature_fit", "policy_selection", "risk_verification"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--purpose", choices=["training", "evaluation"], default="training")
    args = parser.parse_args(argv)
    try:
        worlds = parse_jsonl(read_bytes(args.worlds))
        rows = oracle_batch(worlds, prefix=args.prefix, source_id=args.source,
                            registry=Registry.load(args.registry), split=args.split,
                            calibration_subset=args.calibration_subset,
                            allow_test=args.allow_test, purpose=args.purpose,
                            output=args.output)
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"oracle_batch: {type(exc).__name__}: operation rejected; check inputs",
              file=sys.stderr)
        return 2
    print(f"authored {len(rows)} oracle rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
