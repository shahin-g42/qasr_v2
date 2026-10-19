from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .core import load_plan, prepare, sha256
from .review import import_reviews, review_pack, score
from .runtime import doctor, propose


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Arabic reference pilot: propose, listen, review, compare."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument(
        "--prediction", action="append", required=True, help="NAME=forced-Arabic JSONL path"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-dataset", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261019)
    for command in ("doctor", "propose"):
        p = commands.add_parser(command)
        p.add_argument("--url", default="http://localhost:8010/v1")
        p.add_argument("--model", default="corrector")
        p.add_argument(
            "--checkpoint",
            help="Optional expected checkpoint; default records server metadata and routes by --model alias",
        )
        p.add_argument("--timeout", type=float, default=180)
        p.add_argument("--plan", type=Path, required=True)
        if command == "doctor":
            p.add_argument("--audio-root", type=Path)
        else:
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--revision", default="unrecorded")
            p.add_argument("--concurrency", type=int, default=16)
            p.add_argument("--max-tokens", type=int, default=4096)
            p.add_argument("--smoke", action="store_true")
    p = commands.add_parser("review-pack")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proposals", type=Path)
    p.add_argument("--audio-root", type=Path)
    p.add_argument("--metadata-only", action="store_true")
    p = commands.add_parser("import-reviews")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--reviews", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--reviewed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("status")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--proposals", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            inputs = []
            for entry in args.prediction:
                name, separator, path = entry.partition("=")
                if not separator:
                    raise ValueError("--prediction must be NAME=PATH")
                inputs.append((name, Path(path)))
            result = prepare(inputs, args.output, args.per_dataset, args.seed)
            result = {k: v for k, v in result.items() if k not in ("sources", "policy")}
        elif args.command == "doctor":
            _, rows = load_plan(args.plan)
            for row in rows:
                audio = (
                    args.audio_root / row["dataset"] / f"{row['ordinal']:06d}.npy"
                    if args.audio_root
                    else Path(row["audio"])
                )
                if sha256(audio) != row["audio_sha256"]:
                    raise ValueError(f"Prepared audio changed: {row['id']}")
            result = doctor(args.url, args.model, args.checkpoint, args.timeout)
            result.update({"audio_hashes_checked": len(rows), "cpu_cores_visible": os.cpu_count()})
        elif args.command == "propose":
            result = propose(
                args.plan,
                args.output,
                url=args.url,
                model=args.model,
                checkpoint=args.checkpoint,
                revision=args.revision,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                smoke=args.smoke,
            )
        elif args.command == "review-pack":
            result = review_pack(
                args.plan, args.output, args.proposals, args.audio_root, args.metadata_only
            )
        elif args.command == "import-reviews":
            result = import_reviews(args.plan, args.reviews, args.output)
        elif args.command == "score":
            result = score(args.plan, args.reviewed, args.output)
        else:
            plan, rows = load_plan(args.plan)
            result = {
                "pilot_clips": plan["pilot_clips"],
                "selected_clips": len(rows),
                "flags": plan["flag_counts"],
            }
            if args.proposals:
                from .runtime import load_proposals

                records = load_proposals(args.proposals, plan, rows)
                result["finished_proposals"] = len(records)
                result["missing_or_failed"] = len(rows) - len(records)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(
            args.command == "propose"
            and (
                result["counts"].get("failed", 0) > 0
                or (args.smoke and not result["smoke_checks"]["passed"])
            )
        )
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
