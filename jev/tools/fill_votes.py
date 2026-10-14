#!/usr/bin/env python3
"""Interactive vote filler for one reviewer's panel sheet.

Renders each unanswered job readably, asks validated answers, and saves the
sheet after every vote so progress survives Ctrl-C. Elapsed time is measured,
not estimated.

  python3 tools/fill_votes.py --jobs runs/jobs/panels-001/jobs.json \
      --sheet runs/reviews/sheets-panels/answer-reviewer-1.tsv

Commands at the label prompt: 1/2/3 (option number), option id, s = skip until
later, q = save and quit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/data/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/contracts/src"))

VOTE_COLUMNS = ["job_id", "reviewer_id", "label", "independent", "unambiguous",
                "surface_faithful", "evidence", "elapsed_seconds", "flags"]


def save(path: Path, rows: list[list[str]]) -> None:
    text = "\n".join("\t".join(row) for row in [VOTE_COLUMNS] + rows) + "\n"
    path.write_text(text, encoding="utf-8")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--sheet", required=True)
    args = parser.parse_args(argv)

    jobs = {job["job_id"]: job for job in json.loads(Path(args.jobs).read_text())["jobs"]}
    lines = Path(args.sheet).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t") != VOTE_COLUMNS:
        raise SystemExit(f"fill_votes: {args.sheet} is not an answer sheet with the "
                         f"expected header")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    for row in rows:
        if row[0] not in jobs:
            raise SystemExit(f"fill_votes: sheet row references unknown job {row[0]}")

    pending = [row for row in rows if not row[2].strip()]
    print(f"{args.sheet}: {len(rows) - len(pending)}/{len(rows)} already answered, "
          f"{len(pending)} to go. s=skip, q=quit (progress saves after every answer).")
    try:
        for position, row in enumerate(pending, start=1):
            job = jobs[row[0]]
            payload = job["input"]
            options = payload.get("question", {}).get("options", [])
            print(f"\n{'=' * 70}\n[{position}/{len(pending)}]  job {row[0]}")
            started = time.time()
            print(f"STATE:\n{payload.get('state', '')}")
            print(f"QUESTION:\n{payload.get('question', {}).get('text', payload.get('question'))}")
            for index, option in enumerate(options, start=1):
                print(f"  {index}) {option['id']}: {option.get('description', '')}")
            raw = input("label (1/2/3, id, s=skip, q=quit): ").strip()
            if raw.lower() == "q":
                break
            if raw.lower() == "s":
                continue
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                label = options[int(raw) - 1]["id"]
            elif raw in {option["id"] for option in options}:
                label = raw
            else:
                print(f"  invalid label {raw!r}; job kept for later")
                continue
            row[2] = label
            for column, prompt, default in ((3, "independent (y/n)", "y"),
                                            (4, "unambiguous (y/n)", "y"),
                                            (5, "surface_faithful (y/n)", "y")):
                answer = ask(prompt, default).lower()
                if answer not in {"y", "n"}:
                    print("  invalid boolean; job kept for later")
                    row[2] = ""
                    break
                row[column] = "true" if answer == "y" else "false"
            if row[2] == "" or row[3] == "":
                continue
            row[6] = ask("evidence (one Arabic line)")
            if not row[6]:
                print("  empty evidence; job kept for later")
                row[2] = row[3] = row[4] = row[5] = ""
                continue
            elapsed = min(max(time.time() - started, 0.1), 86399.9)
            row[7] = f"{elapsed:.1f}"
            row[8] = ask("flags (comma-separated, empty if none)")
            row[8] = "|".join(flag.strip() for flag in row[8].split(",") if flag.strip())
            save(Path(args.sheet), rows)
            done = len(rows) - sum(1 for r in rows if not r[2].strip())
            print(f"  saved ({done}/{len(rows)} done)")
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved")
    save(Path(args.sheet), rows)
    remaining = sum(1 for row in rows if not row[2].strip())
    print(f"\n{args.sheet}: {len(rows) - remaining}/{len(rows)} answered, {remaining} left")
    if remaining:
        print("re-run the same command to continue this sheet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
