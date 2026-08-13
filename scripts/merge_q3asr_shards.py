#!/usr/bin/env python3
"""Merge the per-node shards written by ``prepare_q3asr_filter.py --num-nodes``.

Each node emits ``<split>_<lang>_<corpus>.rank<N>of<M>.jsonl`` because
``write_manifest`` truncates its target: if all nodes shared one path, only
whichever finished last would survive. This concatenates each rank family
back into the single file the training configs expect.

Safety properties, in order of how much they'd hurt to get wrong:

* **Completeness.** Each node writes ``_shard_done.rank<N>of<M>.json`` as its
  last act, and this refuses to merge unless all M markers are present — so an
  interrupted or still-running node cannot silently produce a short manifest
  that looks finished. Completion is deliberately NOT inferred from the
  per-language shard files: a rank having no ``ml`` records is normal (small
  languages leave many rank/language combinations empty), whereas a missing
  marker never is. Override with ``--allow-missing`` if a node is genuinely
  never coming back.
* **Atomicity.** Writes to a tmp file and ``os.replace``s it into place, the
  same pattern as the cleaning pipeline's checkpoints. A crash mid-merge
  leaves the previous file intact rather than a half-written one.
* **Idempotence.** Shards are left on disk and the merged file is rebuilt from
  them, so re-running is safe and never double-appends.

Usage::

    PYTHONPATH=src python3 scripts/merge_q3asr_shards.py \
        --output-dir /lustrefs/.../q3asr_sft_manifests --num-nodes 8

    # keep going if a rank produced no records for some language
    ... --num-nodes 8 --allow-missing

    # remove the shards once the merged files are verified
    ... --num-nodes 8 --clean
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger("qasr.merge_q3asr_shards")

# <stem>.rank<N>of<M>.jsonl
SHARD_RE = re.compile(r"^(?P<stem>.+)\.rank(?P<rank>\d+)of(?P<total>\d+)\.jsonl$")
# _shard_done.rank<N>of<M>.json, written last by each node.
DONE_RE = re.compile(r"^_shard_done\.rank(?P<rank>\d+)of(?P<total>\d+)\.json$")


def find_done_markers(root: Path) -> dict[int, dict[int, Path]]:
    """Completion markers grouped by total -> {rank: path}."""
    markers: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("_shard_done.rank*of*.json")):
        match = DONE_RE.match(path.name)
        if match:
            markers[int(match["total"])][int(match["rank"])] = path
    return markers


def find_shard_families(root: Path) -> dict[tuple[Path, str, int], dict[int, Path]]:
    """Group shard files by (directory, stem, total) -> {rank: path}."""
    families: dict[tuple[Path, str, int], dict[int, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("*.jsonl")):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        key = (path.parent, match["stem"], int(match["total"]))
        families[key][int(match["rank"])] = path
    return families


def merge_family(
    directory: Path, stem: str, total: int, shards: dict[int, Path]
) -> tuple[Path, int]:
    """Concatenate one rank family into ``<stem>.jsonl``. Returns (path, lines)."""
    target = directory / f"{stem}.jsonl"
    tmp = directory / f"{stem}.jsonl.tmp"
    lines = 0
    with tmp.open("w", encoding="utf-8") as out:
        for rank in sorted(shards):
            with shards[rank].open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        lines += 1
    os.replace(tmp, target)
    return target, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="The --output-dir that prepare_q3asr_filter.py wrote to",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=0,
        help="Expected rank count. Defaults to the 'of<M>' in the file names; "
        "pass it explicitly to assert the number you launched.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Merge even when some nodes never wrote a completion marker "
        "(normally an error, since a still-running node would yield a "
        "silently short manifest)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the shard files after a successful merge",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    root = Path(args.output_dir)
    if not root.is_dir():
        LOGGER.error("Not a directory: %s", root)
        return 2

    families = find_shard_families(root)
    if not families:
        LOGGER.error(
            "No .rankNofM.jsonl shards under %s — was this run with "
            "--num-nodes > 1?",
            root,
        )
        return 2

    incomplete: list[str] = []
    for (directory, stem, total), _shards in sorted(families.items()):
        if args.num_nodes and args.num_nodes != total:
            LOGGER.error(
                "%s/%s: files say 'of%d' but --num-nodes is %d",
                directory,
                stem,
                total,
                args.num_nodes,
            )
            return 2

    # Completion is judged by the markers, not by which per-language shards
    # exist: a rank with no records for some language writes no file for it,
    # which is normal and must not look like a failure.
    totals = {total for (_d, _s, total) in families}
    markers = find_done_markers(root)
    for total in sorted(totals):
        expected = args.num_nodes or total
        present = set(markers.get(total, {}))
        missing = sorted(set(range(expected)) - present)
        if missing:
            incomplete.append(
                f"{expected}-node run: no completion marker from rank(s) {missing}"
            )

    if incomplete and not args.allow_missing:
        for line in incomplete:
            LOGGER.error("INCOMPLETE %s", line)
        LOGGER.error(
            "Refusing to merge: a missing completion marker means that node is "
            "unfinished or failed, and merging now would produce a manifest "
            "that looks complete but is not. Re-run the missing rank(s), or "
            "pass --allow-missing if a node is never coming back."
        )
        return 1
    for line in incomplete:
        LOGGER.warning("proceeding despite: %s", line)

    total_lines = 0
    for (directory, stem, total), shards in sorted(families.items()):
        target, lines = merge_family(directory, stem, total, shards)
        total_lines += lines
        LOGGER.info("%s <- %d shard(s), %d record(s)", target, len(shards), lines)
        if args.clean:
            for path in shards.values():
                path.unlink()

    LOGGER.info(
        "Merged %d file(s), %d record(s) total%s",
        len(families),
        total_lines,
        " (shards removed)" if args.clean else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
