#!/usr/bin/env python3
"""Assemble cleaned + reprocessed manifests into a versioned training snapshot.

Collects, per language, the accepted output of the cleaning pipeline and any
reprocess recoveries, merges them with correct precedence, and writes a
self-contained, training-ready snapshot:

    <output-dir>/<version>/<lang>/<corpus>.jsonl
    <output-dir>/<version>/MANIFEST.json      (provenance + counts)

Merge rules (per corpus):
  1. Base = <corpus>_cleaned.jsonl when present, else <corpus>_shard_*.jsonl
     (never both — corpora exist in BOTH forms with identical content).
  2. Overlay = <corpus>_recovered.jsonl and <corpus>_suspect_recovered.jsonl:
     a record whose (audio_filepath, original_text) matches a base record
     REPLACES it (audit-suspect re-clean); otherwise it is APPENDED
     (rejected-record recovery).
  3. Excluded: *_rejected, *_still_rejected, *_suspect, *_selected samples,
     reports, backups, tmp files.
  4. Corpora with a `_as_<lang>` stem suffix (e.g. eval_ml_inworld_as_hi)
     are routed to the target language's output directory.

Output records are stripped to training fields {audio_filepath, text,
duration} (duration kept only when present) unless --keep-metadata.
Snapshots are immutable-by-convention: each run rebuilds its --version
directory from scratch (atomic per-file writes), so the next cleaning round
simply assembles into a new version (v2.0, ...) without touching v1.0.

Usage:
    python3 scripts/assemble_training_manifests.py \
        --input-dir cleaned_manifests --output-dir training_manifests \
        --version v1.0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

TRAINING_FIELDS = ("audio_filepath", "text", "duration")

# Suffixes that identify a file's role within its corpus (longest first so
# e.g. `_suspect_recovered` wins over `_recovered`).
ROLE_SUFFIXES = (
    ("_suspect_recovered", "overlay"),
    ("_still_rejected", "exclude"),
    ("_suspect", "exclude"),
    ("_recovered", "overlay"),
    ("_rejected", "exclude"),
    ("_selected", "exclude"),
    ("_cleaned", "base_merged"),
)

_SHARD_RE = re.compile(r"^(?P<corpus>.+)_shard_\d{4}$")
_AS_LANG_RE = re.compile(r"^(?P<corpus>.+)_as_(?P<lang>[a-z]{2})$")


def classify_file(path: Path) -> tuple[str, str] | None:
    """Return (corpus, role) for a manifest file, or None to ignore it."""
    if path.suffix != ".jsonl" or path.name.endswith((".tmp", ".orig")):
        return None
    stem = path.stem
    shard = _SHARD_RE.match(stem)
    if shard:
        return shard.group("corpus"), "base_shard"
    for suffix, role in ROLE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], role
    # Bare <corpus>.jsonl (no recognized suffix): treat as merged base
    return stem, "base_merged"


def record_key(rec: dict) -> tuple[str, str]:
    """Identity of a record across cleaning passes."""
    return rec.get("audio_filepath", ""), rec.get("original_text", "")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN: {path.name}:{line_no} malformed, skipped: {exc}",
                      file=sys.stderr)


def strip_record(rec: dict) -> dict:
    out = {k: rec[k] for k in TRAINING_FIELDS if rec.get(k) is not None}
    return out


def assemble_corpus(
    base_files: list[Path], overlay_files: list[Path]
) -> tuple[list[dict], dict]:
    """Merge base + overlay records with replace-or-append semantics."""
    records: dict[tuple[str, str], dict] = {}
    stats = {"base": 0, "replaced": 0, "appended": 0,
             "base_dupes": 0, "missing_duration": 0}

    for path in sorted(base_files):
        for rec in read_jsonl(path):
            key = record_key(rec)
            if key in records:
                stats["base_dupes"] += 1  # defensive: identical shard overlap
                continue
            records[key] = rec
            stats["base"] += 1

    for path in sorted(overlay_files):
        for rec in read_jsonl(path):
            key = record_key(rec)
            if key in records:
                stats["replaced"] += 1
            else:
                stats["appended"] += 1
            records[key] = rec  # recovered version wins

    merged = list(records.values())
    stats["missing_duration"] = sum(
        1 for r in merged if not isinstance(r.get("duration"), (int, float))
    )
    stats["total"] = len(merged)
    return merged, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, default=Path("cleaned_manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_manifests"))
    parser.add_argument("--version", required=True,
                        help="Snapshot label, e.g. v1.0 (next round: v2.0)")
    parser.add_argument("--languages", nargs="*", default=None,
                        help="Subset of language dirs (default: all present)")
    parser.add_argument("--keep-metadata", action="store_true",
                        help="Keep dialect/confidence/original_text fields")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the plan and counts; write nothing")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    languages = args.languages or sorted(
        p.name for p in args.input_dir.iterdir() if p.is_dir()
    )
    snapshot_dir = args.output_dir / args.version

    # ------------------------------------------------------------------ #
    # Discover: (out_lang, corpus) -> {role: [files]}                     #
    # ------------------------------------------------------------------ #
    plan: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    excluded: list[str] = []
    for lang in languages:
        lang_dir = args.input_dir / lang
        if not lang_dir.is_dir():
            print(f"WARN: no such language dir, skipping: {lang_dir}",
                  file=sys.stderr)
            continue
        for path in sorted(lang_dir.iterdir()):
            classified = classify_file(path)
            if classified is None:
                continue
            corpus, role = classified
            if role == "exclude":
                excluded.append(path.name)
                continue
            out_lang = lang
            as_lang = _AS_LANG_RE.match(corpus)
            if as_lang:  # cross-language recovery (e.g. eval_ml_inworld_as_hi)
                out_lang = as_lang.group("lang")
            plan[(out_lang, corpus)][role].append(path)

    # ------------------------------------------------------------------ #
    # Assemble                                                            #
    # ------------------------------------------------------------------ #
    manifest_meta: dict = {
        "version": args.version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_dir": str(args.input_dir.resolve()),
        "stripped_to_training_fields": not args.keep_metadata,
        "languages": {},
    }
    grand_total = 0

    for (out_lang, corpus), roles in sorted(plan.items()):
        # Prefer the merged file; fall back to shards (never both)
        base_files = roles.get("base_merged") or roles.get("base_shard") or []
        overlay_files = roles.get("overlay", [])
        if not base_files and not overlay_files:
            continue

        merged, stats = assemble_corpus(base_files, overlay_files)
        if not merged:
            continue
        grand_total += stats["total"]

        lang_meta = manifest_meta["languages"].setdefault(out_lang, {})
        lang_meta[corpus] = {
            **stats,
            "sources": [p.name for p in base_files + overlay_files],
        }

        flag = " [cross-lang]" if _AS_LANG_RE.match(corpus) else ""
        print(f"{out_lang}/{corpus}.jsonl{flag}: total={stats['total']} "
              f"(base={stats['base']} +recovered={stats['appended']} "
              f"~replaced={stats['replaced']}"
              + (f" missing_dur={stats['missing_duration']}"
                 if stats["missing_duration"] else "")
              + ")")

        if args.dry_run:
            continue
        out_path = snapshot_dir / out_lang / f"{corpus}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in merged:
                out_rec = rec if args.keep_metadata else strip_record(rec)
                fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)

    print(f"\nsnapshot total: {grand_total} records"
          f" | excluded piles: {len(excluded)} files"
          f" (rejected/still_rejected/suspect/selected)")

    if args.dry_run:
        print("DRY RUN — nothing written.")
        return 0

    meta_path = snapshot_dir / "MANIFEST.json"
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(
        json.dumps(manifest_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(meta_tmp, meta_path)
    print(f"snapshot ready: {snapshot_dir}  (provenance: {meta_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
