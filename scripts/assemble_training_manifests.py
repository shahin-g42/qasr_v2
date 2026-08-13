#!/usr/bin/env python3
"""Assemble cleaned + reprocessed manifests into a versioned training snapshot.

Collects, per language, the accepted output of the cleaning pipeline and ALL
reprocess recoveries, folds them back into their parent corpus, and writes a
self-contained, training-ready snapshot:

    <output-dir>/<version>/<lang>/<corpus>.jsonl
    <output-dir>/<version>/MANIFEST.json      (provenance + counts)

Recovery folding (the important part):
  Reprocess rounds produce <corpus>_recovered, <corpus>_still_recovered,
  <corpus>_still_still_recovered, ... and <corpus>_suspect_recovered.
  These are NOT separate corpora: every recovered record is folded back
  into its parent corpus with replace-or-append semantics. Round N wins
  over round N-1; suspect re-cleans win over everything. The snapshot
  therefore contains exactly ONE file per corpus — no *_still_* junk.

Distributed part files:
  Reprocess writes part files like <corpus>_recovered_p0012.jsonl or
  <corpus>_still_rejected_p0019.jsonl. The _PART_RE regex strips the
  trailing _pNNNN for ALL role types before the ROLE_SUFFIXES check, so
  *_rejected_pNNNN is excluded just like the whole-file *_rejected.

Text sanitisation:
  <<<...>>> fences (the cleaning-LLM's response delimiters) are stripped
  from every record during ingest. A record whose text becomes empty after
  stripping is counted as dropped_invalid.

Merge rules (per corpus):
  1. Base = <corpus>_cleaned.jsonl when present, else <corpus>_shard_*.jsonl
     (never both — corpora exist in BOTH forms with identical content).
  2. Overlays = all recovered rounds, applied in ascending round order so
     the latest cleaning verdict always wins; a record whose
     (audio_filepath, original_text) matches an existing record REPLACES
     it, otherwise it is APPENDED. Distributed-reprocess part files
     (<corpus>_<role>_pNNNN.jsonl) are treated exactly like the
     whole-file <corpus>_<role>.jsonl they slice.
  3. Excluded: *_rejected, *_still_rejected, *_suspect, *_selected samples,
     reports, backups, tmp files — including their part-file variants.
  4. Corpora with a `_as_<lang>` stem suffix (e.g. eval_ml_inworld_as_hi)
     are routed to the target language's output directory — including
     their recovery rounds.
  5. Records with empty text or missing audio_filepath are dropped and
     counted (defensive quality gate).

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

# Suspect re-cleans are the final authority on a record, so they sort last.
SUSPECT_PRIORITY = 10_000

_SHARD_RE = re.compile(r"^(?P<corpus>.+)_shard_\d{4}$")
_AS_LANG_RE = re.compile(r"^(?P<corpus>.+)_as_(?P<lang>[a-z]{2})$")
# Distributed-reprocess part files: <corpus>_<role>_p0012.jsonl
# Covers ALL roles: _recovered, _suspect_recovered, _rejected, _still_rejected,
# _selected. Without this, *_rejected_p0019 escapes the ROLE_SUFFIXES check and
# gets registered as its own training corpus.
_PART_RE = re.compile(
    r"^(?P<stem>.+?_(?:suspect_recovered|recovered|still_rejected|rejected|selected))_p\d{4}$"
)

# <<<...>>> fences are the cleaning-LLM's response delimiters. They should never
# survive into a manifest but were observed in v7.0 at ~1% of records. Strip them
# during assembly so training never sees them.
_FENCE_RE = re.compile(r"<<<|>>>")


def classify_file(path: Path) -> tuple[str, str] | None:
    """Return (corpus, role) for a manifest file, or None to ignore it."""
    if path.suffix != ".jsonl" or path.name.endswith((".tmp", ".orig")):
        return None
    stem = path.stem
    shard = _SHARD_RE.match(stem)
    if shard:
        return shard.group("corpus"), "base_shard"
    part = _PART_RE.match(stem)
    if part:  # slice of a recovered pile -> same overlay as the whole file
        stem = part.group("stem")
    for suffix, role in ROLE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], role
    # Bare <corpus>.jsonl (no recognized suffix): treat as merged base
    return stem, "base_merged"


def normalize_corpus(corpus: str) -> tuple[str, int, bool]:
    """Fold reprocess-chain stems back to their parent corpus.

    Returns (parent_corpus, recovery_round, is_suspect). Round N of the
    reprocess chain names its output <corpus>_still*<N>_recovered; all
    rounds belong to the SAME corpus.

        X                    -> (X, 0, False)   plain base/cleaned
        X_still              -> (X, 1, False)   round-2 recovered pile
        X_still_still_still  -> (X, 3, False)   round-4 recovered pile
        X_suspect            -> (X, 0, True)    suspect re-clean pile
    """
    suspect = corpus.endswith("_suspect")
    if suspect:
        corpus = corpus[: -len("_suspect")]
    rounds = 0
    while corpus.endswith("_still"):
        corpus = corpus[: -len("_still")]
        rounds += 1
    return corpus, rounds, suspect


def overlay_priority(corpus: str) -> int:
    """Application order for recovered overlays (later wins on conflict)."""
    _, rounds, suspect = normalize_corpus(corpus)
    return SUSPECT_PRIORITY if suspect else rounds


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
    text = out.get("text")
    if text and _FENCE_RE.search(text):
        out["text"] = _FENCE_RE.sub("", text).strip()
    return out


def assemble_corpus(
    base_files: list[Path], overlays: list[tuple[int, str, Path]]
) -> tuple[list[dict], dict]:
    """Merge base + folded recovery overlays with replace-or-append semantics.

    Overlays are (priority, source_corpus, path); they are applied in
    ascending priority order so the most recent cleaning verdict wins.
    """
    records: dict[tuple[str, str], dict] = {}
    origin: dict[tuple[str, str], str] = {}   # key -> "base" | overlay label
    stats = {
        "base": 0, "base_dupes": 0, "replaced": 0, "appended": 0,
        "overlay_dupes": 0, "dropped_invalid": 0, "missing_duration": 0,
        "recovery_rounds": defaultdict(int),
    }

    def ingest(rec: dict, src: str) -> None:
        text = str(rec.get("text", "")).strip()
        # Strip <<<>>> fences early — they poison both training and dedup keys.
        if _FENCE_RE.search(text):
            text = _FENCE_RE.sub("", text).strip()
            rec = {**rec, "text": text}
        if not text or not rec.get("audio_filepath"):
            stats["dropped_invalid"] += 1
            return
        key = record_key(rec)
        prev = origin.get(key)
        if prev is None:
            stats["appended" if src != "base" else "base"] += 1
        elif prev == src:
            stats["overlay_dupes" if src != "base" else "base_dupes"] += 1
            return                            # exact re-run: keep first copy
        elif prev == "base":
            stats["replaced" if src != "base" else "base_dupes"] += 1
        elif src == "base":
            stats["base_dupes"] += 1        # base never overrides recoveries
            return
        else:
            stats["overlay_dupes"] += 1
        records[key] = rec
        origin[key] = src

    for path in sorted(base_files):
        for rec in read_jsonl(path):
            ingest(rec, "base")

    # Sorted by priority (first element), which only controls overlay ordering.
    for _priority, source_corpus, path in sorted(overlays):
        _, rounds, suspect = normalize_corpus(source_corpus)
        label = "suspect_recovered" if suspect else f"round_{rounds + 1}"
        before = len(records)
        for rec in read_jsonl(path):
            ingest(rec, label)
        stats["recovery_rounds"][label] += len(records) - before

    merged = list(records.values())
    stats["missing_duration"] = sum(
        1 for r in merged if not isinstance(r.get("duration"), (int, float))
    )
    stats["total"] = len(merged)
    stats["recovery_rounds"] = dict(stats["recovery_rounds"])
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
    parser.add_argument("--shard-size", type=int, default=0,
                        help="Split outputs larger than N records into "
                             "<corpus>_shard_NNNN.jsonl (default: single file)")
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
    # Discover: (out_lang, corpus) -> {"base": [...], "overlay": [...]}   #
    # Recovery chains are folded onto their parent corpus here.           #
    # ------------------------------------------------------------------ #
    plan: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"base": [], "overlay": []}
    )
    excluded: list[str] = []
    folded_chains: dict[str, set[int]] = defaultdict(set)
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
            raw_corpus, role = classified
            if role == "exclude":
                excluded.append(path.name)
                continue
            corpus, rounds, suspect = normalize_corpus(raw_corpus)
            if role == "overlay" and (rounds or suspect):
                folded_chains[corpus].add(rounds + 1 if not suspect else 0)
            out_lang = lang
            as_lang = _AS_LANG_RE.match(corpus)
            if as_lang:  # cross-language recovery (e.g. eval_ml_inworld_as_hi)
                out_lang = as_lang.group("lang")
            if role in ("base_merged", "base_shard"):
                plan[(out_lang, corpus)]["base"].append((role, path))
            else:
                plan[(out_lang, corpus)]["overlay"].append(
                    (overlay_priority(raw_corpus), raw_corpus, path)
                )

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
        base_merged = [p for role, p in roles["base"] if role == "base_merged"]
        base_shards = [p for role, p in roles["base"] if role == "base_shard"]
        base_files = base_merged or base_shards
        if not base_files and not roles["overlay"]:
            continue

        merged, stats = assemble_corpus(base_files, roles["overlay"])
        if not merged:
            continue
        grand_total += stats["total"]

        lang_meta = manifest_meta["languages"].setdefault(out_lang, {})
        lang_meta[corpus] = {
            **stats,
            "sources": [p.name for p in base_files]
                       + [p.name for _, _, p in sorted(roles["overlay"])],
        }

        flag = " [cross-lang]" if _AS_LANG_RE.match(corpus) else ""
        folded = " [recovery folded]" if corpus in folded_chains else ""
        print(f"{out_lang}/{corpus}.jsonl{flag}{folded}: total={stats['total']} "
              f"(base={stats['base']} +recovered={stats['appended']} "
              f"~replaced={stats['replaced']}"
              + (f" ~overlay-dupes={stats['overlay_dupes']}"
                 if stats["overlay_dupes"] else "")
              + (f" !dropped={stats['dropped_invalid']}"
                 if stats["dropped_invalid"] else "")
              + (f" missing_dur={stats['missing_duration']}"
                 if stats["missing_duration"] else "")
              + ")")

        if args.dry_run:
            continue
        out_dir = snapshot_dir / out_lang
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.shard_size > 0 and len(merged) > args.shard_size:
            written = _write_sharded(out_dir, corpus, merged,
                                     args.shard_size, args.keep_metadata)
        else:
            written = [_write_single(out_dir / f"{corpus}.jsonl",
                                     merged, args.keep_metadata)]
        lang_meta[corpus]["files"] = [p.name for p in written]

    print(f"\nsnapshot total: {grand_total} records"
          f" | excluded piles: {len(excluded)} files"
          f" (rejected/still_rejected/suspect/selected)"
          f" | recovery chains folded: {len(folded_chains)} corpora")

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


def _write_single(out_path: Path, records: list[dict],
                  keep_metadata: bool) -> Path:
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            out_rec = rec if keep_metadata else strip_record(rec)
            fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return out_path


def _write_sharded(out_dir: Path, corpus: str, records: list[dict],
                   shard_size: int, keep_metadata: bool) -> list[Path]:
    written = []
    for idx in range(0, len(records), shard_size):
        part = idx // shard_size
        out_path = out_dir / f"{corpus}_shard_{part:04d}.jsonl"
        written.append(_write_single(
            out_path, records[idx:idx + shard_size], keep_metadata))
    return written


if __name__ == "__main__":
    raise SystemExit(main())
