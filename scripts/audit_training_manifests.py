#!/usr/bin/env python3
"""Audit an assembled training-manifest snapshot for cleaning/processing quality.

Answers one question: **if we trained on this directory as-is, what would
silently go wrong?** Every check below corresponds to a real gate in the
training path, not to a general notion of tidiness.

    PYTHONPATH=scripts python3 scripts/audit_training_manifests.py training_manifests/v7.0

Checks, in descending order of how much they cost you:

1. **Rejected piles registered as training data.** ``assemble_training_manifests``
   excludes ``*_rejected`` / ``*_still_rejected``, but only recognises those
   suffixes on WHOLE files. Distributed reprocess writes part files
   (``<corpus>_still_rejected_p0019.jsonl``), which end in ``_p0019`` and so
   match no exclusion suffix — they fall through to "base_merged" and become
   their own corpus. This audit flags any registered file whose name contains a
   non-training role, because that is records-that-failed-cleaning going
   straight into training.

2. **Training gates.** Duration must land in [0.1, 35.0] s and the transcript
   must fit ``max_target_length`` (512) tokens, per every ``configs/*.yaml``.
   Neither failure raises at train time: ``ResilientAudioDataset.__getitem__``
   marks the index unusable and then serves a DIFFERENT record in its place,
   so a violation costs you the record AND silently oversamples a neighbour.

3. **Envelope leakage.** Source SFT text looks like
   ``language Arabic<asr_text>...``; ``processing.py`` adds that wrapper itself
   at train time. A manifest that still carries it would be double-wrapped and
   teach the model to emit the marker as transcript text.

4. **Script/language agreement.** A record filed under ``ar/`` whose dominant
   script is Devanagari is either mis-routed or mislabelled; both poison the
   language token the model conditions on.

5. **Integrity and duplication.** Unparseable lines, missing fields, duplicate
   audio paths, and train/eval audio overlap (which inflates eval scores).

Concurrent downloads are expected: a partial FINAL line is reported as a
truncated tail rather than a parse error, so this is safe to run against a
directory that is still filling up. Anything else that fails to parse is a
real defect and is counted as one.

Without ``--tokenizer`` the length check falls back to a character proxy and
says so — it is a hint, not the 512-token verdict. Pass a QASR model directory
for the exact count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the exact gates the filter/training path uses, so this audit and the
# pipeline can never drift apart. Imported after the sys.path insert above.
from prepare_q3asr_filter import (
    MAX_TARGET_TOKENS,
    TRAIN_MAX_DURATION,
    TRAIN_MIN_DURATION,
    count_target_tokens,
    heuristic_language,
    load_target_tokenizer,
    parse_text_envelope,
    script_char_counts,
)

LOGGER = logging.getLogger("audit_manifests")

TRAINING_FIELDS = {"audio_filepath", "text", "duration"}

# File-name fragments that mean "this is not training data". Matched anywhere in
# the stem so that distributed part files (`..._still_rejected_p0019`) are caught
# even though the role suffix is no longer final.
NON_TRAINING_MARKERS = ("_rejected", "_suspect", "_selected")

# Cleaning residue. Each pattern is something a correct pipeline never emits.
ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("envelope_marker", re.compile(r"<asr_text>|<\|.*?\|>")),
    ("llm_fence", re.compile(r"<<<|>>>|```")),
    ("html", re.compile(r"&(?:amp|lt|gt|quot|nbsp|#\d+);|</?[a-zA-Z][^>]{0,20}>")),
    ("replacement_char", re.compile("\ufffd")),
    ("bidi_control", re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")),
    ("zero_width", re.compile("[\u200b\u200c\u200d\ufeff]")),
    # Repeated `.` is ellipsis (legitimate in ar/ml transcripts) and would swamp
    # the signal, so only NON-period runs count as suspicious punctuation spam.
    ("repeated_punct", re.compile(r"([!?؟،,])\1{2,}")),
    ("bracketed_tag", re.compile(r"\[[a-zA-Z_]{3,}\]|\((?:noise|music|inaudible)\)")),
)

# Characters that must never survive into a transcript field.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

DURATION_BINS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 35.0, math.inf)


def _hash(text: str) -> int:
    """8-byte digest as an int: bounds memory when tracking millions of paths."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


class FileReport:
    """Everything observed about one manifest file."""

    def __init__(self, path: Path, lang: str) -> None:
        self.path = path
        self.lang = lang
        self.lines = 0
        self.records = 0
        self.tail_truncated = False
        self.parse_errors: list[int] = []
        self.not_an_object = 0
        self.missing_fields: Counter[str] = Counter()
        self.extra_fields: Counter[str] = Counter()
        self.empty_text = 0
        self.empty_audio = 0
        self.bad_duration_type = 0
        self.duration_missing = 0
        self.too_short = 0
        self.too_long_duration = 0
        self.duration_hist = [0] * (len(DURATION_BINS) - 1)
        self.over_token_budget = 0
        self.max_tokens_seen = 0
        self.artifacts: Counter[str] = Counter()
        self.artifact_examples: dict[str, str] = {}
        self.control_chars = 0
        self.untrimmed = 0
        self.internal_newline = 0
        self.lang_mismatch: Counter[str] = Counter()
        self.lang_mismatch_examples: dict[str, str] = {}
        self.undetectable_lang = 0
        self.dupe_audio = 0
        self.dupe_record = 0
        self.eval_overlap = 0
        self.chars_per_sec: list[float] = []
        self.suspicious_rate = 0

    @property
    def is_non_training(self) -> bool:
        return any(marker in self.path.stem for marker in NON_TRAINING_MARKERS)

    @property
    def content_defects(self) -> int:
        """Records whose TEXT is corrupted — the transcript itself is wrong.

        Deliberately excludes ``duration_missing``: a missing duration is a
        structural gap the pipeline can repair by probing the audio
        (backfill_durations.py), so folding it in here would drown the actual
        text-quality signal under a field that is trivially recoverable.
        """
        return (
            self.empty_text
            + self.empty_audio
            + self.bad_duration_type
            + self.too_short
            + self.too_long_duration
            + self.over_token_budget
            + self.control_chars
            + sum(self.artifacts.values())
        )


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        yield from enumerate(fh, 1)


def dominant_script(text: str) -> str:
    counts = script_char_counts(text)
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def audit_file(
    path: Path,
    lang: str,
    tokenizer: Any | None,
    eval_hashes: set[int] | None,
    track_dupes: bool,
    max_examples: int,
) -> FileReport:
    report = FileReport(path, lang)
    seen_audio: set[int] = set()
    seen_record: set[int] = set()
    last_error_line = -1
    pending_texts: list[str] = []
    pending_flush = 4000

    def flush_tokens() -> None:
        """Token-count a batch. Batched because per-call overhead dominates."""
        if not pending_texts or tokenizer is None:
            pending_texts.clear()
            return
        for count in count_target_tokens(pending_texts, tokenizer):
            report.max_tokens_seen = max(report.max_tokens_seen, count)
            if count > MAX_TARGET_TOKENS:
                report.over_token_budget += 1
        pending_texts.clear()

    for line_no, raw in iter_lines(path):
        report.lines = line_no
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            # Might be the in-flight tail of a download; decided after the loop
            # once we know whether this was the final line in the file.
            report.parse_errors.append(line_no)
            last_error_line = line_no
            continue
        if not isinstance(rec, dict):
            report.not_an_object += 1
            continue
        report.records += 1

        keys = set(rec.keys())
        for missing in TRAINING_FIELDS - keys:
            report.missing_fields[missing] += 1
        for extra in keys - TRAINING_FIELDS:
            report.extra_fields[extra] += 1

        audio = rec.get("audio_filepath") or ""
        text = rec.get("text")
        text = text if isinstance(text, str) else ""

        if not audio:
            report.empty_audio += 1
        if not text.strip():
            report.empty_text += 1

        # Duration against the real training band.
        duration = rec.get("duration")
        if duration is None:
            report.duration_missing += 1
        elif not isinstance(duration, (int, float)) or isinstance(duration, bool):
            report.bad_duration_type += 1
        else:
            value = float(duration)
            for idx in range(len(DURATION_BINS) - 1):
                if DURATION_BINS[idx] <= value < DURATION_BINS[idx + 1]:
                    report.duration_hist[idx] += 1
                    break
            if value < TRAIN_MIN_DURATION:
                report.too_short += 1
            elif value > TRAIN_MAX_DURATION:
                report.too_long_duration += 1
            if text.strip() and value > 0:
                rate = len(text.strip()) / value
                if len(report.chars_per_sec) < 200_000:
                    report.chars_per_sec.append(rate)
                # Wildly off rates mean the transcript and the audio disagree.
                if rate < 0.5 or rate > 60.0:
                    report.suspicious_rate += 1

        if text:
            if _CONTROL_RE.search(text):
                report.control_chars += 1
            if text != text.strip():
                report.untrimmed += 1
            if "\n" in text or "\t" in text:
                report.internal_newline += 1
            for name, pattern in ARTIFACT_PATTERNS:
                if pattern.search(text):
                    report.artifacts[name] += 1
                    if name not in report.artifact_examples and len(report.artifact_examples) < max_examples:
                        report.artifact_examples[name] = text[:160]
            # Envelope leakage is worth naming separately from the regex hit.
            if parse_text_envelope(text) is not None:
                report.artifacts["full_envelope"] += 1
                report.artifact_examples.setdefault("full_envelope", text[:160])

            pending_texts.append(text)
            if len(pending_texts) >= pending_flush:
                flush_tokens()

            detected = heuristic_language(text)
            if detected is None:
                report.undetectable_lang += 1
            elif detected != lang:
                report.lang_mismatch[detected] += 1
                if detected not in report.lang_mismatch_examples:
                    report.lang_mismatch_examples[detected] = f"{dominant_script(text)}: {text[:120]}"

        if track_dupes and audio:
            key = _hash(audio)
            if key in seen_audio:
                report.dupe_audio += 1
            else:
                seen_audio.add(key)
            rkey = _hash(f"{audio}\x00{text}")
            if rkey in seen_record:
                report.dupe_record += 1
            else:
                seen_record.add(rkey)
        if eval_hashes and audio and _hash(audio) in eval_hashes:
            report.eval_overlap += 1

    flush_tokens()

    # A single unparseable FINAL line is the download still in flight, not a bug.
    if report.parse_errors and last_error_line == report.lines:
        report.parse_errors.remove(last_error_line)
        report.tail_truncated = True
    return report


def collect_eval_hashes(files: list[tuple[Path, str]], lang: str) -> set[int]:
    """Audio hashes of every eval file for one language."""
    hashes: set[int] = set()
    for path, file_lang in files:
        if file_lang != lang or not path.stem.startswith("eval"):
            continue
        for _line_no, raw in iter_lines(path):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            audio = rec.get("audio_filepath")
            if audio:
                hashes.add(_hash(audio))
    return hashes


def cross_check_manifest(root: Path, reports: list[FileReport]) -> dict[str, Any]:
    """Compare MANIFEST.json's promises against the files on disk."""
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        return {"present": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered: dict[str, int] = {}
    leaked: list[tuple[str, str, int]] = []
    for lang, node in manifest.get("languages", {}).items():
        corpora = node.get("corpora", node)
        for entry in corpora.values():
            if not isinstance(entry, dict) or "total" not in entry:
                continue
            for filename in entry.get("files", []):
                registered[f"{lang}/{filename}"] = entry["total"]
                if any(marker in filename for marker in NON_TRAINING_MARKERS):
                    leaked.append((lang, filename, entry["total"]))
    observed = {f"{r.lang}/{r.path.name}": r for r in reports}
    mismatched = []
    for key, expected in registered.items():
        report = observed.get(key)
        if report is None:
            continue  # not downloaded yet
        actual = report.records + (1 if report.tail_truncated else 0)
        if not report.tail_truncated and actual != expected:
            mismatched.append((key, expected, report.records))
    return {
        "present": True,
        "version": manifest.get("version"),
        "created": manifest.get("created"),
        "input_dir": manifest.get("input_dir"),
        "registered_files": len(registered),
        "registered_records": sum(
            entry["total"]
            for node in manifest.get("languages", {}).values()
            for entry in (node.get("corpora", node)).values()
            if isinstance(entry, dict) and "total" in entry
        ),
        "downloaded_files": len([k for k in registered if k in observed]),
        "leaked": leaked,
        "mismatched": mismatched,
        "unregistered": sorted(set(observed) - set(registered)),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.3f}%" if whole else "n/a"


def render(reports: list[FileReport], manifest_info: dict[str, Any], tokenizer: Any | None) -> None:
    total_records = sum(r.records for r in reports)
    training = [r for r in reports if not r.is_non_training]
    non_training = [r for r in reports if r.is_non_training]

    print("=" * 100)
    print("MANIFEST CROSS-CHECK")
    print("=" * 100)
    if not manifest_info.get("present"):
        print("  no MANIFEST.json found")
    else:
        print(f"  version           : {manifest_info['version']}  (created {manifest_info['created']})")
        print(f"  source            : {manifest_info['input_dir']}")
        print(f"  registered        : {manifest_info['registered_records']:,} records "
              f"across {manifest_info['registered_files']} files")
        print(f"  downloaded so far : {manifest_info['downloaded_files']} files")
        if manifest_info["leaked"]:
            leaked_total = sum(t for _l, _f, t in manifest_info["leaked"])
            print()
            print(f"  !! {len(manifest_info['leaked'])} NON-TRAINING files are registered as corpora "
                  f"({leaked_total:,} records)")
            by_lang: Counter[str] = Counter()
            for lang, _f, total in manifest_info["leaked"]:
                by_lang[lang] += total
            for lang, total in by_lang.most_common():
                print(f"       {lang}: {total:,} records")
        if manifest_info["mismatched"]:
            print(f"  !! {len(manifest_info['mismatched'])} complete files disagree with their registered count:")
            for key, expected, actual in manifest_info["mismatched"][:10]:
                print(f"       {key}: manifest {expected:,} vs file {actual:,}")
        if manifest_info["unregistered"]:
            print(f"  !! present on disk but not in MANIFEST: {manifest_info['unregistered']}")

    print()
    print("=" * 100)
    print("PER-FILE")
    print("=" * 100)
    header = f"{'file':<50} {'records':>10} {'bad text':>9} {'rate':>9} {'no dur':>10}  state"
    print(header)
    print("-" * len(header))
    for report in reports:
        state = []
        if report.tail_truncated:
            state.append("TAIL TRUNCATED (downloading)")
        if report.is_non_training:
            state.append("NON-TRAINING PILE")
        if report.parse_errors:
            state.append(f"{len(report.parse_errors)} unparseable")
        print(f"{report.lang + '/' + report.path.name:<50} {report.records:>10,} "
              f"{report.content_defects:>9,} {_pct(report.content_defects, report.records):>9} "
              f"{report.duration_missing:>10,}  {', '.join(state) or 'ok'}")

    print()
    print("=" * 100)
    print("TRAINING GATES  (violations are silently SUBSTITUTED at train time, not dropped)")
    print("=" * 100)
    gate_rows = [
        ("duration missing", sum(r.duration_missing for r in reports)),
        ("duration wrong type", sum(r.bad_duration_type for r in reports)),
        (f"duration < {TRAIN_MIN_DURATION}s", sum(r.too_short for r in reports)),
        (f"duration > {TRAIN_MAX_DURATION}s", sum(r.too_long_duration for r in reports)),
        (
            f"transcript > {MAX_TARGET_TOKENS} tokens"
            + ("" if tokenizer else "  [NOT CHECKED: pass --tokenizer]"),
            sum(r.over_token_budget for r in reports),
        ),
        ("empty text", sum(r.empty_text for r in reports)),
        ("missing audio_filepath", sum(r.empty_audio for r in reports)),
    ]
    for label, count in gate_rows:
        print(f"  {label:<58} {count:>10,}  {_pct(count, total_records):>9}")
    if tokenizer:
        print(f"  longest transcript observed: {max((r.max_tokens_seen for r in reports), default=0)} tokens")

    print()
    print("=" * 100)
    print("CLEANING ARTIFACTS")
    print("=" * 100)
    combined: Counter[str] = Counter()
    for report in reports:
        combined.update(report.artifacts)
    combined["control_chars"] = sum(r.control_chars for r in reports)
    combined["untrimmed_whitespace"] = sum(r.untrimmed for r in reports)
    combined["internal_newline_or_tab"] = sum(r.internal_newline for r in reports)
    if not any(combined.values()):
        print("  none detected")
    for name, count in sorted(combined.items(), key=lambda kv: -kv[1]):
        if not count:
            continue
        print(f"  {name:<58} {count:>10,}  {_pct(count, total_records):>9}")
        for report in reports:
            example = report.artifact_examples.get(name)
            if example:
                print(f"      e.g. {report.lang}/{report.path.name}: {example!r}")
                break

    print()
    print("=" * 100)
    print("SCRIPT / LANGUAGE AGREEMENT")
    print("=" * 100)
    for report in reports:
        total_mismatch = sum(report.lang_mismatch.values())
        if not total_mismatch and not report.undetectable_lang:
            continue
        print(f"  {report.lang}/{report.path.name}  ({report.records:,} records)")
        if total_mismatch:
            detail = ", ".join(f"{k}={v:,}" for k, v in report.lang_mismatch.most_common(6))
            print(f"      dominant script says another language: {total_mismatch:,} "
                  f"({_pct(total_mismatch, report.records)})  [{detail}]")
            for detected, example in list(report.lang_mismatch_examples.items())[:3]:
                print(f"        -> {detected}: {example!r}")
        if report.undetectable_lang:
            print(f"      no script signal at all: {report.undetectable_lang:,} "
                  f"({_pct(report.undetectable_lang, report.records)})")

    print()
    print("=" * 100)
    print("DUPLICATION AND EVAL LEAKAGE")
    print("=" * 100)
    for report in reports:
        if report.dupe_audio or report.dupe_record or report.eval_overlap:
            print(f"  {report.lang}/{report.path.name}")
            if report.dupe_record:
                print(f"      identical (audio, text) repeated : {report.dupe_record:,}")
            if report.dupe_audio:
                print(f"      same audio_filepath reused      : {report.dupe_audio:,} "
                      f"({_pct(report.dupe_audio, report.records)})")
            if report.eval_overlap:
                print(f"      !! audio also present in eval   : {report.eval_overlap:,}")
    if not any(r.dupe_audio or r.dupe_record or r.eval_overlap for r in reports):
        print("  no duplicate audio and no train/eval overlap detected")

    print()
    print("=" * 100)
    print("DURATION DISTRIBUTION  (share of records per band)")
    print("=" * 100)
    labels = []
    for idx in range(len(DURATION_BINS) - 1):
        hi = DURATION_BINS[idx + 1]
        labels.append(f"{DURATION_BINS[idx]:g}-{'inf' if hi == math.inf else f'{hi:g}'}s")
    print(f"  {'band':<12}" + "".join(f"{lang:>12}" for lang in sorted({r.lang for r in reports})))
    per_lang: dict[str, list[int]] = defaultdict(lambda: [0] * (len(DURATION_BINS) - 1))
    for report in reports:
        for idx, count in enumerate(report.duration_hist):
            per_lang[report.lang][idx] += count
    for idx, label in enumerate(labels):
        row = f"  {label:<12}"
        for lang in sorted(per_lang):
            total = sum(per_lang[lang])
            row += f"{_pct(per_lang[lang][idx], total):>12}"
        print(row)

    print()
    print("=" * 100)
    print("TRANSCRIPT RATE  (chars/sec; a proxy for transcript-audio misalignment)")
    print("=" * 100)
    for report in reports:
        if not report.chars_per_sec:
            continue
        p = report.chars_per_sec
        print(f"  {report.lang + '/' + report.path.name:<50} "
              f"p1={percentile(p, 1):5.1f}  median={percentile(p, 50):5.1f}  "
              f"p99={percentile(p, 99):5.1f}  extreme={report.suspicious_rate:,}")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    train_records = sum(r.records for r in training)
    train_content = sum(r.content_defects for r in training)
    train_nodur = sum(r.duration_missing for r in training)
    train_fences = sum(r.artifacts.get("llm_fence", 0) + r.artifacts.get("full_envelope", 0)
                       for r in training)
    print(f"  training files : {len(training)}  ({train_records:,} records)")
    print(f"    corrupted text (artifacts/empty/control) : {train_content:>12,}  "
          f"{_pct(train_content, train_records)}")
    print(f"    of which <<<...>>> / envelope leakage    : {train_fences:>12,}  "
          f"{_pct(train_fences, train_records)}")
    print(f"    missing duration (recoverable by probe)  : {train_nodur:>12,}  "
          f"{_pct(train_nodur, train_records)}")
    eval_fences = sum(
        r.artifacts.get("llm_fence", 0) + r.artifacts.get("full_envelope", 0)
        for r in reports
        if r.path.stem.startswith("eval")
    )
    if eval_fences:
        print(f"  !! EVAL sets carry {eval_fences:,} fenced/enveloped transcripts "
              f"-- these corrupt the metric itself, not just training")
    if non_training:
        nt_records = sum(r.records for r in non_training)
        print(f"  non-training piles present on disk: {len(non_training)} "
              f"({nt_records:,} records) -- must not be trained on")
        for report in non_training:
            print(f"      {report.lang}/{report.path.name}")
    incomplete = [r for r in reports if r.tail_truncated]
    if incomplete:
        print(f"  still downloading: {len(incomplete)} file(s) -- re-run when complete")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="snapshot directory, e.g. training_manifests/v7.0")
    parser.add_argument("--tokenizer", help="QASR model dir for the exact 512-token check")
    parser.add_argument("--no-dupes", action="store_true", help="skip duplicate tracking (saves memory)")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--only", help="audit only files whose path contains this substring")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.root)
    if not root.is_dir():
        LOGGER.error("not a directory: %s", root)
        return 2

    files = sorted(
        (path, path.parent.name)
        for path in root.rglob("*.jsonl")
        if not args.only or args.only in str(path)
    )
    if not files:
        LOGGER.error("no .jsonl files under %s", root)
        return 2

    tokenizer = None
    if args.tokenizer:
        try:
            tokenizer = load_target_tokenizer(args.tokenizer)
        except Exception as exc:
            LOGGER.error("could not load tokenizer: %s", exc)
            return 2
    else:
        LOGGER.warning(
            "no --tokenizer: transcript length vs %d tokens is NOT checked", MAX_TARGET_TOKENS
        )

    reports: list[FileReport] = []
    for path, lang in files:
        eval_hashes = None
        if path.stem.startswith("train"):
            eval_hashes = collect_eval_hashes(files, lang) or None
        LOGGER.info("auditing %s ...", path.relative_to(root))
        reports.append(
            audit_file(path, lang, tokenizer, eval_hashes, not args.no_dupes, args.max_examples)
        )

    print()
    render(reports, cross_check_manifest(root, reports), tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
