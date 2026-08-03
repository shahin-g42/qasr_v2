"""Audit ACCEPTED records for verbatim-fidelity violations.

Runs produced by older prompt versions may contain accepted records whose
``text`` silently drops fillers, substitutes words, or "fixes" grammar
relative to ``original_text`` — breaking the audio/transcript match. Since
every record preserves ``original_text``, word-level drift is detectable
deterministically (no LLM needed).

Classification per record (comparison ignores punctuation, casing, Arabic
diacritics/hamza-seat orthography, and Devanagari nasal/nukta orthography):
    identical  — text == original_text verbatim
    punct_only — only punctuation/casing/diacritics/orthography differ (safe)
    itn_only   — all word diffs involve digits (expected ITN; safe unless
                 --strict)
    word_diff  — spoken words dropped/added/substituted (SUSPECT)

Suspect records are written next to each input file (same schema, so they
can be re-run from original_text via data_processing.reprocess); safe
records are written alongside for final-manifest assembly:
    <name>_safe.jsonl     — identical / punct_only / itn_only records
    <name>_suspect.jsonl  — word_diff records (re-run these)

Final manifest per source: cat <name>_safe.jsonl <name>_suspect_recovered.jsonl

Usage:
    PYTHONPATH=src python -m data_processing.audit \
        --dir cleaned_manifests/en --pattern "*_cleaned.jsonl"
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("data_processing.audit")

# Arabic diacritics (harakat, tanwin, shadda, sukun, superscript alef) + tatweel
_AR_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u0640]")
# Orthographic normalizations so legit cleaner edits don't count as word diffs
_AR_NORMALIZE = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}
)
# Arabic-Indic (U+0660) and Extended Arabic-Indic (U+06F0, Persian/Urdu) digits
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Devanagari orthographic variants of the same spoken word:
#   nukta (क़/क, ज़/ज) stripped; chandrabindu (ँ) folded to anusvara (ं);
#   conjunct nasal + halant (लम्बा) folded to anusvara form (लंबा)
_HI_NUKTA = re.compile("\u093c")
_HI_CHANDRABINDU = str.maketrans({"\u0901": "\u0902"})
_HI_CONJUNCT_NASAL = re.compile("[\u0919\u091e\u0923\u0928\u092e]\u094d(?=[\u0915-\u0939])")
_HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# \w excludes combining marks — include the Devanagari block so matras
# (phonemic vowel signs) stay inside tokens instead of splitting words.
# Danda । (U+0964) / double danda ॥ (U+0965) are punctuation — excluded, or
# "है।" vs "है" would count as a word diff.
_TOKEN_RE = re.compile(r"[\w\u0900-\u0963\u0966-\u097f']+")


def _tokenize(text: str) -> list[str]:
    """Comparison tokens: lowercased words, punctuation/diacritics stripped."""
    text = _AR_DIACRITICS.sub("", text)
    text = text.translate(_AR_NORMALIZE).translate(_EASTERN_DIGITS)
    text = _HI_NUKTA.sub("", text)
    text = text.translate(_HI_CHANDRABINDU).translate(_HI_DIGITS)
    text = _HI_CONJUNCT_NASAL.sub("\u0902", text)
    return _TOKEN_RE.findall(text.lower())


def _has_digit(tokens: list[str]) -> bool:
    return any(any(ch.isdigit() for ch in tok) for tok in tokens)


def classify_record(entry: dict) -> tuple[str, list[str]]:
    """Classify divergence between text and original_text.

    Returns (category, diff_descriptions).
    """
    text = str(entry.get("text") or "")
    original = str(entry.get("original_text") or "")
    if not original or text == original:
        return "identical", []

    cleaned_tokens = _tokenize(text)
    original_tokens = _tokenize(original)
    if cleaned_tokens == original_tokens:
        return "punct_only", []

    # Spacing-only differences (zh "更多ideas" -> "更多 ideas", ar clitic joins
    # "و انا" -> "وانا") shift token boundaries without changing any spoken
    # character — formatting, not a word change.
    if "".join(cleaned_tokens) == "".join(original_tokens):
        return "punct_only", []

    diffs: list[str] = []
    all_digit_related = True
    matcher = difflib.SequenceMatcher(a=original_tokens, b=cleaned_tokens)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        removed = original_tokens[a0:a1]
        added = cleaned_tokens[b0:b1]
        if not (_has_digit(removed) or _has_digit(added)):
            all_digit_related = False
        diffs.append(f"{' '.join(removed) or '∅'} -> {' '.join(added) or '∅'}")

    return ("itn_only" if all_digit_related else "word_diff"), diffs


@dataclass
class AuditStats:
    """Aggregated audit counters."""

    total: int = 0
    identical: int = 0
    punct_only: int = 0
    itn_only: int = 0
    word_diff: int = 0
    parse_errors: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def safe(self) -> int:
        return self.identical + self.punct_only


def audit_file(
    path: Path,
    stats: AuditStats,
    strict: bool = False,
    sample_limit: int = 0,
) -> tuple[Path, int]:
    """Audit one JSONL file, writing <stem>_safe.jsonl and <stem>_suspect.jsonl.

    Returns (suspect_path, suspect_count). Deduplicates within the file by
    (audio_filepath, original_text) — append-only outputs may repeat records.
    """
    safe_path = path.with_name(f"{path.stem}_safe.jsonl")
    suspect_path = path.with_name(f"{path.stem}_suspect.jsonl")
    seen: set[tuple[str, str]] = set()
    suspect_count = 0

    with (
        path.open("r", encoding="utf-8") as handle,
        safe_path.open("w", encoding="utf-8") as safe_handle,
        suspect_path.open("w", encoding="utf-8") as suspect_handle,
    ):
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                stats.parse_errors += 1
                LOGGER.warning("Bad JSON at %s:%d", path.name, line_number)
                continue

            key = (
                str(entry.get("audio_filepath", "")),
                str(entry.get("original_text") or entry.get("text") or ""),
            )
            if key in seen:
                continue
            seen.add(key)

            stats.total += 1
            category, diffs = classify_record(entry)
            setattr(stats, category, getattr(stats, category) + 1)

            is_suspect = category == "word_diff" or (
                strict and category == "itn_only"
            )
            target = suspect_handle if is_suspect else safe_handle
            target.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if is_suspect:
                suspect_count += 1
                if len(stats.samples) < sample_limit:
                    stats.samples.append(
                        f"{path.name}:{line_number}  " + " | ".join(diffs[:4])
                    )

    return suspect_path, suspect_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit accepted records for verbatim-fidelity violations"
    )
    parser.add_argument(
        "--dir", required=True, help="Directory containing accepted JSONL files"
    )
    parser.add_argument(
        "--pattern",
        default="*_cleaned.jsonl",
        help="Glob for files to audit (e.g. '*_shard_*.jsonl' for live runs)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also treat digit-only (ITN) diffs as suspect",
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="Print N example diffs"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    audit_dir = Path(args.dir)
    files = sorted(
        p
        for p in audit_dir.glob(args.pattern)
        if not p.stem.endswith(("_safe", "_suspect"))
    )
    if not files:
        LOGGER.error("No files match %s in %s", args.pattern, audit_dir)
        return 1

    stats = AuditStats()
    suspect_files: list[tuple[Path, int]] = []

    for path in files:
        LOGGER.info("Auditing %s", path.name)
        suspect_path, count = audit_file(
            path,
            stats,
            strict=args.strict,
            sample_limit=args.sample,
        )
        suspect_files.append((suspect_path, count))

    suspects = sum(count for _, count in suspect_files)
    print()
    print(f"Audited {len(files)} files, {stats.total} records")
    print(f"  identical:   {stats.identical}")
    print(f"  punct_only:  {stats.punct_only}  (safe formatting edits)")
    print(f"  itn_only:    {stats.itn_only}  (digit conversions{'' if not args.strict else ', SUSPECT via --strict'})")
    print(f"  word_diff:   {stats.word_diff}  (SUSPECT: spoken words changed)")
    if stats.parse_errors:
        print(f"  parse errors: {stats.parse_errors}")
    print(f"\nSuspects: {suspects}")
    for suspect_path, count in suspect_files:
        if count:
            print(f"  {count:8d}  {suspect_path}")
    if stats.samples:
        print("\nExample diffs (original -> cleaned):")
        for sample in stats.samples:
            print(f"  {sample}")
    if suspects:
        print(
            "\nNext: re-run suspects from original_text with the fixed prompts:\n"
            "  PYTHONPATH=src python -m data_processing.reprocess \\\n"
            f"      --config <config.yaml> --rejected-dir {audit_dir} \\\n"
            '      --files "<name>_suspect.jsonl,..." --language <lang>\n'
            "Then per manifest:\n"
            "  cat <name>_safe.jsonl <name>_suspect_recovered.jsonl > <name>_final.jsonl"
        )
    return 0


__all__ = ["AuditStats", "audit_file", "classify_record", "main"]


if __name__ == "__main__":
    sys.exit(main())
