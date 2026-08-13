"""Read-only deep analysis of cleaned_manifests/ quality.

Streams every accepted + rejected JSONL, aggregates per language:
  - processing_version distribution
  - fidelity classification (identical/punct_only/itn_only/word_diff)
  - confidence histogram, changes-tag distribution
  - rejected-file reason distribution + wrong-language contamination
  - schema checks (missing duration, empty text)
  - sample word_diff records for manual review

Usage: PYTHONPATH=src python3 scripts/analyze_cleaned.py cleaned_manifests
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from data_processing.audit import classify_record

random.seed(7)

SCRIPT_RANGES = {
    "ar": ("\u0600", "\u06ff"),
    "hi": ("\u0900", "\u097f"),
    "ml": ("\u0d00", "\u0d7f"),
    "zh": ("\u4e00", "\u9fff"),
}


def detect_script(text: str) -> str:
    """Rough dominant-script detector for contamination checks."""
    counts: Counter[str] = Counter()
    for ch in text:
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= ch <= hi:
                counts[lang] += 1
                break
        else:
            if ch.isalpha() and ch.isascii():
                counts["latin"] += 1
    if not counts:
        return "none"
    return counts.most_common(1)[0][0]


def conf_bucket(c: float) -> str:
    if c >= 0.9:
        return "0.9+"
    if c >= 0.7:
        return "0.7-0.9"
    if c >= 0.5:
        return "0.5-0.7"
    if c >= 0.3:
        return "0.3-0.5"
    return "<0.3"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "cleaned_manifests")
    report: dict[str, dict] = {}

    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        acc = {
            "records": 0,
            "versions": Counter(),
            "classes": Counter(),
            "conf": Counter(),
            "tags": Counter(),
            "no_duration": 0,
            "empty_text": 0,
            "parse_errors": 0,
            "files": 0,
            "wd_samples": [],
            "wd_diff_tokens": Counter(),
        }
        rej = {
            "records": 0,
            "reasons": Counter(),
            "scripts": Counter(),
            "files": 0,
            "samples": [],
        }

        for path in sorted(lang_dir.glob("*.jsonl")):
            is_rejected = "rejected" in path.stem
            bucket = rej if is_rejected else acc
            bucket["files"] += 1
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        acc["parse_errors"] += 1
                        continue
                    bucket["records"] += 1
                    text = str(entry.get("text") or "")

                    if is_rejected:
                        for tag in entry.get("changes") or ["<no-tag>"]:
                            rej["reasons"][str(tag)] += 1
                        rej["scripts"][detect_script(text)] += 1
                        if len(rej["samples"]) < 200:
                            rej["samples"].append(entry)
                        continue

                    acc["versions"][str(entry.get("processing_version"))] += 1
                    if "duration" not in entry:
                        acc["no_duration"] += 1
                    if not text.strip():
                        acc["empty_text"] += 1
                    acc["conf"][conf_bucket(float(entry.get("confidence", 0)))] += 1
                    for tag in set(entry.get("changes") or []):
                        acc["tags"][str(tag)] += 1

                    category, diffs = classify_record(entry)
                    acc["classes"][category] += 1
                    if category == "word_diff":
                        for d in diffs:
                            acc["wd_diff_tokens"][d] += 1
                        if len(acc["wd_samples"]) < 5000:
                            acc["wd_samples"].append(
                                {
                                    "file": path.name,
                                    "orig": entry.get("original_text"),
                                    "text": text,
                                    "diffs": diffs[:4],
                                    "conf": entry.get("confidence"),
                                }
                            )

        report[lang] = {"accepted": acc, "rejected": rej}

        # ---- print per-language summary as we go (streaming feedback)
        total = acc["records"]
        print(f"\n{'='*70}\n{lang.upper()}: {total:,} accepted / "
              f"{rej['records']:,} rejected  "
              f"({acc['files']} + {rej['files']} files)")
        if not total:
            continue
        print(f"  versions:  {dict(acc['versions'])}")
        c = acc["classes"]
        for k in ("identical", "punct_only", "itn_only", "word_diff"):
            pct = 100.0 * c[k] / total
            print(f"  {k:<11} {c[k]:>10,}  {pct:5.1f}%")
        safe = c["identical"] + c["punct_only"] + c["itn_only"]
        print(f"  -> verbatim-safe: {100.0*safe/total:.2f}%")
        print(f"  confidence: {dict(sorted(acc['conf'].items()))}")
        print(f"  top tags:   {acc['tags'].most_common(8)}")
        if acc["no_duration"]:
            print(f"  !! missing duration: {acc['no_duration']:,}")
        if acc["empty_text"]:
            print(f"  !! empty text: {acc['empty_text']:,}")
        if acc["parse_errors"]:
            print(f"  !! parse errors: {acc['parse_errors']:,}")
        if rej["records"]:
            print(f"  rejected reasons: {rej['reasons'].most_common(8)}")
            print(f"  rejected scripts: {dict(rej['scripts'].most_common())}")

    # ---- dump machine-readable details for follow-up inspection
    out = Path("logs/cleaned_analysis.json")
    out.parent.mkdir(exist_ok=True)
    slim = {}
    for lang, r in report.items():
        a = r["accepted"]
        slim[lang] = {
            "accepted": a["records"],
            "rejected": r["rejected"]["records"],
            "versions": dict(a["versions"]),
            "classes": dict(a["classes"]),
            "conf": dict(a["conf"]),
            "tags": dict(a["tags"]),
            "no_duration": a["no_duration"],
            "empty_text": a["empty_text"],
            "rejected_reasons": dict(r["rejected"]["reasons"]),
            "rejected_scripts": dict(r["rejected"]["scripts"]),
            "top_word_diffs": a["wd_diff_tokens"].most_common(40),
            "word_diff_samples": random.sample(
                a["wd_samples"], min(60, len(a["wd_samples"]))
            ),
            "rejected_samples": r["rejected"]["samples"][:60],
        }
    out.write_text(
        json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nDetails written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
