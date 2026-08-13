"""Deep quality checks on ACCEPTED cleaned manifests (read-only).

Beyond audit.py classification:
  1. LLM artifacts in text (markdown fences, meta-phrases, think tags, JSON)
  2. Punctuation conventions (ar: Latin ?/, leakage; danda usage in hi)
  3. Arabic diacritic density vs the 5-15% prompt target
  4. Duplicate audio_filepath within/across accepted files
  5. Truncation signals: cleaned much shorter than original
  6. Language-specific fidelity violations:
     zh: code-switched Latin words translated away
     hi: agreement "fixes" (verb-form swaps)
     en: function words inserted
  7. Control/replacement chars, mixed digit systems
  8. Per-corpus rejection rates

Usage: PYTHONPATH=src python3 scripts/deep_quality_check.py cleaned_manifests
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from data_processing.audit import classify_record

FENCE = re.compile(r"```|~~~")
META = re.compile(
    r"(?i)\b(here is|here's|cleaned text|corrected text|transcript:|"
    r"output:|json|As an AI)\b"
)
THINK = re.compile(r"</?think>|<\|.*?\|>")
AR_LATIN_PUNCT = re.compile(r"[?;]")  # ar should use ؟ ؛
AR_LETTER = re.compile(r"[\u0621-\u064a]")
AR_DIAC = re.compile(r"[\u064b-\u0652\u0670]")
EASTERN_DIGIT = re.compile(r"[\u0660-\u0669\u06f0-\u06f9]")
DEVANAGARI_DIGIT = re.compile(r"[\u0966-\u096f]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'&.-]*")

# hi verb-form agreement swaps observed at scale (spoken-word changes)
HI_AGREEMENT = [
    ("है", "हैं"), ("हैं", "है"), ("थी", "थीं"), ("था", "थे"),
    ("करते", "करतें"), ("सकते", "सकतें"), ("ये", "यह"), ("ना", "न"),
]
EN_FUNC = {"a", "an", "the", "to", "is", "are", "was", "of", "in", "and"}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "cleaned_manifests")
    corpus_stats: dict[str, dict] = {}

    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        art = Counter()          # artifact counters
        viol = Counter()         # language-specific violations
        diac_hist = Counter()    # ar diacritic density buckets
        dup_paths: Counter = Counter()
        shrink = Counter()       # length-ratio buckets
        samples: dict[str, list] = defaultdict(list)
        n = 0

        for path in sorted(lang_dir.glob("*.jsonl")):
            if "rejected" in path.stem:
                continue
            base = re.sub(r"_shard_\d+$", "", path.stem).replace("_cleaned", "")
            cs = corpus_stats.setdefault(
                f"{lang}/{base}", {"acc": 0, "rej": 0}
            )
            for line in path.open("r", encoding="utf-8"):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                cs["acc"] += 1
                t = str(e.get("text") or "")
                o = str(e.get("original_text") or "")
                dup_paths[e.get("audio_filepath", "")] += 1

                # 1. artifacts
                if FENCE.search(t):
                    art["fence"] += 1
                    _keep(samples, "fence", t)
                if THINK.search(t):
                    art["think_tag"] += 1
                    _keep(samples, "think_tag", t)
                if META.search(t) and not META.search(o):
                    art["meta_phrase"] += 1
                    _keep(samples, "meta_phrase", t)
                if CONTROL.search(t):
                    art["control_char"] += 1
                if t.strip().startswith(("{", "[")) and not o.strip().startswith(("{", "[")):
                    art["json_like"] += 1
                    _keep(samples, "json_like", t)

                # 2. conventions
                if lang == "ar":
                    if AR_LATIN_PUNCT.search(t):
                        art["latin_punct"] += 1
                        _keep(samples, "latin_punct", t)
                    if EASTERN_DIGIT.search(t):
                        art["eastern_digits"] += 1
                    letters = len(AR_LETTER.findall(t))
                    if letters >= 20:
                        d = len(AR_DIAC.findall(t)) / letters
                        diac_hist[
                            "0%" if d == 0 else
                            "<5%" if d < 0.05 else
                            "5-15%" if d <= 0.15 else
                            "15-30%" if d <= 0.30 else ">30%"
                        ] += 1
                if lang == "hi" and DEVANAGARI_DIGIT.search(t):
                    art["devanagari_digits"] += 1

                # 5. truncation signal
                if len(o) > 40:
                    r = len(t) / len(o)
                    if r < 0.5:
                        shrink["<0.5"] += 1
                        _keep(samples, "shrink", f"{o[:70]} ==> {t[:70]}")
                    elif r < 0.8:
                        shrink["0.5-0.8"] += 1

                # 6. language-specific fidelity violations
                if lang == "zh":
                    lost = [
                        w for w in set(LATIN_WORD.findall(o))
                        if len(w) > 2 and w.lower() not in t.lower()
                    ]
                    if lost:
                        viol["latin_word_lost"] += 1
                        _keep(samples, "zh_latin_lost",
                              f"{','.join(lost[:3])} | {o[:60]} ==> {t[:60]}")
                elif lang == "hi":
                    cat, diffs = classify_record(e)
                    if cat == "word_diff":
                        for d in diffs:
                            parts = d.split(" -> ")
                            if len(parts) == 2 and (parts[0], parts[1]) in [
                                (a, b) for a, b in HI_AGREEMENT
                            ]:
                                viol["agreement_fix"] += 1
                                break
                elif lang == "en":
                    cat, diffs = classify_record(e)
                    if cat == "word_diff":
                        for d in diffs:
                            parts = d.split(" -> ")
                            if (
                                len(parts) == 2
                                and parts[0] == "∅"
                                and parts[1].strip() in EN_FUNC
                            ):
                                viol["func_word_inserted"] += 1
                                _keep(samples, "en_insert",
                                      f"{d} | {o[:60]} ==> {t[:60]}")
                                break

        # rejected counts per corpus
        for path in lang_dir.glob("*rejected*.jsonl"):
            base = re.sub(r"_shard_\d+$", "", path.stem).replace("_rejected", "")
            cs = corpus_stats.setdefault(f"{lang}/{base}", {"acc": 0, "rej": 0})
            with path.open("r", encoding="utf-8") as fh:
                cs["rej"] += sum(1 for _ in fh)

        dups = {p: c for p, c in dup_paths.items() if c > 1}
        print(f"\n{'='*72}\n{lang.upper()} accepted={n:,}")
        print(f"  artifacts:  {dict(art) or 'none'}")
        print(f"  violations: {dict(viol) or 'none'}")
        if lang == "ar":
            print(f"  diacritic density (texts>=20 letters): {dict(diac_hist.most_common())}")
        print(f"  shrink ratios: {dict(shrink) or 'none'}")
        print(f"  duplicate audio paths: {len(dups):,} paths, "
              f"{sum(dups.values()) - len(dups):,} extra records")
        for key, items in samples.items():
            print(f"  -- {key} samples:")
            for s in items[:3]:
                print(f"     {s[:150]}")

    print(f"\n{'='*72}\nPER-CORPUS REJECTION RATES")
    for key in sorted(corpus_stats):
        s = corpus_stats[key]
        tot = s["acc"] + s["rej"]
        if tot == 0:
            continue
        rate = 100.0 * s["rej"] / tot
        bar = "#" * int(rate / 4)
        print(f"  {key:<48} {s['acc']:>9,} acc {s['rej']:>8,} rej "
              f"{rate:5.1f}% {bar}")
    return 0


def _keep(samples: dict, key: str, value: str, limit: int = 5) -> None:
    if len(samples[key]) < limit:
        samples[key].append(value)


if __name__ == "__main__":
    sys.exit(main())
