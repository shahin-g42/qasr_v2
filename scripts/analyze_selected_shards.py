#!/usr/bin/env python3
"""Ad-hoc effectiveness report for selected (accepted) AR q3asr shard samples."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SHARDS = ["0033", "0037", "0073", "0077", "0087", "0088", "0089", "0090"]
DIR = Path(".")

DIAC = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670")
EASTERN = set("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
AR_LETTER = re.compile(r"[\u0621-\u064a\u0671-\u06d3]")
ARTIFACT = re.compile(r"```|</?think>|\{\"text\"|corrected_text|transcript|json", re.I)
CONTROL = re.compile(r"[\u0000-\u0008\u000b-\u001f\u200e\u200f\u202a-\u202e]")
LATIN_PUNCT = re.compile(r"[?;,]")
WESTERN_DIGIT = re.compile(r"[0-9]")
TATWEEL = "\u0640"
FALSE_START = re.compile(r"\b\w{1,3}-\s")

def strip_diac(s: str) -> str:
    return "".join(c for c in s if c not in DIAC)

PUNCT_ALL = set(".,!?;:\u060c\u061f\u061b\"'()[]{}«»…-–—%٪")
def strip_punct(s: str) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in s if c not in PUNCT_ALL)).strip()

def words(s: str) -> list[str]:
    return s.split()

n = 0
identical = 0
tag_counter = Counter()
dialect_counter = Counter()
conf_buckets = Counter()
diac_buckets = Counter()
cat_counter = Counter()
examples: dict[str, list[tuple[str, str]]] = {}
defects: dict[str, list[str]] = {}
shrink_flags = []
word_sub_examples = []
lens = []

def add_ex(cat, orig, new, cap=4):
    examples.setdefault(cat, [])
    if len(examples[cat]) < cap:
        examples[cat].append((orig, new))

for sh in SHARDS:
    path = DIR / f"train_ar_q3asr_shard_{sh}_selected.jsonl"
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        text, orig = rec["text"], rec["original_text"]
        n += 1
        lens.append(len(text))
        for t in rec.get("changes", []):
            tag_counter[t] += 1
        dialect_counter[rec.get("dialect", "?")] += 1
        c = rec.get("confidence", 0)
        conf_buckets[round(c, 1)] += 1

        # defects
        if ARTIFACT.search(text):
            defects.setdefault("artifact", []).append(text[:120])
        if CONTROL.search(text):
            defects.setdefault("control_char", []).append(repr(text[:80]))
        if any(ch in EASTERN for ch in text):
            defects.setdefault("eastern_digit", []).append(text[:120])
        if "\u06d4" in text:
            defects.setdefault("urdu_fullstop", []).append(text[:120])
        if LATIN_PUNCT.search(text):
            defects.setdefault("latin_punct", []).append(text[:120])
        if TATWEEL in text:
            defects.setdefault("tatweel", []).append(text[:120])
        if FALSE_START.search(text):
            defects.setdefault("residual_false_start", []).append(text[:120])

        # diacritics density
        letters = len(AR_LETTER.findall(text))
        dcount = sum(1 for ch in text if ch in DIAC)
        ratio = dcount / letters if letters else 0
        if ratio == 0: diac_buckets["0%"] += 1
        elif ratio < 0.05: diac_buckets["<5%"] += 1
        elif ratio < 0.15: diac_buckets["5-15%"] += 1
        elif ratio < 0.4: diac_buckets["15-40%"] += 1
        else: diac_buckets[">=40% (over-vocalized)"] += 1

        if text == orig:
            identical += 1
            continue

        # categorize the change
        t_nd, o_nd = strip_diac(text), strip_diac(orig)
        if t_nd == o_nd:
            cat_counter["diacritics_only"] += 1
            add_ex("diacritics_only", orig, text)
            continue
        t_np, o_np = strip_punct(t_nd), strip_punct(o_nd)
        if t_np == o_np:
            cat_counter["punct_only(+diac)"] += 1
            add_ex("punct_only(+diac)", orig, text)
            continue
        # digit conversion?
        if WESTERN_DIGIT.search(text) and not WESTERN_DIGIT.search(orig):
            cat_counter["itn_digits"] += 1
            add_ex("itn_digits", orig, text)
            continue
        tw, ow = words(t_np), words(o_np)
        if len(tw) == len(ow):
            diffs = [(a, b) for a, b in zip(ow, tw) if a != b]
            # spelling-normalization? (same skeleton after hamza/ta-marbuta/ya folding)
            def fold(w):
                return (w.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                         .replace("ة","ه").replace("ى","ي").replace("ئ","ي")
                         .replace("ؤ","و").replace("ء",""))
            if all(fold(a) == fold(b) for a, b in diffs):
                cat_counter["spelling_norm(hamza/taa/ya)"] += 1
                add_ex("spelling_norm(hamza/taa/ya)", orig, text)
            else:
                cat_counter["word_substitution"] += 1
                add_ex("word_substitution", orig, text, cap=8)
                if len(word_sub_examples) < 30:
                    word_sub_examples.append((diffs, orig, text))
        elif len(tw) < len(ow):
            cat_counter["words_removed"] += 1
            add_ex("words_removed", orig, text, cap=6)
        else:
            cat_counter["words_added"] += 1
            add_ex("words_added", orig, text, cap=6)

        if len(orig) > 20 and len(text) < 0.5 * len(orig):
            shrink_flags.append((orig, text))

print(f"records={n} identical={identical} ({identical/n:.1%}) changed={n-identical} ({(n-identical)/n:.1%})")
print(f"avg_len={sum(lens)/n:.0f} chars")
print("\nchange tags:", dict(tag_counter.most_common()))
print("dialects:", dict(dialect_counter.most_common()))
print("confidence:", dict(sorted(conf_buckets.items())))
print("diacritic density:", dict(diac_buckets.most_common()))
print("\nchange categories (changed records):")
for k, v in cat_counter.most_common():
    print(f"  {k:32s} {v:6d}  ({v/(n-identical):.1%} of changed)")
print("\ndefects:")
for k, v in defects.items():
    print(f"  {k}: {len(v)}   e.g. {v[0]}")
if not defects:
    print("  NONE")
print(f"\nshrink>50%: {len(shrink_flags)}")
for o, t in shrink_flags[:5]:
    print(f"  ORIG: {o}\n  NEW : {t}")

print("\n=== EXAMPLES ===")
for cat, exs in examples.items():
    print(f"\n[{cat}]")
    for o, t in exs:
        print(f"  ORIG: {o}")
        print(f"  NEW : {t}")

print("\n=== WORD SUBSTITUTION DIFF PAIRS (fidelity risk triage) ===")
for diffs, o, t in word_sub_examples[:15]:
    print(f"  pairs={diffs}")
