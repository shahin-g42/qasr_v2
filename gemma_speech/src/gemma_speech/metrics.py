"""Auditable corpus-level scores, keeping strict and Arabic-normalized views separate."""

import re
import unicodedata
from collections import defaultdict

from .text import normalize_text

ARABIC_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def score_text(text, arabic_normalized=False):
    text = normalize_text(text)
    if arabic_normalized:
        text = ARABIC_MARKS.sub("", text).replace("ـ", "")
        text = text.translate(str.maketrans("أإآٱ", "اااا"))
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
        text = normalize_text(text).casefold()
    return text


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score_predictions(rows):
    groups = defaultdict(lambda: defaultdict(int))
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"duplicate prediction ID: {row['id']}")
        ids.add(row["id"])
        for normalized in (False, True):
            ref = score_text(row["reference"], normalized and row["language"].startswith("ar"))
            hyp = score_text(row["hypothesis"], normalized and row["language"].startswith("ar"))
            mode = "arabic_normalized" if normalized else "strict"
            labels = ["all", f"language:{row['language']}"]
            if row.get("dialect") and row["dialect"] != "unspecified":
                labels.append(f"dialect:{row['language']}:{row['dialect']}")
            for group in labels:
                counts = groups[f"{mode}/{group}"]
                counts["utterances"] += 1
                counts["word_errors"] += edit_distance(ref.split(), hyp.split())
                counts["words"] += len(ref.split())
                counts["character_errors"] += edit_distance(
                    ref.replace(" ", ""), hyp.replace(" ", "")
                )
                counts["characters"] += len(ref.replace(" ", ""))
                counts["empty_references"] += not bool(ref)
                counts["nonempty_on_empty"] += not bool(ref) and bool(hyp)
    results = {}
    for group, counts in groups.items():
        results[group] = dict(counts)
        results[group]["wer"] = counts["word_errors"] / counts["words"] if counts["words"] else None
        results[group]["cer"] = (
            counts["character_errors"] / counts["characters"] if counts["characters"] else None
        )
    return results
