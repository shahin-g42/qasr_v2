"""Independent implementation of the pinned ELM scoring protocol.

Specification: upstream eval.py at 8fce8595b49057c98bafc2ba41b69c331cb5176e.
NeMo's CER includes spaces. Do not replace this with gqasr.metrics.
"""
import re
from collections import defaultdict

from .common import DATASETS


def normalize_elm(text):
    text = re.sub(r'''[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]''', "", text)
    text = re.sub("[\u064b-\u0652]", "", text)
    text = text.translate(str.maketrans({
        "پ": "ب", "ڤ": "ف", "آ": "ا", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ي", "ء": "",
        **dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789", strict=True)),
    }))
    text = re.sub(r"(^|\s)و\s+", r"\1و", text)
    return " ".join(text.split())


def counts(reference, hypothesis, normalized=False):
    # RapidFuzz uses compiled Levenshtein, accepts both strings and token lists.
    from rapidfuzz.distance import Levenshtein

    if normalized:
        reference, hypothesis = normalize_elm(reference), normalize_elm(hypothesis)
    else:
        reference, hypothesis = " ".join(reference.split()), " ".join(hypothesis.split())
    return dict(utterances=1, word_errors=Levenshtein.distance(reference.split(), hypothesis.split()),
                words=len(reference.split()), character_errors=Levenshtein.distance(reference, hypothesis),
                characters=len(reference), empty_references=int(not reference))


def rates(c):
    return {**c, "wer": c["word_errors"] / c["words"] if c["words"] else None,
            "cer": c["character_errors"] / c["characters"] if c["characters"] else None}


def aggregate(rows, full_suite=False):
    groups = defaultdict(lambda: defaultdict(int))
    seen = set()
    for row in rows:
        if row["id"] in seen or row["dataset"] not in DATASETS:
            raise ValueError(f"Duplicate ID or unknown dataset: {row['id']}")
        seen.add(row["id"])
        for mode in ("strict", "elm"):
            c = counts(row["text"], row["pred_text"], mode == "elm")
            labels = ["pooled", row["dataset"]]
            for field in ("dialect", "gender", "age", "environment"):
                if row.get("metadata", {}).get(field):
                    labels.append(f"{row['dataset']}/{field}:{row['metadata'][field]}")
            for label in labels:
                for k, v in c.items():
                    groups[f"{mode}/{label}"][k] += v
    result = {k: rates(v) for k, v in groups.items()}
    # Equal weight per test set, not a pooled/micro score or mean utterance WER.
    for mode in ("strict", "elm"):
        values = [result[f"{mode}/{ds}"] for ds in DATASETS if f"{mode}/{ds}" in result]
        result[f"{mode}/macro"] = {
            "datasets": len(values), "full_suite": full_suite and len(values) == len(DATASETS),
            **{m: sum(v[m] for v in values) / len(values)
               if values and all(v[m] is not None for v in values) else None for m in ("wer", "cer")},
        }
    return result
