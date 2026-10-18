"""Immutable pilot selection and provenance. No model or audio dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = ("sada", "commonvoice", "masc_clean", "masc_noisy", "mgb2", "casablanca")
POLICY = {
    "version": "arabic-reference-pilot-v1",
    "purpose": "evaluation_only",
    "audio_review_required": True,
    "number_policy": "Preserve spoken number words; digit conversion is deferred in v1.",
    "diacritics": "Conservative, disambiguating marks; do not invent case endings.",
    "content": "Preserve dialect, code switching, numbers, negation, names, audible repetitions and incomplete speech. Never paraphrase or complete quotations from memory.",
    "unclear": "Never infer missing speech from fluency. Unresolved clips stay explicitly unresolved.",
    "verification": "Text-only corrector output is always proposed, never verified.",
    "publication": "Internal pilot; source rights, independent review and reference freeze are required before a public benchmark release.",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(
        path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


def format_signature(text):
    """Formatting may change punctuation/marks/spacing/case, never base letters or digits."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text).casefold()
        if not c.isspace() and c != "\u0640" and unicodedata.category(c)[0] not in ("P", "M")
    )


def binding(row):
    return digest({k: v for k, v in row.items() if k != "binding"})


def validate_source(row):
    ds = row.get("dataset")
    if ds not in DATASETS or not re.fullmatch(re.escape(str(ds)) + r":\d{6}", row.get("id", "")):
        raise ValueError(f"Invalid benchmark ID: {row.get('id')}")
    if type(row.get("ordinal")) is not int or int(row["id"].split(":")[1]) != row["ordinal"]:
        raise ValueError(f"Ordinal mismatch: {row['id']}")
    for key in ("text", "pred_text", "audio", "audio_sha256", "preparation_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{row['id']}: missing {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", row["audio_sha256"]):
        raise ValueError("Invalid audio SHA256")
    duration = row.get("actual_duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{row['id']}: invalid actual_duration")
    if row.get("mode") != "arabic":
        raise ValueError("Pilot v1 accepts forced-Arabic predictions only")


def prepare(predictions, output, per_dataset=100, seed=20261019):
    if per_dataset < 1 or not predictions:
        raise ValueError("Positive sample size and prediction inputs required")
    output = Path(output)
    pools, source_info = [], []
    names = set()
    for name, path in predictions:
        if not name or name in names:
            raise ValueError("Candidate names must be nonempty and unique")
        names.add(name)
        pool = {}
        before = sha256(path)
        for row in read_jsonl(path):
            validate_source(row)
            if row["id"] in pool:
                raise ValueError(f"Duplicate ID: {row['id']}")
            pool[row["id"]] = row
        if before != sha256(path):
            raise ValueError(f"Input changed while reading: {path}")
        pools.append(pool)
        run_path = Path(path).parent / "run.json"
        source_info.append(
            {
                "name": name,
                "predictions_sha256": before,
                "run": json.loads(run_path.read_text()) if run_path.exists() else None,
            }
        )
    base = pools[0]
    for pool in pools[1:]:
        if set(pool) != set(base):
            raise ValueError("Candidate ID coverage differs")
        for identifier, row in base.items():
            for key in ("text", "audio_sha256", "preparation_id", "actual_duration", "dataset"):
                if pool[identifier][key] != row[key]:
                    raise ValueError(f"{identifier}: candidate {key} mismatch")
    counts = Counter(row["dataset"] for row in base.values())
    if set(counts) != set(DATASETS):
        raise ValueError("Pilot requires all six datasets")
    hashes = defaultdict(list)
    for row in base.values():
        hashes[row["audio_sha256"]].append(row["id"])
    flags = defaultdict(set)
    for row in base.values():
        identifier = row["id"]
        if re.search(r"غير\s*واضح", row["text"]):
            flags[identifier].add("unclear_reference_candidate")
        if any(pool[identifier].get("generation_limit_hit") is True for pool in pools):
            flags[identifier].add("generation_limit")
        if any(re.search(r"[A-Za-z]", pool[identifier]["pred_text"]) for pool in pools):
            flags[identifier].add("latin_prediction_candidate")
    for identifiers in hashes.values():
        if len(identifiers) > 1:
            conflicting = len({format_signature(base[i]["text"]) for i in identifiers}) > 1
            for identifier in identifiers:
                flags[identifier].add("duplicate_audio")
                if conflicting:
                    flags[identifier].add("conflicting_duplicate_reference")
    pilot = set()
    for ds in DATASETS:
        # Sampling ignores all predictions and scores. Exact duplicates occupy one slot.
        representatives = {}
        for row in sorted(base.values(), key=lambda r: r["id"]):
            if row["dataset"] == ds:
                representatives.setdefault(row["audio_sha256"], row["id"])
        ranked = sorted(representatives, key=lambda h: digest([seed, ds, h]))
        if len(ranked) < per_dataset:
            raise ValueError(f"{ds}: fewer unique audio clips than requested")
        pilot.update(representatives[h] for h in ranked[:per_dataset])
    selection = pilot | set(flags)
    rows = []
    for identifier in sorted(selection):
        raw = base[identifier]
        row = {
            k: raw[k]
            for k in (
                "id",
                "dataset",
                "ordinal",
                "text",
                "audio",
                "audio_sha256",
                "preparation_id",
                "actual_duration",
            )
        }
        row.update(
            {
                "source": raw.get("source"),
                "metadata": raw.get("metadata", {}),
                "relative_audio": raw.get("relative_audio"),
                "split": "test",
                "purpose": "evaluation_only",
                "verified_text": None,
                "ideal_text": None,
                "review_status": "pending",
                "policy_sha256": digest(POLICY),
                "cohorts": (["pilot"] if identifier in pilot else [])
                + (["diagnostic"] if identifier in flags else []),
                "flags": sorted(flags.get(identifier, [])),
                "candidates": [
                    {
                        "name": name,
                        "text": pool[identifier]["pred_text"],
                        "generation_limit_hit": pool[identifier].get("generation_limit_hit", False),
                        "generated_tokens": pool[identifier].get("generated_tokens"),
                    }
                    for (name, _), pool in zip(predictions, pools, strict=True)
                ],
            }
        )
        row["binding"] = binding(row)
        rows.append(row)
    plan = {
        "schema_version": 1,
        "policy": POLICY,
        "policy_sha256": digest(POLICY),
        "seed": seed,
        "per_dataset": per_dataset,
        "sources": source_info,
        "input_counts": dict(counts),
        "pilot_clips": len(pilot),
        "selected_clips": len(rows),
        "diagnostic_clips": len(flags),
        "cohort_overlap": len(pilot & set(flags)),
        "flag_counts": dict(Counter(f for row in rows for f in row["flags"])),
        "manifest_digest": digest(rows),
        "sampling": "100 per dataset by default; uniform hash ranking of unique prepared audio. Diagnostics are a separate, overlapping, outcome-selected cohort. Neither is a full-suite score.",
    }
    if (output / "plan.json").exists():
        old, previous = load_plan(output)
        if old != plan or previous != rows:
            raise ValueError("Existing pilot differs; choose a new output directory")
        return plan
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "plan.json", plan)
    return plan


def load_plan(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    rows = list(read_jsonl(root / "manifest.jsonl"))
    if plan["policy_sha256"] != digest(POLICY) or plan["policy"] != POLICY:
        raise ValueError("Policy changed; use the code matching this plan")
    if plan["manifest_digest"] != digest(rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Pilot manifest changed or contains duplicate IDs")
    if any(r["binding"] != binding(r) for r in rows):
        raise ValueError("Row binding mismatch")
    return plan, rows
