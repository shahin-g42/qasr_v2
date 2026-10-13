"""Frozen sample panels, deterministic bounded selection, and overlap checks."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from pathlib import Path

from .io import (
    digest,
    file_hash,
    fresh_directory,
    positive_number,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)

SCHEMA = "qasr-lab-panel-v1"


def audio_path(row, base):
    value = next(
        (
            row[key]
            for key in ("audio_filepath", "audio_path", "wav_path")
            if isinstance(row.get(key), str) and row[key]
        ),
        None,
    )
    if value is None:
        raise ValueError("Every sample requires an audio_filepath")
    path = Path(value).expanduser()
    return str((Path(base) / path).resolve() if not path.is_absolute() else path.resolve())


def canonical_sample(row, *, base, language=None):
    path = audio_path(row, base)
    text = next((row[key] for key in ("text", "reference", "transcript") if key in row), None)
    if not isinstance(text, str):
        raise ValueError(f"{path}: text must be a string; use an empty string for non-speech")
    lang = row.get("language") or language
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError(f"{path}: language is required")
    duration = positive_number(row.get("duration", row.get("audio_seconds")), "duration")
    sample_id = row.get("id", digest(path)[:24])
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Sample id must be a nonempty string")
    tags = row.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError(f"{sample_id}: tags must be a list of nonempty strings")
    sample = {
        "id": sample_id,
        "audio_filepath": path,
        "text": text,
        "language": lang.lower(),
        "duration": duration,
        "tags": sorted(set(tags)),
        "stratum": row.get("stratum") or ("speech" if text.strip() else "non_speech"),
    }
    if not isinstance(sample["stratum"], str):
        raise ValueError("stratum must be a string")
    for key in ("source", "speaker_id", "recording_id", "audio_sha256"):
        if row.get(key) is not None:
            sample[key] = str(row[key])
    return sample


def identity_keys(row, base):
    keys = {("path", audio_path(row, base))}
    if row.get("audio_sha256"):
        keys.add(("audio_sha256", str(row["audio_sha256"])))
    # Speaker/recording labels must be corpus-qualified, never guess a namespace.
    if row.get("source"):
        for key in ("speaker_id", "recording_id"):
            if row.get(key) is not None:
                keys.add((key, str(row["source"]), str(row[key])))
    return keys


def leakage_check(samples, manifests):
    identities = defaultdict(set)
    for row in samples:
        for key in identity_keys(row, "/"):
            identities[key].add(row["id"])
    matches = []
    scanned = 0
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        for row in read_jsonl(manifest):
            scanned += 1
            for key in identity_keys(row, manifest.parent):
                for sample_id in identities.get(key, ()):
                    if len(matches) < 30:
                        matches.append(
                            {"sample_id": sample_id, "kind": key[0], "manifest": str(manifest)}
                        )
    return {
        "status": "overlap" if matches else ("checked" if manifests else "not_checked"),
        "scanned_training_rows": scanned,
        "examples": matches,
        "scope": "resolved paths, supplied audio hashes, corpus-qualified speaker/recording IDs",
        "content_hashes_computed": False,
    }


def validate_samples(samples):
    if not samples:
        raise ValueError("Panel is empty")
    ids, paths = set(), set()
    for row in samples:
        if row["id"] in ids or row["audio_filepath"] in paths:
            raise ValueError(f"Duplicate sample id or audio path: {row['id']}")
        ids.add(row["id"])
        paths.add(row["audio_filepath"])


def save_panel(samples, output, provenance=None):
    samples = sorted(samples, key=lambda row: row["id"])
    validate_samples(samples)
    out = fresh_directory(output)
    write_jsonl(out / "samples.jsonl", samples)
    metadata = {
        "schema": SCHEMA,
        "panel_sha256": digest(samples),
        "samples_file_sha256": file_hash(out / "samples.jsonl"),
        "samples": len(samples),
        "languages": dict(Counter(row["language"] for row in samples)),
        "provenance": provenance or {},
    }
    write_json(out / "panel.json", metadata)
    return metadata


def load_panel(path):
    path = Path(path)
    metadata = read_json(path / "panel.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported panel schema")
    samples = list(read_jsonl(path / "samples.jsonl"))
    validate_samples(samples)
    if digest(samples) != metadata["panel_sha256"]:
        raise ValueError("Panel content changed after it was frozen")
    if file_hash(path / "samples.jsonl") != metadata["samples_file_sha256"]:
        raise ValueError("Panel file checksum mismatch")
    return metadata, samples


def prepare_panel(
    manifests,
    output,
    *,
    per_stratum=500,
    seed=42,
    language=None,
    audio_root=None,
    exclude_manifests=(),
    require_audio=False,
    hash_audio=False,
):
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    heaps = defaultdict(list)
    scanned = 0
    sources = []
    for manifest in manifests:
        path = Path(manifest).resolve()
        sources.append({"path": str(path), "sha256": file_hash(path)})
        for row in read_jsonl(path):
            scanned += 1
            sample = canonical_sample(row, base=audio_root or path.parent, language=language)
            key = (sample["language"], sample["stratum"])
            rank = int(digest([seed, sample["id"]]), 16)
            entry = (-rank, scanned, sample)
            if len(heaps[key]) < per_stratum:
                heapq.heappush(heaps[key], entry)
            elif entry[0] > heaps[key][0][0]:
                heapq.heapreplace(heaps[key], entry)
    samples = [entry[2] for heap in heaps.values() for entry in heap]
    validate_samples(samples)
    if require_audio:
        for row in samples:
            if not Path(row["audio_filepath"]).is_file():
                raise FileNotFoundError(row["audio_filepath"])
    if hash_audio:
        for row in samples:
            actual = file_hash(row["audio_filepath"])
            if row.get("audio_sha256") and actual != row["audio_sha256"]:
                raise ValueError(f"Audio checksum mismatch: {row['id']}")
            row["audio_sha256"] = actual
    overlap = leakage_check(samples, exclude_manifests)
    if overlap["status"] == "overlap":
        raise ValueError(f"Training/evaluation overlap: {overlap['examples']}")
    return save_panel(
        samples,
        output,
        {
            "sources": sources,
            "seed": seed,
            "per_language_stratum_limit": per_stratum,
            "input_rows": scanned,
            "leakage": overlap,
            "selected_audio_hashes_computed": hash_audio,
            "excluded_manifests": [
                {"path": str(Path(p).resolve()), "sha256": file_hash(p)} for p in exclude_manifests
            ],
        },
    )
