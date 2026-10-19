"""Read every written audio row before a release is made available for upload."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def annotation_summary(rows):
    groups = defaultdict(list)
    sources = Counter()
    flags = Counter()
    durations = defaultdict(list)
    statuses = Counter()
    reviewed = 0
    for row in rows:
        durations[row["dataset"]].append(row["duration_seconds"])
        statuses[row["proposal_status"]] += 1
        reviewed += int(row["audio_reviewed"])
        groups[row["audio_sha256"]].append(row)
        source = json.loads(row["source_locator_json"])
        if source.get("repository"):
            sources[(row["dataset"], source["repository"], source.get("revision"))] += 1
        checks = json.loads(row["checks_json"]) if row.get("checks_json") else {}
        flags.update(checks.get("editorial_flags", []))
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {
        "rows": sum(statuses.values()),
        "hours": sum(sum(values) for values in durations.values()) / 3600,
        "statuses": dict(statuses),
        "audio_reviewed_transcripts": reviewed,
        "datasets": {
            ds: {
                "rows": len(values),
                "hours": sum(values) / 3600,
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
            for ds, values in durations.items()
        },
        "sources": [
            {"dataset": ds, "repository": repo, "revision": rev, "rows": count}
            for (ds, repo, rev), count in sorted(sources.items(), key=lambda pair: str(pair[0]))
        ],
        "editorial_flags": dict(flags),
        "duplicate_audio_groups": len(duplicates),
        "duplicate_extra_rows": sum(len(group) - 1 for group in duplicates),
        "duplicate_groups_with_distinct_original_text": sum(
            len({r["text"] for r in group}) > 1 for group in duplicates
        ),
    }


def audit_audio_shards(root, annotations, data_files, release_id, *, emit=print):
    """Independently decode persisted shards and compare every annotation column."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from datasets import Audio, Features

    root, annotations = Path(root), Path(annotations)
    expected = {}
    schema = None
    for ds in data_files:
        table = pq.read_table(annotations / "data" / ds / "test.parquet")
        if schema is None:
            schema = table.schema
        elif not schema.equals(table.schema):
            raise ValueError("Annotation schemas differ between datasets")
        for row in table.to_pylist():
            if row["id"] in expected or row["dataset"] != ds:
                raise ValueError("Duplicate or misplaced source ID")
            if not math.isfinite(row["duration_seconds"]) or row["duration_seconds"] <= 0:
                raise ValueError("Invalid source duration")
            expected[row["id"]] = row
    seen = set()
    silence, out_of_range, max_peak = [], [], 0.0
    output_schema = None
    for ds, paths in data_files.items():
        for relative in paths:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe audio shard path")
            parquet = pq.ParquetFile(root / relative)
            features = Features.from_arrow_schema(parquet.schema_arrow)
            if (
                not isinstance(features.get("audio"), Audio)
                or features["audio"].sampling_rate != 16000
                or parquet.schema_arrow.names[0] != "audio"
            ):
                raise ValueError("Missing or incorrect first-column Hugging Face Audio metadata")
            if output_schema is None:
                output_schema = parquet.schema_arrow
            elif not output_schema.equals(parquet.schema_arrow, check_metadata=True):
                raise ValueError("Audio shard schemas differ")
            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    key = row["id"]
                    if key not in expected or key in seen or row["dataset"] != ds:
                        raise ValueError("Unknown, duplicate or misplaced output ID")
                    original = expected[key]
                    for field, value in original.items():
                        target = release_id if field == "release_id" else value
                        if row.get(field) != target:
                            raise ValueError(
                                f"Annotation column changed in audio packaging: {key}/{field}"
                            )
                    if row["speculative"] != (row["proposal_status"] == "speculative_draft"):
                        raise ValueError("Speculative status mismatch")
                    if not row["audio_reviewed"] and (
                        row["verified_text"] is not None or row["ideal_text"] is not None
                    ):
                        raise ValueError("Unreviewed final reference value")
                    if row["proposal_status"] in {"proposed", "speculative_draft"} and (
                        not row["verified_text_draft"] or not row["ideal_text_draft"]
                    ):
                        raise ValueError("Completed draft is missing its text")
                    audio = row["audio"]
                    if not audio["bytes"] or audio["path"] != key.replace(":", "-") + ".wav":
                        raise ValueError("Missing audio bytes or invalid portable audio filename")
                    if hashlib.sha256(audio["bytes"]).hexdigest() != row["audio_bytes_sha256"]:
                        raise ValueError("Embedded WAV checksum mismatch")
                    with sf.SoundFile(io.BytesIO(audio["bytes"])) as sound:
                        if (
                            sound.samplerate != 16000
                            or sound.channels != 1
                            or sound.subtype != "FLOAT"
                        ):
                            raise ValueError(
                                "Audio encoding differs from documented 16 kHz mono WAV FLOAT"
                            )
                        wave = sound.read(dtype="float32")
                    if not len(wave) or not np.isfinite(wave).all():
                        raise ValueError("Empty or non-finite decoded audio")
                    if (
                        len(wave) != row["audio_num_samples"]
                        or abs(len(wave) / 16000 - row["duration_seconds"]) > 1 / 16000
                    ):
                        raise ValueError("Decoded sample count or duration mismatch")
                    if (
                        hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest()
                        != row["audio_samples_sha256"]
                    ):
                        raise ValueError("Decoded waveform checksum mismatch")
                    peak = float(np.max(np.abs(wave)))
                    max_peak = max(max_peak, peak)
                    if peak == 0:
                        silence.append(key)
                    if peak > 1:
                        out_of_range.append(key)
                    seen.add(key)
            emit(
                json.dumps(
                    {"event": "audio_release_audit", "checked": len(seen), "total": len(expected)}
                )
            )
    if seen != set(expected):
        raise ValueError("Audio release is missing source IDs")
    summary = annotation_summary(expected.values())
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "all_rows_and_written_audio",
        "release_id": release_id,
        "decoded_audio_rows": len(seen),
        "all_annotation_columns_preserved": True,
        "schema_consistent": True,
        "sample_hashes_verified": True,
        "source_cache_hashes_checked_during_encoding": True,
        "max_absolute_amplitude": max_peak,
        "signal_observations": {"all_zero_audio": silence, "peak_above_one": out_of_range},
        "signal_note": "Observations retain the benchmark waveform; no clipping, gain normalization or silent row removal is applied.",
        "transcript_accuracy_certified": False,
        **summary,
    }
