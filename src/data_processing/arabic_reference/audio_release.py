"""Package exact benchmark waveforms into bounded, playable Hugging Face shards."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import write_json
from .release import dataset_card, finish_bundle, valid_release_id, validate_bundle


def encode_audio(row, audio_root):
    import numpy as np
    import soundfile as sf

    dataset, separator, ordinal = row["id"].partition(":")
    if not separator or dataset != row["dataset"] or not re.fullmatch(r"[0-9]{6}", ordinal):
        raise ValueError("Invalid audio row identifier")
    path = audio_root / dataset / (ordinal + ".npy")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["audio_sha256"]:
        raise ValueError(f"Prepared audio hash mismatch: {row['id']}")
    wave = np.load(io.BytesIO(raw), allow_pickle=False)
    if (
        wave.dtype != np.dtype("float32")
        or wave.ndim != 1
        or not len(wave)
        or not np.isfinite(wave).all()
    ):
        raise ValueError(f"Expected finite mono float32 benchmark audio: {row['id']}")
    if abs(len(wave) / 16000 - row["duration_seconds"]) > 1 / 16000:
        raise ValueError(f"Audio duration mismatch: {row['id']}")
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format="WAV", subtype="FLOAT")
    encoded = buffer.getvalue()
    decoded, rate = sf.read(io.BytesIO(encoded), dtype="float32")
    if rate != 16000 or not np.array_equal(decoded, wave):
        raise ValueError(f"Audio encoding changed samples: {row['id']}")
    return {
        **row,
        "audio": {"bytes": encoded, "path": f"{dataset}-{ordinal}.wav"},
        "audio_bytes_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def package_audio(
    annotations,
    audio_root,
    output,
    *,
    release_id="v4.4-audio-001",
    workers=16,
    shard_mib=256,
    emit=print,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Features, Value

    valid_release_id(release_id)
    if not 1 <= workers <= 90 or not 1 <= shard_mib <= 1024:
        raise ValueError("Use 1–90 audio workers and 1–1024 MiB shards")
    annotations, audio_root, output = (
        Path(annotations).resolve(),
        Path(audio_root).resolve(),
        Path(output).resolve(),
    )
    if annotations == output or annotations in output.parents or output in annotations.parents:
        raise ValueError("Audio output must be separate from the annotation bundle")
    source = validate_bundle(annotations)
    if source["repo_type"] != "dataset":
        raise ValueError("Expected an annotation dataset bundle")
    provenance = json.loads((annotations / "provenance.json").read_text())
    if provenance["audio_included"]:
        raise ValueError("This input already contains audio")
    if output.exists():
        previous = validate_bundle(output)
        if (
            previous.get("annotations_bundle_sha256") != source["bundle_sha256"]
            or previous["release_id"] != release_id
        ):
            raise ValueError("Audio release already exists with different inputs/version")
        return {"output": str(output), **previous}
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum(c["rows"] for c in source["counts"].values())
    seen, completed, wav_bytes, seconds = set(), 0, 0, 0.0
    with tempfile.TemporaryDirectory(prefix=".reference-audio-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ds in source["datasets"]:
                relative = f"data/{ds}/test.parquet"
                if relative not in source["files"]:
                    raise ValueError(f"Missing annotation table: {ds}")
                parquet = pq.ParquetFile(annotations / relative)
                features = Features.from_arrow_schema(parquet.schema_arrow)
                features["audio"] = Audio(sampling_rate=16000)
                features["audio_bytes_sha256"] = Value("string")
                schema = features.arrow_schema
                shard_rows, size, shard_number, count = [], 0, 0, 0

                def flush(ds=ds, schema=schema):
                    nonlocal shard_rows, size, shard_number
                    path = stage / "data" / ds / f"test-{shard_number:05d}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(
                        pa.Table.from_pylist(shard_rows, schema=schema), path, compression="zstd"
                    )
                    shard_number += 1
                    shard_rows, size = [], 0

                # Bounded submissions: do not queue the whole corpus or retain all audio in RAM.
                for batch in parquet.iter_batches(batch_size=workers * 2):
                    rows = batch.to_pylist()
                    for row in rows:
                        if row["id"] in seen or row["dataset"] != ds:
                            raise ValueError("Duplicate or mismatched annotation row")
                        seen.add(row["id"])
                    for result in pool.map(lambda row: encode_audio(row, audio_root), rows):
                        estimate = len(result["audio"]["bytes"]) + len(
                            json.dumps(
                                {k: v for k, v in result.items() if k != "audio"},
                                ensure_ascii=False,
                            ).encode()
                        )
                        if shard_rows and size + estimate > shard_mib * 1024**2:
                            flush()
                        result["release_id"] = release_id
                        shard_rows.append(result)
                        size += estimate
                        wav_bytes += len(result["audio"]["bytes"])
                        seconds += result["duration_seconds"]
                        completed += 1
                        count += 1
                    emit(
                        json.dumps(
                            {
                                "event": "audio_release_progress",
                                "completed": completed,
                                "total": total,
                                "dataset": ds,
                            }
                        )
                    )
                if shard_rows:
                    flush()
                if count != source["counts"][ds]["rows"]:
                    raise ValueError(f"Annotation row count mismatch: {ds}")
        if completed != total:
            raise ValueError("Audio release coverage mismatch")
        provenance.update(
            {
                "release_id": release_id,
                "audio_included": True,
                "annotations_bundle_sha256": source["bundle_sha256"],
                "annotations_release_id": source["release_id"],
                "audio_encoding": {
                    "format": "WAV",
                    "subtype": "FLOAT",
                    "sampling_rate": 16000,
                    "channels": 1,
                    "exact_float32_samples": True,
                    "records": completed,
                    "wav_bytes": wav_bytes,
                    "hours": seconds / 3600,
                },
            }
        )
        provenance["limitations"] = [
            s for s in provenance["limitations"] if not s.startswith("No audio bytes")
        ]
        write_json(stage / "provenance.json", provenance)
        (stage / "policy.json").write_bytes((annotations / "policy.json").read_bytes())
        (stage / "README.md").write_text(dataset_card(source["datasets"], provenance))
        # Recheck the frozen annotation input before making the output visible.
        validate_bundle(annotations)
        return finish_bundle(
            stage,
            output,
            {
                "repo_type": "dataset",
                "release_id": release_id,
                "manifest_digest": source["manifest_digest"],
                "counts": source["counts"],
                "datasets": source["datasets"],
                "audio_included": True,
                "annotations_bundle_sha256": source["bundle_sha256"],
            },
        )
