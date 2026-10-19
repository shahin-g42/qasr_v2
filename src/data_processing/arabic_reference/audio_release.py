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
from .release_quality import audit_audio_shards

AUDIO_PACKAGE_VERSION = "arabic-audio-release-v2"


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
        "audio_num_samples": len(wave),
        "audio_samples_sha256": hashlib.sha256(
            wave.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
    }


def package_audio(
    annotations,
    audio_root,
    output,
    *,
    release_id="v4.5-audio-001",
    repo_id="shahink/asr_bench",
    title="Arabic ASR Bench · Enhanced References",
    datasets=None,
    workers=16,
    shard_mib=256,
    emit=print,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Features, Value

    valid_release_id(release_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise ValueError("Use an explicit owner/repository dataset ID")
    if not title.strip() or len(title) > 120 or any(ord(c) < 32 for c in title):
        raise ValueError("Use a single-line dataset title of 1–120 characters")
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
    selected = list(source["datasets"] if datasets is None else datasets)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(source["datasets"])
    ):
        raise ValueError("Choose distinct dataset names present in the annotation bundle")
    selected = [ds for ds in source["datasets"] if ds in selected]
    package_config = {
        "version": AUDIO_PACKAGE_VERSION,
        "dataset_repo": repo_id,
        "dataset_title": title,
        "datasets": selected,
        "shard_mib": shard_mib,
    }
    if output.exists():
        previous = validate_bundle(output)
        if (
            previous.get("annotations_bundle_sha256") != source["bundle_sha256"]
            or previous["release_id"] != release_id
            or previous.get("package_config") != package_config
        ):
            raise ValueError("Audio release already exists with different inputs/version")
        return {"output": str(output), **previous}
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum(source["counts"][ds]["rows"] for ds in selected)
    data_files = {ds: [] for ds in selected}
    seen, completed, wav_bytes, seconds = set(), 0, 0, 0.0
    with tempfile.TemporaryDirectory(prefix=".reference-audio-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ds in selected:
                relative = f"data/{ds}/test.parquet"
                if relative not in source["files"]:
                    raise ValueError(f"Missing annotation table: {ds}")
                parquet = pq.ParquetFile(annotations / relative)
                features = Features(
                    {
                        "audio": Audio(sampling_rate=16000),
                        **Features.from_arrow_schema(parquet.schema_arrow),
                    }
                )
                features["audio_bytes_sha256"] = Value("string")
                features["audio_num_samples"] = Value("int64")
                features["audio_samples_sha256"] = Value("string")
                schema = features.arrow_schema
                shard_rows, size, shard_number, count = [], 0, 0, 0

                def flush(ds=ds, schema=schema):
                    nonlocal shard_rows, size, shard_number
                    path = stage / "data" / ds / f"test-{shard_number:05d}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(
                        pa.Table.from_pylist(shard_rows, schema=schema), path, compression="zstd"
                    )
                    data_files[ds].append(path.relative_to(stage).as_posix())
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
        quality = audit_audio_shards(stage, annotations, data_files, release_id, emit=emit)
        write_json(stage / "quality_report.json", quality)
        provenance.update(
            {
                "release_id": release_id,
                "dataset_repo": repo_id,
                "dataset_title": title,
                "package_config": package_config,
                "data_files": data_files,
                "quality": quality,
                "source_selection_scope": provenance["selection_scope"],
                "selection_scope": provenance["selection_scope"]
                if selected == source["datasets"]
                else "dataset_subset",
                "selected_datasets": selected,
                "counts": {ds: source["counts"][ds] for ds in selected},
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
        if provenance.get("speculative_drafts"):
            provenance["speculative_drafts"]["records_in_release"] = quality["statuses"].get(
                "speculative_draft", 0
            )
        provenance["limitations"] = [
            s for s in provenance["limitations"] if not s.startswith("No audio bytes")
        ]
        write_json(stage / "provenance.json", provenance)
        (stage / "policy.json").write_bytes((annotations / "policy.json").read_bytes())
        (stage / "README.md").write_text(dataset_card(selected, provenance))
        # Recheck the frozen annotation input before making the output visible.
        validate_bundle(annotations)
        return finish_bundle(
            stage,
            output,
            {
                "repo_type": "dataset",
                "release_id": release_id,
                "manifest_digest": source["manifest_digest"],
                "counts": {ds: source["counts"][ds] for ds in selected},
                "datasets": selected,
                "package_config": package_config,
                "quality_status": quality["status"],
                "audio_included": True,
                "annotations_bundle_sha256": source["bundle_sha256"],
            },
        )
