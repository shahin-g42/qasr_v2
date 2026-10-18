"""Pinned test membership and parallel, resumable audio preparation."""
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from .common import (
    CATALOG,
    DATASETS,
    atomic_json,
    exclusive,
    fingerprint,
    read_rows,
    sha256,
    write_rows,
)


def fetch_references(work_dir):
    destination = Path(work_dir) / "references"
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        ds, spec = item
        target = destination / spec["file"]
        if not target.exists():
            url = (f"https://raw.githubusercontent.com/{CATALOG['repository']}/"
                   f"{CATALOG['revision']}/datasets/{spec['file']}")
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            tmp = target.with_suffix(".download")
            tmp.write_bytes(data)
            if sha256(tmp) != spec["sha256"]:
                raise ValueError(f"Upstream checksum mismatch: {ds}")
            os.replace(tmp, target)
        if sha256(target) != spec["sha256"] or len(read_rows(target)) != spec["rows"]:
            raise ValueError(f"Invalid pinned reference manifest: {target}")
        return {"dataset": ds, **spec}

    with cf.ThreadPoolExecutor(6) as pool:
        for result in pool.map(fetch, CATALOG["datasets"].items()):
            print(json.dumps({"event": "references_ready", **result}), flush=True)
    atomic_json(destination / "catalog.json", CATALOG)


def official_rows(work_dir, datasets=DATASETS):
    rows = []
    for ds in datasets:
        spec = CATALOG["datasets"][ds]
        path = Path(work_dir) / "references" / spec["file"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}; run references")
        for ordinal, row in enumerate(read_rows(path)):
            relative = row["audio_filepath"].removeprefix("{data_folder}/")
            if Path(relative).is_absolute() or ".." in Path(relative).parts or "{" in relative:
                raise ValueError(f"Unsafe upstream path: {relative}")
            # Upstream IDs can denote speakers, not unique utterances; retain duplicate audio rows.
            metadata = dict(row.get("metadata") or {})
            for key in ("dialect", "gender", "age", "environment", "speaker_id"):
                if row.get(key) not in (None, ""):
                    metadata[key] = row[key]
            rows.append({"id": f"{ds}:{ordinal:06d}", "dataset": ds, "ordinal": ordinal,
                         "relative_audio": relative, "text": row["text"],
                         "duration": float(row["duration"]), "upstream_id": row.get("id"),
                         "metadata": metadata})
    return rows


def inventory(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files = {}
    for directory, _, names in os.walk(root):
        for name in names:
            if Path(name).suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".sph"):
                files.setdefault(name, []).append(str(Path(directory) / name))
    return files


def resolve_audio(row, root, files, overrides):
    name = row["relative_audio"]
    if name in overrides:
        source = dict(overrides[name])
        source["path"] = str((Path(root) / source["path"]).resolve())
        return source
    exact = Path(root) / name
    if exact.is_file():
        return {"path": str(exact.resolve())}
    for suffix in (Path(name).suffix, ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".sph"):
        candidates = files.get(Path(name).with_suffix(suffix).name, [])
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous audio basename {name}; provide an audio_map")
        if candidates:
            return {"path": str(Path(candidates[0]).resolve())}
    # Published SADA/MGB-2 segments encode boundaries. Support original recordings too.
    stem = Path(name).stem
    match = re.fullmatch(r"(.+)-seg_(\d+)_(\d+)-(\d+)_(\d+)", stem)
    offset = None
    if row["dataset"] == "sada" and match:
        recording, sec, millis, endsec, endmillis = match.groups()
        offset = float(f"{sec}.{millis}")
        duration = float(f"{endsec}.{endmillis}") - offset
    elif row["dataset"] == "mgb2" and (match := re.fullmatch(r"(.+)_spk-.+_seg-(\d+):(\d+)", stem)):
        recording, start, end = match.groups()
        offset, duration = int(start) / 100, (int(end) - int(start)) / 100
    if offset is not None:
        for suffix in (".wav", ".flac", ".sph"):
            candidates = files.get(recording + suffix, [])
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous original recording {recording}")
            if candidates:
                return {"path": str(Path(candidates[0]).resolve()), "offset_seconds": offset,
                        "duration_seconds": duration}
    raise FileNotFoundError(f"{row['id']}: missing {name} under {root}; supply clips or audio_map")


def decode(source):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    start = float(source.get("offset_seconds", 0))
    duration = source.get("duration_seconds")
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError("Invalid audio segment boundaries")
    try:
        with sf.SoundFile(source["path"]) as audio:
            sr = audio.samplerate
            audio.seek(round(start * sr))
            frames = -1 if duration is None else round(duration * sr)
            wave = audio.read(frames, dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError, OSError):
        # CV mp3 and other formats unavailable in the installed libsndfile.
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1"]
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", source["path"]]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
        wave = np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype="<f4")
        sr = 16000
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty or nonfinite audio: {source['path']}")
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // divisor, sr // divisor)
    return np.ascontiguousarray(wave, dtype=np.float32)


def prepare_one(task):
    import numpy as np

    row, source, metadata, destination, tolerance = task
    target = Path(destination) / "audio" / (row["id"].replace(":", "/") + ".npy")
    sidecar = target.with_suffix(".json")
    stat = Path(source["path"]).stat()
    identity = fingerprint(dict(row=row, source=source, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                metadata=metadata, tolerance=tolerance, preparation_version=1))
    if sidecar.exists() and target.exists():
        cached = json.loads(sidecar.read_text())
        if cached["preparation_id"] == identity and sha256(target) == cached["audio_sha256"]:
            return cached
    wave = decode(source)
    actual = len(wave) / 16000
    if abs(actual - row["duration"]) > max(tolerance, row["duration"] * 0.01):
        raise ValueError(f"{row['id']}: decoded {actual:.3f}s vs published {row['duration']:.3f}s; "
                         "check segmentation/version; no sample was trimmed or dropped")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".tmp").open("wb") as f:
        np.save(f, wave, allow_pickle=False)
    os.replace(target.with_suffix(".tmp"), target)
    result = {**row, "audio": str(target.resolve()), "audio_sha256": sha256(target),
              "actual_duration": actual, "source": source,
              "metadata": {**row.get("metadata", {}), **metadata},
              "preparation_id": identity}
    atomic_json(sidecar, result)
    return result


def _prepare_worker_init():
    # Keep 96 decoder processes from each starting their own BLAS/OpenMP pool.
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def check_data(cfg, datasets=DATASETS):
    """Resolve every official test audio path before spending time decoding audio."""
    work = Path(cfg["work_dir"])
    output, errors = {}, []
    for ds in datasets:
        spec = CATALOG["datasets"][ds]
        reference = work / "references" / spec["file"]
        if not reference.is_file() or sha256(reference) != spec["sha256"]:
            raise ValueError(f"Pinned references are missing or invalid; run references first: {reference}")
        entry = cfg["datasets"][ds]
        files = inventory(entry["root"])
        overrides = {r["relative_audio"]: {k: v for k, v in r.items() if k != "relative_audio"}
                     for r in read_rows(entry["audio_map"])} if entry.get("audio_map") else {}
        rows = official_rows(work, [ds])
        resolved, dataset_errors = 0, []
        for row in rows:
            try:
                source = resolve_audio(row, entry["root"], files, overrides)
                if not Path(source["path"]).is_file():
                    raise FileNotFoundError(source["path"])
                resolved += 1
            except (ValueError, FileNotFoundError) as exc:
                item = {"id": row["id"], "error": str(exc)}
                dataset_errors.append(item)
                errors.append(item)
        output[ds] = {"rows": len(rows), "resolved": resolved,
                      "hours": sum(r["duration"] for r in rows) / 3600,
                      "missing_or_ambiguous": len(dataset_errors)}
        print(json.dumps({"event": "data_preflight", "dataset": ds, **output[ds]}), flush=True)
        if dataset_errors:
            write_rows(Path(cfg["work_dir"]) / f"{ds}-audio-errors.jsonl", dataset_errors)
    return {"ready": not errors, "datasets": output, "errors": len(errors)}


def prepare(cfg, datasets=DATASETS, workers=96):
    if workers < 1:
        raise ValueError("Audio preparation workers must be positive")
    work = Path(cfg["work_dir"])
    destination = work / "prepared"
    with exclusive(destination):
        for ds in datasets:
            entry = cfg["datasets"][ds]
            files = inventory(entry["root"])
            overrides = {r["relative_audio"]: {k: v for k, v in r.items() if k != "relative_audio"}
                         for r in read_rows(entry["audio_map"])} if entry.get("audio_map") else {}
            metadata = {r["relative_audio"]: r["metadata"] for r in read_rows(entry["metadata"])} if entry.get("metadata") else {}
            rows = official_rows(work, [ds])
            tasks, errors = [], []
            for row in rows:
                try:
                    source = resolve_audio(row, entry["root"], files, overrides)
                    tasks.append((row, source, metadata.get(row["relative_audio"], {}), str(destination),
                                  cfg.get("duration_tolerance_seconds", 0.25)))
                except (ValueError, FileNotFoundError) as exc:
                    errors.append({"id": row["id"], "error": str(exc)})
            if errors:
                write_rows(destination / f"{ds}-errors.jsonl", errors)
                raise ValueError(f"{ds}: {len(errors)} unresolved audio files; see {destination}/{ds}-errors.jsonl")
            started = last = time.monotonic()
            results = []
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_prepare_worker_init) as pool:
                for result in pool.map(prepare_one, tasks, chunksize=8):
                    results.append(result)
                    now = time.monotonic()
                    if now - last > 5 or len(results) == len(rows):
                        print(json.dumps({"event": "prepare_progress", "dataset": ds, "done": len(results),
                                          "total": len(rows), "percent": round(100 * len(results) / len(rows), 2),
                                          "elapsed_seconds": round(now - started, 1)}), flush=True)
                        last = now
            write_rows(destination / f"{ds}.jsonl", results)
            atomic_json(destination / f"{ds}-provenance.json", {
                "upstream": CATALOG["datasets"][ds], "revision": CATALOG["revision"],
                "manifest_sha256": sha256(destination / f"{ds}.jsonl"), "rows": len(results),
                "actual_hours": sum(r["actual_duration"] for r in results) / 3600,
                "duplicate_audio_rows_retained": len(rows) - len({r["relative_audio"] for r in rows}),
            })


def prepared_rows(cfg, datasets=DATASETS):
    rows = []
    for ds in datasets:
        path = Path(cfg["work_dir"]) / "prepared" / f"{ds}.jsonl"
        provenance = json.loads(path.with_name(f"{ds}-provenance.json").read_text())
        if sha256(path) != provenance["manifest_sha256"]:
            raise ValueError(f"Changed prepared manifest: {path}")
        current = read_rows(path)
        original = official_rows(cfg["work_dir"], [ds])
        if len(current) != len(original):
            raise ValueError(f"Incomplete prepared dataset: {ds}")
        for a, b in zip(current, original, strict=True):
            # Preparation may enrich metadata (Casablanca dialect/gender, local maps).
            # Core membership, references, and paths must still match the pinned source.
            if any(a[k] != b[k] for k in b if k != "metadata"):
                raise ValueError(f"Changed official test record: {b['id']}")
        rows.extend(current)
    return rows


def download_casablanca(cfg, workers=8):
    """Download only test parquet shards; extract by seg_id, never substitute their text."""
    if workers < 1:
        raise ValueError("Casablanca download workers must be positive")
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    revision = "8951b1b88e28c1107142ced57967b8d16350951d"
    root = Path(cfg["datasets"]["casablanca"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    official = official_rows(cfg["work_dir"], ["casablanca"])
    expected = {Path(r["relative_audio"]).stem: r["relative_audio"] for r in official}
    if len(expected) != len(official):
        raise ValueError("Casablanca reference manifest has duplicate audio IDs")
    snapshot = snapshot_download("UBC-NLP/Casablanca", repo_type="dataset", revision=revision,
                                 allow_patterns=["*/test-*.parquet"], max_workers=workers)
    def extract(path):
        found, meta = [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                name = row["seg_id"]
                if name not in expected:
                    continue
                data = row["audio"]["bytes"]
                if not data:
                    raise ValueError(f"Missing embedded audio for {name}")
                target = root / f"{name}.wav"
                tmp = target.with_suffix(".download")
                tmp.write_bytes(data)
                os.replace(tmp, target)
                found.append(name)
                meta.append({"relative_audio": expected[name], "metadata": {
                    "dialect": path.parent.name, "gender": row.get("gender")}})
        return found, meta
    seen, metadata = set(), []
    with cf.ThreadPoolExecutor(workers) as pool:
        for found, meta in pool.map(extract, sorted(Path(snapshot).glob("*/test-*.parquet"))):
            if seen.intersection(found) or len(found) != len(set(found)):
                raise ValueError("Duplicate Casablanca seg_id across test shards")
            seen.update(found)
            metadata.extend(meta)
            print(json.dumps({"event": "casablanca_extract", "done": len(seen), "total": len(expected)}), flush=True)
    if seen != expected:
        raise ValueError(f"Casablanca snapshot is missing {len(expected - seen)} official test IDs")
    write_rows(root / "metadata.jsonl", metadata)
    atomic_json(root / "download-provenance.json", {"repository": "UBC-NLP/Casablanca", "revision": revision,
                                                   "split": "test", "matched": len(seen)})
