"""Range-read pinned test sources; retain only prepared evaluation waveforms.

No remote dataset scripts, snapshot downloads, or archive extraction. A single
coordinator checks exact membership before publishing the existing prepared format.
"""
import concurrent.futures as cf
import csv
import fnmatch
import io
import json
import os
import re
import tarfile
import tempfile
import threading
import time
from collections import defaultdict
from decimal import Decimal
from functools import partial
from pathlib import Path

from .common import CATALOG, DATASETS, atomic_json, exclusive, fingerprint, sha256, write_rows
from .data import decode, fetch_references, official_rows

SOURCES = json.loads(Path(__file__).with_name("stream_sources.json").read_text())


def remote_path(spec, path):
    return f"datasets/{spec['repository']}@{spec['revision']}/{path}"


def open_remote(spec, path, block_size=1024 * 1024):
    from huggingface_hub import HfFileSystem

    # Only bounded in-memory range caching; no Hugging Face snapshot on disk.
    return HfFileSystem().open(remote_path(spec, path), "rb", block_size=block_size,
                              cache_type="readahead")


def source_files(spec):
    from huggingface_hub import HfApi

    paths = sorted(p for p in HfApi().list_repo_files(
        spec["repository"], repo_type="dataset", revision=spec["revision"]
    ) if fnmatch.fnmatch(p, spec["pattern"]))
    if not paths:
        raise ValueError(f"No pinned test files found: {spec['repository']}/{spec['pattern']}")
    return paths


def _by_stem(rows):
    result = defaultdict(list)
    for row in rows:
        result[Path(row["relative_audio"]).stem].append(row)
    return result


def masc_mapping(rows, spec):
    """Use the original start timestamp, never text similarity or row ordering."""
    expected = defaultdict(list)
    for row in rows:
        video, start = Path(row["relative_audio"]).stem.rsplit("_", 1)
        expected[(video, Decimal(start), row["dataset"])].append(row)
    mapping = {"repository": spec["metadata_repository"], "revision": spec["metadata_revision"]}
    with open_remote(mapping, spec["metadata_path"]) as source:
        content = source.read().decode("utf-8-sig")
    found, identities = {}, {}
    for meta in csv.DictReader(io.StringIO(content)):
        if meta["type"] not in ("c", "n"):
            raise ValueError(f"Unknown MASC partition: {meta['type']}")
        ds = "masc_clean" if meta["type"] == "c" else "masc_noisy"
        key = (meta["video_id"], Decimal(meta["start"]), ds)
        if key not in expected:
            continue
        name = Path(meta["file_path"]).name
        for row in expected[key]:
            if row["text"] != meta["text"] or abs(row["duration"] - float(meta["duration"])) > .02:
                raise ValueError(f"MASC reference/metadata mismatch: {row['id']}")
        identity = (Decimal(meta["end"]), Decimal(meta["duration"]), meta["text"])
        if key in identities and identities[key] != identity:
            raise ValueError(f"Ambiguous MASC timestamp mapping: {name}")
        identities[key] = identity
        # The source duplicates one identical video/start/end/text segment under
        # two UUIDs. Select its stable canonical alias and retain BOTH official
        # evaluation rows; never count two distinct timestamps as one clip.
        found[key] = min(name, found.get(key, name))
    if found.keys() != expected.keys():
        raise ValueError(f"MASC mapping omits {len(expected.keys() - found.keys())} official clips")
    if len(set(found.values())) != len(found):
        raise ValueError("Ambiguous MASC UUID reused across distinct segments")
    return {name: expected[key] for key, name in found.items()}


def _source(spec, path, **extra):
    return {"repository": spec["repository"], "revision": spec["revision"],
            "remote_path": path, "kind": spec["kind"], **extra}


def _parquet_job(spec, path, mapping):
    import pyarrow.parquet as pq

    entries = []
    with open_remote(spec, path) as source:
        parquet = pq.ParquetFile(source, pre_buffer=False)
        column = spec["id_column"]
        columns = [column]
        if "gender" in parquet.schema_arrow.names:
            columns.append("gender")
        # Project identifiers only: audio.bytes pages are not fetched here.
        for group in range(parquet.num_row_groups):
            for index, record in enumerate(parquet.read_row_group(group, columns=columns).to_pylist()):
                key = Path(record["audio"]["path"]).name if column == "audio.path" else record[column]
                for row in mapping.get(key, []):
                    metadata = {}
                    if row["dataset"] == "casablanca":
                        metadata = {"dialect": Path(path).parent.name, "gender": record.get("gender")}
                    entries.append({"row": row, "metadata": metadata,
                                    "source": _source(spec, path, row_group=group, row_index=index,
                                                      remote_id=key)})
    return {"spec": spec, "path": path, "entries": entries}


def plan_sources(work_dir, datasets=DATASETS, workers=16):
    """Check exact remote metadata coverage; tar member coverage is checked while reading."""
    rows = official_rows(work_dir, datasets)
    jobs, statuses = [], []
    for name, spec in SOURCES.items():
        selected = [r for r in rows if r["dataset"] in
                    (("masc_clean", "masc_noisy") if name == "masc" else (name,))]
        if not selected:
            continue
        print(json.dumps({"event": "stream_source_open", "source": name,
                          "official_rows": len(selected)}), flush=True)
        try:
            paths = source_files(spec)
            current = []
            if spec["kind"] == "wav_segments":
                paths_by_name = defaultdict(list)
                for path in paths:
                    paths_by_name[Path(path).stem].append(path)
                grouped = defaultdict(list)
                for row in selected:
                    match = re.fullmatch(r"(.+)-seg_(\d+)_(\d+)-(\d+)_(\d+)",
                                         Path(row["relative_audio"]).stem)
                    if not match:
                        raise ValueError(f"Invalid SADA boundaries: {row['id']}")
                    recording, sec, frac, endsec, endfrac = match.groups()
                    candidates = paths_by_name[recording]
                    if len(candidates) != 1:
                        raise ValueError(f"Missing or ambiguous remote SADA recording: {recording}")
                    start, end = float(f"{sec}.{frac}"), float(f"{endsec}.{endfrac}")
                    if start < 0 or end <= start or abs(end - start - row["duration"]) > .002:
                        raise ValueError(f"Invalid SADA segment duration: {row['id']}")
                    path = candidates[0]
                    grouped[path].append({"row": row, "metadata": {}, "source": _source(
                        spec, path, offset_seconds=start, duration_seconds=end - start)})
                current = [{"spec": spec, "path": path, "entries": entries}
                           for path, entries in sorted(grouped.items())]
                with open_remote(spec, current[0]["path"]) as source:
                    source.read(64)
            elif spec["kind"] == "parquet":
                mapping = masc_mapping(selected, spec) if name == "masc" else _by_stem(selected)
                with cf.ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
                    futures = [pool.submit(partial(_parquet_job, spec, mapping=mapping), path) for path in paths]
                    for future in cf.as_completed(futures):
                        current.append(future.result())
                        print(json.dumps({"event": "stream_metadata_progress", "source": name,
                                          "files_done": len(current), "files_total": len(paths)}), flush=True)
                current.sort(key=lambda job: job["path"])
                found = [item["row"]["id"] for job in current for item in job["entries"]]
                wanted = {r["id"] for r in selected}
                if len(found) != len(set(found)):
                    raise ValueError(f"Duplicate remote test IDs: {name}")
                if set(found) != wanted:
                    raise ValueError(f"{name}: missing {len(wanted - set(found))} official test IDs")
            elif spec["kind"] == "tar":
                # This pinned Arabic test release has one uncompressed tar. Do
                # not guess membership across extra archives in a changed source.
                if len(paths) != 1:
                    raise ValueError("Expected one pinned Common Voice Arabic test archive")
                with open_remote(spec, paths[0]) as source:
                    source.read(512)  # Exercise authentication before any long preparation.
                current = [{"spec": spec, "path": paths[0], "entries": [
                    {"row": row, "metadata": {}, "source": _source(
                        spec, paths[0], remote_id=Path(row["relative_audio"]).stem)} for row in selected]}]
            else:
                raise ValueError(f"Unsupported remote source kind: {spec['kind']}")
            jobs.extend(job for job in current if job["entries"])
            status = {"source": name, "ready": True, "rows": len(selected),
                      "membership_from_metadata": spec["kind"] != "tar",
                      "repository": spec["repository"], "revision": spec["revision"]}
        except Exception as exc:
            # Do not emit request headers/tokens or signed download URLs.
            status = {"source": name, "ready": False, "error_type": type(exc).__name__,
                      "repository": spec["repository"], "revision": spec["revision"],
                      "action": f"Check access to https://huggingface.co/datasets/{spec['repository']} "
                                "and authenticate with hf auth login. Also check pinned metadata/IDs."}
            if isinstance(exc, ValueError):
                status["detail"] = str(exc)
        statuses.append(status)
        print(json.dumps({"event": "stream_source_check", **status}), flush=True)
    return rows, jobs, statuses


def check_sources(work_dir, datasets=DATASETS, workers=16):
    fetch_references(work_dir)
    _, _, statuses = plan_sources(work_dir, datasets, workers)
    result = {"ready": all(s["ready"] for s in statuses), "sources": statuses,
              "audio_content_validated": False, "model_weights_loaded": False}
    atomic_json(Path(work_dir) / "stream-sources-check.json", result)
    return result


def _paths(destination, row):
    target = Path(destination) / "audio" / (row["id"].replace(":", "/") + ".npy")
    return target, target.with_suffix(".json")


def _identity(item, tolerance):
    return fingerprint({**item, "tolerance": tolerance, "stream_preparation_version": 1})


def _cached(item, destination, tolerance):
    target, sidecar = _paths(destination, item["row"])
    if not sidecar.exists() or not target.exists():
        return None
    try:
        cached = json.loads(sidecar.read_text())
        if cached["preparation_id"] == _identity(item, tolerance) and sha256(target) == cached["audio_sha256"]:
            return cached
    except (ValueError, KeyError):
        pass
    return None


def _save(item, wave, destination, tolerance):
    import numpy as np

    row = item["row"]
    actual = len(wave) / 16000
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty, invalid or nonfinite streamed audio: {row['id']}")
    if abs(actual - row["duration"]) > max(tolerance, row["duration"] * .01):
        raise ValueError(f"{row['id']}: streamed duration {actual:.3f}s vs official {row['duration']:.3f}s")
    target, sidecar = _paths(destination, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".tmp").open("wb") as f:
        np.save(f, np.asarray(wave, dtype=np.float32), allow_pickle=False)
    os.replace(target.with_suffix(".tmp"), target)
    result = {**row, "audio": str(target.resolve()), "audio_sha256": sha256(target),
              "actual_duration": actual, "source": item["source"],
              "metadata": {**row.get("metadata", {}), **item["metadata"]},
              "preparation_id": _identity(item, tolerance)}
    atomic_json(sidecar, result)
    return result


def _decode_payload(payload):
    # The temporary compressed clip is deleted immediately; no original corpus
    # or archive is retained. Reuse production decoder including FFmpeg fallback.
    with tempfile.NamedTemporaryFile(suffix=".audio") as f:
        f.write(payload)
        f.flush()
        return decode({"path": f.name})


def _run_job(job, destination, tolerance, completed):
    import pyarrow.parquet as pq
    import soundfile as sf

    spec, path, entries = job["spec"], job["path"], job["entries"]
    receipt = Path(destination) / "stream-jobs" / (fingerprint(job) + ".json")
    receipt_valid = False
    if receipt.is_file():
        try:
            saved = json.loads(receipt.read_text())
            receipt_valid = saved["job_id"] == fingerprint(job) and (
                saved["complete_archive_checked"] or job.get("partial", False))
        except (ValueError, KeyError):
            pass
    pending = []
    for item in entries:
        cached = _cached(item, destination, tolerance)
        if cached is None:
            pending.append(item)
        else:
            completed(cached, True)
    if not pending and (spec["kind"] != "tar" or receipt_valid):
        return

    def save(item, wave):
        completed(_save(item, wave, destination, tolerance), False)

    if spec["kind"] == "wav_segments":
        with open_remote(spec, path) as source, sf.SoundFile(source) as audio:
            # The SADA originals are 16 kHz mono PCM; refuse transformed mirrors.
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError(f"Unexpected SADA recording format: {path}")
            for item in sorted(pending, key=lambda item: item["source"]["offset_seconds"]):
                locator = item["source"]
                audio.seek(round(locator["offset_seconds"] * 16000))
                wave = audio.read(round(locator["duration_seconds"] * 16000), dtype="float32")
                save(item, wave)
    elif spec["kind"] == "parquet":
        groups = defaultdict(lambda: defaultdict(list))
        for item in pending:
            loc = item["source"]
            groups[loc["row_group"]][loc["row_index"]].append(item)
        with open_remote(spec, path) as source:
            parquet = pq.ParquetFile(source, pre_buffer=False)
            for group, indexes in sorted(groups.items()):
                position = 0
                for batch in parquet.iter_batches(batch_size=8, row_groups=[group], columns=["audio"]):
                    for record in batch.to_pylist():
                        if position in indexes:
                            payload = record["audio"]["bytes"]
                            if not payload:
                                raise ValueError(f"Missing embedded audio: {path}, group {group}, row {position}")
                            wave = _decode_payload(payload)
                            for item in indexes[position]:
                                save(item, wave)
                        position += 1
    elif spec["kind"] == "tar":
        # Stream through the test tar once, including on partial resume. Never
        # extract paths supplied by archive entries to the local filesystem.
        wanted = defaultdict(list)
        for item in pending:
            wanted[item["source"]["remote_id"]].append(item)
        all_ids = {i["source"]["remote_id"] for i in entries}
        seen = set()
        with open_remote(spec, path) as source, tarfile.open(fileobj=source, mode="r|*") as archive:
            for member in archive:
                if Path(member.name).suffix.lower() not in (".mp3", ".wav", ".flac", ".ogg"):
                    continue
                key = Path(member.name).stem
                if key not in all_ids:
                    continue
                if key in seen or not member.isfile() or member.size > 128 * 1024 * 1024:
                    raise ValueError(f"Duplicate or invalid test audio member: {key}")
                seen.add(key)
                if key not in wanted:
                    if job.get("partial") and seen == all_ids:
                        break
                    continue
                with archive.extractfile(member) as f:
                    wave = _decode_payload(f.read())
                for item in wanted[key]:
                    save(item, wave)
                if job.get("partial") and seen == all_ids:
                    break
        if seen != all_ids:
            raise ValueError(f"Remote test archive missing {len(all_ids - seen)} official clips")
        # Only successful scans get a receipt: cached clips from a failed
        # duplicate/missing-member run must not bypass the archive coverage check.
        atomic_json(receipt, {"job_id": fingerprint(job), "matched": len(seen),
                              "complete_archive_checked": not job.get("partial", False)})
    else:
        raise ValueError(f"Unsupported remote source kind: {spec['kind']}")


def stream_prepare(work_dir, datasets=DATASETS, workers=16, limit_per_dataset=0):
    """Publish exactly the same prepared input format consumed by the 88 GPU workers."""
    if workers < 1 or limit_per_dataset < 0:
        raise ValueError("workers must be positive and sample limit nonnegative")
    work = Path(work_dir).resolve()
    smoke = limit_per_dataset > 0
    destination = work / ("stream-smoke/prepared" if smoke else "prepared")
    started = time.monotonic()
    with exclusive(destination):
        fetch_references(work)
        rows, jobs, statuses = plan_sources(work, datasets, workers)
        if not all(status["ready"] for status in statuses):
            atomic_json(work / "stream-sources-check.json", {"ready": False, "sources": statuses})
            raise RuntimeError("Remote source check failed; see stream-sources-check.json. "
                               "No dataset was silently omitted. Use --datasets to explicitly run a subset.")
        if smoke:
            selected = {row["id"] for ds in datasets for row in
                        [r for r in rows if r["dataset"] == ds][:limit_per_dataset]}
            rows = [r for r in rows if r["id"] in selected]
            jobs = [{**job, "partial": True, "entries": [i for i in job["entries"] if i["row"]["id"] in selected]}
                    for job in jobs]
            jobs = [job for job in jobs if job["entries"]]
        # Invalidate old final manifests before a changed/failed preparation can
        # leave a stale full dataset looking ready. Valid per-clip caches remain.
        if not smoke:
            for ds in datasets:
                (destination / f"{ds}-provenance.json").unlink(missing_ok=True)
        results, reused, last = {}, 0, 0.0
        lock = threading.Lock()

        def completed(result, cached):
            nonlocal reused, last
            with lock:
                if result["id"] in results:
                    raise ValueError(f"Repeated streamed audio: {result['id']}")
                results[result["id"]] = result
                reused += int(cached)
                now = time.monotonic()
                if now - last >= 5 or len(results) == len(rows):
                    print(json.dumps({"event": "stream_prepare_progress", "done": len(results),
                                      "total": len(rows), "cached": reused,
                                      "percent": round(100 * len(results) / len(rows), 2),
                                      "elapsed_seconds": round(now - started, 1)}), flush=True)
                    last = now

        # Network and per-clip decode concurrency is bounded independently of the
        # 88 inference replicas. No replica issues duplicate remote requests.
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_job, job, destination, .25, completed) for job in jobs]
            try:
                for future in cf.as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if results.keys() != {r["id"] for r in rows}:
            raise ValueError("Streamed results do not cover the selected official test IDs")
        counts = {}
        for ds in datasets:
            prepared = [results[r["id"]] for r in rows if r["dataset"] == ds]
            path = destination / f"{ds}.jsonl"
            write_rows(path, prepared)
            counts[ds] = len(prepared)
            atomic_json(destination / f"{ds}-provenance.json", {
                "upstream": CATALOG["datasets"][ds], "revision": CATALOG["revision"],
                "manifest_sha256": sha256(path), "rows": len(prepared),
                "actual_hours": sum(r["actual_duration"] for r in prepared) / 3600,
                "stream_sources": SOURCES, "complete_dataset": not smoke,
            })
        result = {"kind": "remote test audio preparation; no ASR inference", "counts": counts,
                  "samples": len(results), "reused": reused, "prepared_directory": str(destination),
                  "full_selected_datasets_prepared": not smoke,
                  "full_benchmark_validated": False, "model_weights_loaded": False,
                  "original_archives_saved": False, "sources": statuses,
                  "max_duration_error_seconds": max(abs(r["actual_duration"] - r["duration"]) for r in results.values()),
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        atomic_json(destination.parent / ("stream-smoke.json" if smoke else "stream-prepare.json"), result)
        return result
