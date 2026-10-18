"""Stream a small public sample through the production audio preparation path."""
import concurrent.futures as cf
import fnmatch
import importlib.metadata
import json
import time
from collections import defaultdict
from pathlib import Path

from .common import CATALOG, atomic_json, exclusive, write_rows
from .data import (
    CASABLANCA_REPOSITORY,
    CASABLANCA_REVISION,
    fetch_references,
    official_rows,
    prepare_one,
)


def stream_rows(paths):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for path in paths:
        remote = f"datasets/{CASABLANCA_REPOSITORY}@{CASABLANCA_REVISION}/{path}"
        # Range reads fetch parquet pages/row groups, not an entire snapshot.
        with fs.open(remote, "rb", block_size=4 * 1024 * 1024) as source:
            parquet = pq.ParquetFile(source, pre_buffer=False)
            for batch in parquet.iter_batches(batch_size=8, columns=["seg_id", "audio", "gender"]):
                for row in batch.to_pylist():
                    yield path, row


def source_smoke(output, per_dialect=2, workers=2):
    """No model and no score: check real streamed bytes, IDs, references and durations."""
    from huggingface_hub import HfApi

    if per_dialect < 1 or workers < 1:
        raise ValueError("per_dialect and workers must be positive")
    output = Path(output).resolve()
    with exclusive(output):
        started = time.monotonic()
        fetch_references(output)
        official = official_rows(output, ["casablanca"])
        expected = {Path(row["relative_audio"]).stem: row for row in official}
        if len(expected) != len(official):
            raise ValueError("Duplicate Casablanca reference IDs")
        files = HfApi().list_repo_files(CASABLANCA_REPOSITORY, repo_type="dataset",
                                      revision=CASABLANCA_REVISION)
        shards = defaultdict(list)
        for path in sorted(files):
            if fnmatch.fnmatch(path, "*/test-*.parquet"):
                shards[Path(path).parent.name].append(path)
        if len(shards) != 8:
            raise ValueError(f"Expected eight pinned Casablanca dialects, found {sorted(shards)}")
        (output / "raw").mkdir(exist_ok=True)

        def check_dialect(item):
            dialect, paths = item
            results, seen, scanned = [], set(), 0
            stream = stream_rows(paths)
            try:
                for shard, row in stream:
                    scanned += 1
                    name = row["seg_id"]
                    if name not in expected:
                        continue
                    if name in seen:
                        raise ValueError(f"Repeated streamed Casablanca ID: {name}")
                    seen.add(name)
                    reference = expected[name]
                    audio = row["audio"]["bytes"]
                    if not audio:
                        raise ValueError(f"Missing embedded audio: {name}")
                    path = output / "raw" / f"{reference['id'].replace(':', '_')}.wav"
                    path.write_bytes(audio)
                    prepared = prepare_one((reference, {"path": str(path)},
                                            {"dialect": dialect, "gender": row.get("gender")},
                                            str(output / "prepared"), .25))
                    results.append({**prepared, "source_shard": shard,
                                    "duration_error_seconds": prepared["actual_duration"] - reference["duration"]})
                    if len(results) >= per_dialect:
                        break
            finally:
                stream.close()
            if len(results) != per_dialect:
                raise ValueError(f"{dialect}: found only {len(results)} of {per_dialect} requested official test samples")
            return dialect, scanned, results

        rows, counts = [], {}
        with cf.ThreadPoolExecutor(min(workers, len(shards))) as pool:
            futures = [pool.submit(check_dialect, item) for item in sorted(shards.items())]
            for future in cf.as_completed(futures):
                dialect, scanned, samples = future.result()
                rows.extend(samples)
                counts[dialect] = {"checked": len(samples), "streamed_rows": scanned}
                print(json.dumps({"event": "source_smoke_progress", "dialect": dialect,
                                  "checked": len(rows), "total": len(shards) * per_dialect}), flush=True)
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate test IDs across dialects")
        result = {
            "kind": "public dataset streaming and audio preparation smoke test; no ASR inference",
            "repository": CASABLANCA_REPOSITORY, "revision": CASABLANCA_REVISION,
            "reference_revision": CATALOG["revision"], "dialects": counts, "samples": len(rows),
            "all_ids_match_official_references": True, "sample_rate": 16000,
            "all_audio_decoded_finite": True, "all_duration_checks_passed": True,
            "max_duration_error_seconds": max(abs(r["duration_error_seconds"]) for r in rows),
            "elapsed_seconds": round(time.monotonic() - started, 2), "model_weights_loaded": False,
            "full_benchmark_validated": False,
            "runtime": {name: importlib.metadata.version(name)
                        for name in ("huggingface_hub", "pyarrow", "numpy", "scipy", "soundfile")},
        }
        write_rows(output / "samples.jsonl", sorted(rows, key=lambda row: row["id"]))
        atomic_json(output / "source-smoke.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result
