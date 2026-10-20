"""Timing from completed, prediction-bound runs; never infer request latency from throughput."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from gqasr.arabic_benchmark.common import fingerprint

from .core import digest, sha256


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid benchmark timing")
    return value


def read_performance(root, predictions_sha256, *, expected_run=None):
    root = Path(root)
    if not (root / "scores.json").exists() or not (root / "tasks").is_dir():
        return {"status": "unavailable", "reason": "Timing report and task records were not supplied."}
    run = json.loads((root / "run.json").read_text())
    report = json.loads((root / "scores.json").read_text())
    if report["protocol"] != run or (expected_run is not None and run != expected_run):
        raise ValueError("Timing run identity does not match its prediction source")
    prediction_file = root / "predictions-arabic.jsonl"
    if sha256(prediction_file) != predictions_sha256:
        raise ValueError("Timing predictions differ from the scored predictions")
    predictions = [json.loads(line) for line in prediction_file.read_text().splitlines() if line.strip()]
    expected = {p["id"]: p for p in predictions}
    if not expected or len(expected) != len(predictions):
        raise ValueError("Timing prediction membership is empty or duplicated")
    modes = run["modes"]
    required = {(key, mode) for key in expected for mode in modes}
    seen, batch_seen = set(), set()
    inference, batch_times, task_hashes, measured_tasks, tasks = 0.0, [], {}, 0, 0
    for path in sorted((root / "tasks").glob("*.json")):
        task = json.loads(path.read_text())
        if task["run_id"] != fingerprint(run):
            raise ValueError("Timing task belongs to a different run")
        task_hashes[path.name] = sha256(path)
        task_keys = set()
        for p in task["predictions"]:
            key = (p["id"], p["mode"])
            if key not in required or key in seen:
                raise ValueError("Timing task membership is duplicated or unexpected")
            source = expected[p["id"]]
            if p["audio_sha256"] != source["audio_sha256"] or (p["mode"] == "arabic" and p["pred_text"] != source["pred_text"]):
                raise ValueError("Timing task predictions differ from scored outputs")
            seen.add(key)
            task_keys.add(key)
        stats = task["stats"]
        duration = seconds(stats["inference_seconds"])
        inference += duration
        tasks += 1
        batches = stats.get("inference_batches")
        if batches is not None:
            measured_tasks += 1
            local, local_times = set(), []
            for batch in batches:
                keys = [(key, batch["mode"]) for key in batch["ids"]]
                if not keys or len(set(keys)) != len(keys) or local.intersection(keys):
                    raise ValueError("Duplicate or empty batch timing membership")
                local.update(keys)
                local_times.append(seconds(batch["seconds"]))
            if local != task_keys or not math.isclose(sum(local_times), duration, rel_tol=1e-7, abs_tol=1e-6):
                raise ValueError("Batch timings do not cover the task inference")
            batch_seen.update(local)
            batch_times.extend(local_times)
    if seen != required:
        raise ValueError("Timing records do not cover the completed run")
    perf = report["performance"]
    if not math.isclose(inference, seconds(perf["sum_worker_inference_seconds"]), rel_tol=1e-7):
        raise ValueError("Timing report differs from its task records")
    attempts = perf.get("attempts", [])
    wall = sum(seconds(a["wall_seconds"]) for a in attempts)
    if not math.isclose(wall, seconds(perf["total_attempt_wall_seconds"]), rel_tol=1e-7):
        raise ValueError("Timing attempt totals differ")
    configurations = []
    for a in attempts:
        config = {
            "gpu_count": len(a["gpus"]), "workers_per_gpu": a["workers_per_gpu"],
            "workers": a["actual_workers"], "batch_size": a["batch_size"],
            "devices": sorted({w["device"] for w in a.get("workers", [])}),
            "start_policy": a.get("start_policy", "all_ready"),
        }
        if config not in configurations:
            configurations.append(config)
    complete_batches = measured_tasks == tasks and batch_seen == required
    return {
        "status": "available", "clips": len(expected), "outputs": len(seen), "modes": modes,
        "mean_batch_seconds": sum(batch_times) / len(batch_times) if complete_batches and batch_times else None,
        "batch_timing_status": "measured" if complete_batches else "not_recorded_for_every_task",
        "measured_batches": len(batch_times),
        "amortized_compute_seconds_per_output": inference / len(seen),
        "outputs_per_wall_second": len(seen) / wall if wall else None,
        "worker_inference_seconds": inference, "total_attempt_wall_seconds": wall,
        "attempt_count": len(attempts), "configurations": configurations,
        "precision": run["precision"], "oom_batch_splits": perf["oom_batch_splits"],
        "predictions_sha256": predictions_sha256, "run_sha256": sha256(root / "run.json"),
        "report_sha256": sha256(root / "scores.json"), "task_records_digest": digest(task_hashes),
    }


def performance_csv(path, models, performance):
    fields = ["model", "status", "clips", "outputs", "modes", "mean_batch_seconds",
              "batch_timing_status", "amortized_compute_seconds_per_output", "outputs_per_wall_second",
              "worker_inference_seconds", "total_attempt_wall_seconds", "attempt_count", "configurations"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name in models:
            row = {"model": name, **performance.get(name, {"status": "unavailable"})}
            for field in ("modes", "configurations"):
                if field in row:
                    row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)
