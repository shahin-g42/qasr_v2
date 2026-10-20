"""Bounded producer/consumer scheduling, exact coverage and resumable task outputs."""
import importlib.metadata
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

from .common import (
    CATALOG,
    DATASETS,
    atomic_json,
    exclusive,
    fingerprint,
    sha256,
    write_rows,
)
from .data import official_rows, prepared_rows
from .scoring import aggregate


def checkpoint_identity(path, backend="gqasr"):
    if backend in ("hf-qwen", "cohere", "whisper", "jais"):
        from .public_backend import checkpoint_identity as public_identity
        return public_identity(path, backend)
    if backend == "qasr":
        from .qasr_backend import checkpoint_identity as qasr_identity
        return qasr_identity(path)
    path = Path(path)
    required = [path / name for name in ("config.json", "preprocessor_config.json", "model.safetensors")]
    required += sorted((path / "tokenizer").glob("*"))
    if len(required) == 3:
        raise ValueError(f"No tokenizer found in {path}; use the checkpoint's artifact directory")
    return {str(p.relative_to(path)): sha256(p) for p in required if p.is_file() or p in required[:3]}


def implementation_identity():
    package = Path(__file__).parent
    files = [*package.glob("*.py"), package.parent / "modeling.py", package.parent / "processing.py"]
    return {str(p.relative_to(package.parent)): sha256(p) for p in sorted(files)}


def backend_implementation_identity(backend):
    result = implementation_identity()
    if backend == "qasr":
        import importlib.util
        spec = importlib.util.find_spec("qasr")
        if spec is None or not spec.origin:
            raise ValueError("QASR source unavailable; use the QASR benchmark launcher with repo/src on PYTHONPATH")
        source = Path(spec.origin).parent
        result.update({f"qasr/{p.relative_to(source)}": sha256(p) for p in sorted(source.rglob("*.py"))})
    return result


def validate_predictions(expected, found, mode):
    reference = {r["id"]: r for r in expected}
    if len(reference) != len(expected):
        raise ValueError("Duplicate expected IDs")
    result = {}
    for row in found:
        key = row["id"]
        if key not in reference or key in result or row["mode"] != mode:
            raise ValueError(f"Duplicate, unexpected or wrong-mode prediction: {key}")
        for field in ("dataset", "text", "relative_audio", "duration", "ordinal", "audio_sha256"):
            if row[field] != reference[key][field]:
                raise ValueError(f"Prediction identity mismatch: {key}/{field}")
        if not isinstance(row["pred_text"], str) or not row.get("audio_sha256"):
            raise ValueError(f"Malformed prediction: {key}")
        result[key] = row
    missing = reference.keys() - result.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} predictions for {mode}: {sorted(missing)[:5]}")
    return [result[r["id"]] for r in expected]


def completed_tasks(output, run_id):
    records = []
    for path in sorted((Path(output) / "tasks").glob("*.json")):
        data = json.loads(path.read_text())
        if data["run_id"] != run_id:
            raise ValueError(f"Mismatched result provenance: {path}")
        records.append(data)
    return records


def report(output, expected, identity, elapsed, attempts):
    records = completed_tasks(output, fingerprint(identity))
    scores, diagnostics = {}, {}
    published = json.loads(Path(__file__).with_name("published_results.json").read_text())
    counts_by_dataset = {ds: sum(r["dataset"] == ds for r in expected) for ds in DATASETS}
    full_suite = all(counts_by_dataset[ds] == CATALOG["datasets"][ds]["rows"] for ds in DATASETS)
    backend = identity.get("backend", "gqasr")
    label = backend.upper()
    lines = [f"# {label} — Open Universal Arabic ASR benchmark", "",
             f"Protocol revision: `{CATALOG['revision']}`. Precision: `{identity['precision']}`.",
             "", "Scores are fractions in JSON and percentages below. ELM CER includes spaces.",
             "Strict scores retain punctuation, diacritics and case; whitespace is canonicalized.",
             "Training overlap has not been audited: these results do not establish zero-shot generalization.",
             "The current Space lists MASC noisy-test as 14.9 h; the pinned official manifest/paper totals 8.92 h. "
             "This run follows the checksum-pinned repository manifest and flags that version difference.", ""]
    streamed_sources = sorted({(r.get("source", {}).get("repository"), r.get("source", {}).get("revision"))
                               for r in expected if r.get("source", {}).get("repository")})
    if streamed_sources:
        lines += ["Test audio was read from pinned remote sources, including public mirrors. "
                  "Official clip IDs/segment boundaries and durations are checked; byte equivalence "
                  "to separately obtained original-provider files has not been independently audited.", ""]
    if not full_suite:
        lines += ["**PARTIAL / SMOKE RUN — no complete leaderboard average or rank.**", ""]
    for mode in identity["modes"]:
        found = [r for task in records for r in task["predictions"] if r["mode"] == mode]
        rows = validate_predictions(expected, found, mode)
        write_rows(Path(output) / f"predictions-{mode}.jsonl", rows)
        scores[mode] = aggregate(rows, full_suite=full_suite)
        diagnostics[mode] = {
            "utterances": len(rows), "generation_limit_hits": sum(r.get("generation_limit_hit") is True for r in rows),
            "empty_outputs": sum(not r["pred_text"].strip() for r in rows),
            "generated_tokens": sum(r.get("generated_tokens") or 0 for r in rows),
            "generation_limit_unobservable": sum(r.get("generation_limit_hit") is None for r in rows),
            "generated_tokens_unobservable": sum(r.get("generated_tokens") is None for r in rows),
            "language_header_mismatch": sum(
                r.get("predicted_language") not in (None, "Arabic") for r in rows
            ) if mode == "auto" else None,
        }
        lines += [f"## {mode} decoding", "", "| Dataset | N | ELM WER | ELM CER | Strict WER | Strict CER |",
                  "|---|---:|---:|---:|---:|---:|"]
        for ds in DATASETS:
            if f"elm/{ds}" not in scores[mode]:
                continue
            a, b = scores[mode][f"elm/{ds}"], scores[mode][f"strict/{ds}"]
            values = ["undefined" if v is None else f"{v:.2%}" for v in (a["wer"], a["cer"], b["wer"], b["cer"])]
            lines.append(f"| {ds} | {a['utterances']} | " + " | ".join(values) + " |")
        macro = scores[mode]["elm/macro"]
        if macro["full_suite"] and macro["wer"] is not None and macro["cer"] is not None:
            lines += ["", f"**Six-set macro average: WER {macro['wer']:.2%}, CER {macro['cer']:.2%}.**"]
        elif macro["full_suite"]:
            lines += ["", "**Six-set aggregate is undefined because at least one reference denominator is zero.**"]
        lines += ["", f"Generation diagnostics: `{diagnostics[mode]}`", ""]
    last_attempt = attempts[-1] if attempts else {}
    published_columns = published["percentages"]
    model_rows = {name: index for index, name in enumerate(published_columns["Model"])}
    comparison_names = [published_columns["Model"][0],
                        "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"]
    comparison = {"space_revision": published["revision"], "retrieved": published["retrieved"],
                  "metrics_percent": {name: {
                      "average_wer": published_columns["Average WER⬇️"][model_rows[name]],
                      "average_cer": published_columns["Average CER"][model_rows[name]],
                  } for name in comparison_names if name in model_rows},
                  backend: {}}
    comparable = full_suite and all(
        values["elm/macro"]["wer"] is not None and values["elm/macro"]["cer"] is not None
        for values in scores.values()
    )
    comparison["complete_pinned_suite_scored"] = comparable
    comparison["comparable_to_leaderboard"] = False
    comparison["unresolved_requirements"] = [
        "MASC noisy manifest duration differs from the current leaderboard listing",
        "Training overlap has not been audited",
        "Model-specific decoding policies differ between submissions",
    ]
    if streamed_sources:
        comparison["streamed_audio_sources"] = [dict(repository=repo, revision=rev) for repo, rev in streamed_sources]
        comparison["unresolved_requirements"].append("Remote mirror audio byte equivalence to original-provider files is unaudited")
    if comparable:
        comparison[backend] = {mode: {"average_wer": 100 * values["elm/macro"]["wer"],
                                      "average_cer": 100 * values["elm/macro"]["cer"]}
                               for mode, values in scores.items()}
        lines += ["## Context from the live leaderboard snapshot", "",
                  f"Reference snapshot: Space revision `{published['revision']}` (retrieved {published['retrieved']}).",
                  "This table provides context only; the pinned MASC noisy manifest duration differs from the "
                  "current Space listing, and training overlap has not been audited.", "",
                  "| Model / mode | Average WER | Average CER |", "|---|---:|---:|"]
        for mode, result in comparison[backend].items():
            lines.append(f"| {label} ({mode}) | {result['average_wer']:.2f}% | {result['average_cer']:.2f}% |")
        for name, result in comparison["metrics_percent"].items():
            lines.append(f"| {name} (published) | {result['average_wer']:.2f}% | {result['average_cer']:.2f}% |")
    else:
        comparison["partial_set_means"] = {
            mode: {"tested_datasets": values["elm/macro"]["datasets"],
                   "mean_wer": (100 * values["elm/macro"]["wer"]
                                if values["elm/macro"]["wer"] is not None else None),
                   "mean_cer": (100 * values["elm/macro"]["cer"]
                                if values["elm/macro"]["cer"] is not None else None)}
            for mode, values in scores.items()
        }
        lines += ["## Leaderboard comparison", ""]
        if full_suite:
            lines.append("All six sets are present, but a reference denominator is zero, so an aggregate "
                         "cannot be compared with the published leaderboard average.")
        else:
            lines.append("This is a partial run. Its tested-set mean is reported for diagnostics only; it is not "
                         "comparable to the six-set leaderboard average, so published model averages are omitted.")

    performance = {"gpus": last_attempt.get("gpus", []),
                   "workers_per_gpu": last_attempt.get("workers_per_gpu"),
                   "total_workers": last_attempt.get("actual_workers", 0),
                   "ready_workers": last_attempt.get("ready_workers", len(last_attempt.get("workers", []))),
                   "start_policy": last_attempt.get("start_policy", "all_ready"),
                   "simultaneous_model_loads": last_attempt.get("simultaneous_model_loads"),
                   "first_ready_seconds": last_attempt.get("first_ready_seconds"),
                   "all_ready_seconds": last_attempt.get("all_ready_seconds"),
                   "processing_includes_worker_startup": last_attempt.get("processing_includes_worker_startup", False),
                   "batch_size": last_attempt.get("batch_size"),
                   "attempts": attempts,
                   "last_attempt_wall_seconds": elapsed or last_attempt.get("wall_seconds", 0),
                   "last_attempt_inference_seconds": last_attempt.get("processing_seconds"),
                   "total_attempt_wall_seconds": sum(a["wall_seconds"] for a in attempts),
                   "sum_worker_inference_seconds": sum(r["stats"]["inference_seconds"] for r in records),
                   "audio_seconds_processed": sum(r["stats"]["audio_seconds"] for r in records),
                   "oom_batch_splits": sum(r["stats"]["oom_batch_splits"] for r in records),
                   "peak_worker_allocated_gib": max((r["stats"]["peak_allocated_gib"] for r in records), default=0),
                   "peak_worker_reserved_gib": max((r["stats"]["peak_reserved_gib"] for r in records), default=0)}
    output_count = sum(len(r["predictions"]) for r in records)
    batch_records = [b for r in records for b in r["stats"].get("inference_batches", [])]
    all_batches_measured = bool(records) and all("inference_batches" in r["stats"] for r in records)
    performance["completed_outputs"] = output_count
    performance["mean_batch_seconds"] = (sum(b["seconds"] for b in batch_records) / len(batch_records)
                                          if all_batches_measured and batch_records else None)
    performance["amortized_compute_seconds_per_output"] = (performance["sum_worker_inference_seconds"] / output_count
                                                            if output_count else None)
    total_wall = performance["total_attempt_wall_seconds"]
    performance["end_to_end_audio_seconds_per_wall_second"] = performance["audio_seconds_processed"] / total_wall if total_wall else None
    performance["outputs_per_wall_second"] = output_count / total_wall if total_wall else None
    lines += ["## Performance", "", "```json", json.dumps(performance, indent=2), "```", "",
              "Mean batch time measures synchronized worker inference calls, including OOM retries. Amortized compute per output is not request latency. "
              "Older task records without batch measurements report that value as null.\n\n"
              "This is concurrent H100-node throughput, including model startup and prepared-audio reads. "
              "It is not directly comparable to the paper's single A100, batch-one, warmed-up RTF.", "",
              "With as-ready startup, processing time begins when the first worker is released and overlaps "
              "remaining model loads. End-to-end wall time includes all startup. First/all-ready timings and "
              "the number of ready workers are recorded; unused loaders may be stopped once all tasks finish.", "",
              "Published comparison values are saved separately in published-reference.json, with their Space revision. "
              "No automatic leaderboard rank is claimed; verify source coverage and training overlap before submission."]
    atomic_json(Path(output) / "scores.json", {"protocol": identity, "full_suite": full_suite,
                                               "scores": scores, "diagnostics": diagnostics,
                                               "leaderboard_context": comparison, "performance": performance})
    atomic_json(Path(output) / "published-reference.json", json.loads(Path(__file__).with_name("published_results.json").read_text()))
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"COMPLETE: {Path(output) / 'REPORT.md'}", flush=True)


def run(cfg, *, datasets=DATASETS, gpus="0,1,2,3,4,5,6,7", workers_per_gpu=11,
        batch_size=1, limit_per_dataset=0, output=None, modes=None):
    gpu_ids = [gpu.strip() for gpu in gpus.split(",")]
    if len(set(gpu_ids)) != len(gpu_ids) or not all(gpu_ids):
        raise ValueError("GPU IDs must be unique and nonempty")
    if not 1 <= workers_per_gpu <= 16 or batch_size < 1 or limit_per_dataset < 0:
        raise ValueError("Use 1–16 workers per GPU, positive batch size and nonnegative sample limit")
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
        raise ValueError(f"Datasets must be unique members of {DATASETS}")
    output = Path(output or cfg["output_dir"]).resolve()
    with exclusive(output):
        return _run(cfg, datasets, gpu_ids, workers_per_gpu, batch_size, limit_per_dataset,
                    output, modes or cfg.get("modes", ["arabic", "auto"]))


def model_load_concurrency(value, slots):
    if value == "all":
        return slots
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("simultaneous_model_loads must be 'all' or a positive integer")
    try:
        count = int(value)
    except ValueError:
        raise ValueError("simultaneous_model_loads must be 'all' or a positive integer") from None
    if count < 1:
        raise ValueError("simultaneous_model_loads must be 'all' or a positive integer")
    return min(count, slots)


def _run(cfg, datasets, gpu_ids, workers_per_gpu, batch_size, limit, output, modes):
    from .worker import worker_main

    if not modes or len(set(modes)) != len(modes) or set(modes) - {"auto", "arabic"}:
        raise ValueError("Modes must be unique: auto and/or arabic")
    task_size = int(cfg.get("task_size", 16))
    if task_size < 1:
        raise ValueError("task_size must be positive")
    # Workers batch within one task, so a smaller task would silently cap the
    # requested inference batch (e.g. task_size=4 with batch_size=16).
    task_size = max(task_size, batch_size)
    expected = official_rows(cfg["work_dir"], datasets)
    if limit:
        # Deterministic duration-stratified subset, including each set's longest clip.
        selected = []
        for ds in datasets:
            group = sorted((r for r in expected if r["dataset"] == ds), key=lambda r: (r["duration"], r["id"]))
            selected.extend(group[round(i * (len(group) - 1) / max(1, min(limit, len(group)) - 1))]
                            for i in range(min(limit, len(group))))
        expected = sorted(selected, key=lambda r: (r["dataset"], r["ordinal"]))
    prepared = prepared_rows(cfg, datasets)
    prepared_by_id = {row["id"]: row for row in prepared}
    if len(prepared_by_id) != len(prepared):
        raise ValueError("Prepared manifests contain duplicate sample IDs")
    missing_prepared = {row["id"] for row in expected} - prepared_by_id.keys()
    if missing_prepared:
        raise ValueError(f"Prepared manifests omit {len(missing_prepared)} selected official samples")
    expected = [prepared_by_id[row["id"]] for row in expected]
    backend = cfg.get("backend", "gqasr")
    if backend not in ("gqasr", "qasr", "hf-qwen", "cohere", "whisper", "jais"):
        raise ValueError("Unsupported Arabic benchmark backend")
    identities = checkpoint_identity(cfg["checkpoint"], backend)
    identity = {"version": 1, "backend": backend, "catalog": CATALOG, "checkpoint": identities,
                "selection": fingerprint(expected), "modes": modes, "precision": cfg.get("precision", "bf16"),
                "max_new_tokens": cfg.get("max_new_tokens", 768), "batch_size": batch_size,
                "task_size": task_size, "input_mode": "prepared",
                "runtime": {"python": sys.version.split()[0],
                            "torch": importlib.metadata.version("torch"),
                            "transformers": importlib.metadata.version("transformers"),
                            "rapidfuzz": importlib.metadata.version("rapidfuzz")},
                "implementation": backend_implementation_identity(backend)}
    if cfg.get("public_model"):
        identity["public_model"] = cfg["public_model"]
    run_id = fingerprint(identity)
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Output belongs to a different checkpoint/selection/protocol/code. Choose a new output directory.")
    atomic_json(path, identity)
    write_rows(output / "expected.jsonl", expected)
    records = completed_tasks(output, run_id)
    done = set()
    expected_by_id = {r["id"]: r for r in expected}
    for task in records:
        for row in task["predictions"]:
            key = (row["id"], row["mode"])
            if key in done or row["id"] not in expected_by_id or row["mode"] not in modes:
                raise ValueError("Duplicate or foreign prediction in existing tasks")
            if any(row[k] != expected_by_id[row["id"]][k] for k in expected_by_id[row["id"]]):
                raise ValueError(f"Changed prediction identity: {row['id']}")
            done.add(key)
    needed = {r["id"] for r in expected if any((r["id"], mode) not in done for mode in modes)}
    attempts_path = output / "attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    if not needed:
        report(output, expected, identity, 0, attempts)
        atomic_json(output / "progress.json", {
            "event": "benchmark_complete", "ready_workers": 0, "workers": 0,
            "completed": len(done), "total": len(expected) * len(modes), "percent": 100.0,
            "reused_completed_run": True,
        })
        return
    ctx = mp.get_context("spawn")
    task_count = (len(needed) + task_size - 1) // task_size
    slots = min(len(gpu_ids) * workers_per_gpu, task_count)
    if slots < 1:
        raise ValueError("No inference tasks were selected")
    load_concurrency = model_load_concurrency(cfg.get("simultaneous_model_loads", 4), slots)
    start_when_ready = cfg.get("start_when_ready", False)
    if not isinstance(start_when_ready, bool):
        raise ValueError("start_when_ready must be a boolean")
    jobs, events = ctx.Queue(maxsize=max(2, slots * 2)), ctx.Queue()
    begin = ctx.Event()
    load_slots = ctx.Semaphore(load_concurrency)
    workers_on_gpus = [sum(slot % len(gpu_ids) == index for slot in range(slots))
                       for index in range(len(gpu_ids))]
    max_workers_on_gpu = max(workers_on_gpus)
    settings = {"backend": backend, "precision": identity["precision"], "max_new_tokens": identity["max_new_tokens"],
                "batch_size": batch_size, "memory_fraction": 0.90 / max_workers_on_gpu}
    if identity["precision"] not in ("bf16", "fp32") or identity["max_new_tokens"] < 1:
        raise ValueError("Invalid inference precision or token budget")
    children, stop, producer_errors = [], threading.Event(), []
    producer_done = threading.Event()
    emitted = set()
    started = time.monotonic()
    attempt = {"workers_per_gpu": workers_per_gpu, "gpus": gpu_ids, "actual_workers": slots,
               "batch_size": batch_size, "task_size": task_size,
               "start_policy": "as_ready" if start_when_ready else "all_ready",
               "simultaneous_model_loads": load_concurrency,
               "first_ready_seconds": None, "all_ready_seconds": None,
               "processing_includes_worker_startup": start_when_ready,
               "resumed_predictions": len(done), "status": "running", "wall_seconds": 0}
    attempts.append(attempt)
    atomic_json(attempts_path, attempts)
    print(json.dumps({"event": "benchmark_schedule", "requested_workers": len(gpu_ids) * workers_per_gpu,
                      "actual_workers": slots, "workers_per_gpu": workers_on_gpus,
                      "batch_size": batch_size, "task_size": task_size,
                      "start_policy": attempt["start_policy"],
                      "simultaneous_model_loads": load_concurrency,
                      "pending_samples": len(needed), "pending_tasks": task_count}), flush=True)
    task_ids = set()
    lock = threading.Lock()

    def put(task):
        while not stop.is_set():
            try:
                jobs.put(task, timeout=1)
                return
            except queue.Full:
                pass
        raise RuntimeError("Benchmark stopped")

    def produce():
        try:
            rows = [prepared_by_id[r["id"]] for r in expected if r["id"] in needed]
            rows.sort(key=lambda r: (-r["actual_duration"], r["id"]))
            batch = []

            def submit(part):
                modes_by_id = {
                    row["id"]: [mode for mode in modes if (row["id"], mode) not in done]
                    for row in part
                }
                active = [mode for mode in modes if any(mode in values for values in modes_by_id.values())]
                task_id = fingerprint({"rows": [r["id"] for r in part],
                                       "modes_by_id": modes_by_id})[:24]
                task = {"task_id": task_id, "modes": active, "modes_by_id": modes_by_id,
                        "rows": part, "run_id": run_id,
                        "output": str(output / "tasks" / f"{task_id}.json")}
                with lock:
                    task_ids.add(task_id)
                put(task)

            for row in rows:
                if row["id"] in emitted or row["id"] not in needed:
                    raise ValueError(f"Duplicate or unexpected prepared ID: {row['id']}")
                emitted.add(row["id"])
                batch.append(row)
                if len(batch) >= task_size:
                    submit(batch)
                    batch = []
            if batch:
                submit(batch)
            if emitted != needed:
                raise ValueError(f"Prepared source omitted {len(needed - emitted)} selected samples")
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            producer_done.set()

    producer = None
    ready, finished, ready_info = set(), set(), []
    success = False
    last = 0
    last_completion = None
    try:
        for slot in range(slots):
            gpu = gpu_ids[slot % len(gpu_ids)]
            child = ctx.Process(target=worker_main, args=(slot, gpu, cfg["checkpoint"], settings,
                                jobs, events, begin, load_slots, str(output / f"worker-{slot:03d}.log")))
            child.start()
            children.append(child)
        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        inference_started = None
        while True:
            if producer_errors:
                raise producer_errors[0]
            if any(p.exitcode is not None for p in children):
                raise RuntimeError(f"A GPU worker exited; inspect {output}/worker-*.log")
            try:
                event = events.get(timeout=1)
            except queue.Empty:
                event = None
            if event:
                if event["event"] == "worker_error":
                    raise RuntimeError(f"Worker {event['slot']} failed: {event['traceback']}")
                if event["event"] == "ready":
                    ready.add(event["slot"])
                    ready_info.append(event)
                    if attempt["first_ready_seconds"] is None:
                        attempt["first_ready_seconds"] = time.monotonic() - started
                    if len(ready) == slots:
                        attempt["all_ready_seconds"] = time.monotonic() - started
                    if not begin.is_set() and (start_when_ready or len(ready) == slots):
                        inference_started = time.monotonic()
                        last_completion = inference_started
                        begin.set()
                elif event["event"] == "task_complete":
                    last_completion = time.monotonic()
                    finished.add(event["task_id"])
                    result = json.loads((output / "tasks" / f"{event['task_id']}.json").read_text())
                    for row in result["predictions"]:
                        done.add((row["id"], row["mode"]))
            now = time.monotonic()
            if now - last >= 5:
                total = len(expected) * len(modes)
                progress = {"event": "benchmark_progress", "ready_workers": len(ready), "workers": slots,
                            "loading_workers": slots - len(ready),
                            "phase": ("loading" if not begin.is_set() else
                                      "processing_with_workers_loading" if len(ready) < slots else "processing"),
                            "completed": len(done), "total": total, "percent": round(100 * len(done) / total, 2),
                            "samples_scheduled": len(emitted), "tasks_scheduled": len(task_ids),
                            "tasks_incomplete": len(task_ids - finished),
                            "elapsed_seconds": round(now - started, 1)}
                print(json.dumps(progress), flush=True)
                atomic_json(output / "progress.json", progress)
                attempt["wall_seconds"] = now - started
                atomic_json(attempts_path, attempts)
                last = now
            with lock:
                all_finished = finished == task_ids
            if producer_done.is_set() and all_finished:
                success = True
                attempt["processing_seconds"] = now - inference_started if inference_started else 0
                attempt["new_predictions"] = len(done) - attempt["resumed_predictions"]
                break
            # Starting early must not disable the deadline for workers still loading.
            if len(ready) < slots and now - started > cfg.get("startup_timeout_seconds", 1800):
                raise TimeoutError(f"Model startup timeout ({len(ready)}/{slots} ready); inspect worker logs")
            if last_completion is not None and now - last_completion > cfg.get("progress_timeout_seconds", 1800):
                raise TimeoutError("No inference task completed within progress_timeout_seconds; "
                                   "inspect worker logs and rerun to resume completed tasks")
    finally:
        stop.set()
        if producer:
            producer.join(timeout=2)
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join()
        jobs.cancel_join_thread()
        jobs.close()
        events.close()
        attempt.update(wall_seconds=time.monotonic() - started, status="complete" if success else "interrupted",
                       workers=ready_info, ready_workers=len(ready))
        atomic_json(attempts_path, attempts)
    if checkpoint_identity(cfg["checkpoint"], backend) != identities:
        raise ValueError("Checkpoint changed during benchmarking; results cannot be published")
    if backend_implementation_identity(backend) != identity["implementation"]:
        raise ValueError("Model or benchmark code changed during the run; use a new output directory")
    report(output, expected, identity, attempt["wall_seconds"], attempts)
    final_progress = {"event": "benchmark_complete", "ready_workers": len(ready), "workers": slots,
                      "completed": len(done), "total": len(expected) * len(modes), "percent": 100.0,
                      "elapsed_seconds": round(attempt["wall_seconds"], 1)}
    atomic_json(output / "progress.json", final_progress)
    print(json.dumps(final_progress), flush=True)


def tune(cfg, gpus="0,1,2,3,4,5,6,7", output=None, limit=64, workers_per_gpu=11):
    """Measure batch-size headroom at the requested workers per GPU on all GPUs."""
    if len(gpus.split(",")) != 8:
        raise ValueError("Benchmark tuning uses all eight allocated GPUs")
    root = Path(output or f"{cfg['output_dir']}-tuning")
    candidates = []
    for batch in (1, 2, 4):
        destination = root / f"w{workers_per_gpu}-b{batch}"
        try:
            run(cfg, gpus=gpus, workers_per_gpu=workers_per_gpu, batch_size=batch,
                limit_per_dataset=limit, output=destination, modes=["arabic"])
            scores = json.loads((destination / "scores.json").read_text())
            splits = scores["performance"]["oom_batch_splits"]
            # Compare the complete same-sized subset even if a candidate was resumed.
            throughput = (scores["diagnostics"]["arabic"]["utterances"]
                          / max(scores["performance"]["total_attempt_wall_seconds"], 1e-9))
            candidates.append({"workers_per_gpu": workers_per_gpu, "batch_size": batch,
                               "samples_per_second": throughput, "oom_batch_splits": splits,
                               "status": "complete", "path": str(destination)})
        except RuntimeError as exc:
            candidates.append({"workers_per_gpu": workers_per_gpu, "batch_size": batch,
                               "status": "failed", "error": str(exc)})
        atomic_json(root / "tuning.json", candidates)
    successful = [r for r in candidates if r["status"] == "complete"]
    if not successful:
        raise RuntimeError("No tuning configuration completed")
    clean = [r for r in successful if r["oom_batch_splits"] == 0]
    best = max(clean or successful, key=lambda r: r["samples_per_second"])
    atomic_json(root / "recommended.json", best)
    print(json.dumps({"event": "tuning_recommendation", **best}), flush=True)
