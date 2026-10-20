"""Frozen single-request latency protocol, statistics, and leaderboard imports.

All times are wall seconds. A request starts with a resident 16 kHz float waveform
and ends with the final decoded transcript and CUDA synchronization. Audio upload,
file reading, server queueing, and streaming first-token latency are not measured.
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

from .core import DATASETS, digest, read_jsonl, sha256

DURATION_BINS = ("0–5 s", "5–10 s", "10–20 s", "20+ s")
TIMING_FIELDS = ("request_seconds", "input_preparation_seconds", "inference_decode_seconds")


def duration_bin(duration):
    return DURATION_BINS[sum(duration >= edge for edge in (5, 10, 20))]


def select_clips(rows, per_dataset=100, seed=20261020):
    if per_dataset < 0:
        raise ValueError("LATENCY_PER_DATASET must be nonnegative; 0 selects the full suite")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate latency input IDs")
    selected = []
    for dataset in DATASETS:
        pool = sorted(
            (r for r in rows if r["dataset"] == dataset),
            key=lambda r: (r["actual_duration"], digest([seed, r["id"]])),
        )
        if not pool:
            raise ValueError(f"Latency selection is missing dataset {dataset}")
        n = min(per_dataset or len(pool), len(pool))
        # Midpoints of equal-sized duration strata cover the duration distribution
        # without selecting on a model's predictions, confidence, errors or speed.
        selected.extend(
            pool[min(len(pool) - 1, (2 * i + 1) * len(pool) // (2 * n))] for i in range(n)
        )
    return [
        {k: r[k] for k in ("id", "dataset", "audio_sha256", "actual_duration")} for r in selected
    ]


def make_protocol(rows, manifest_digest, *, per_dataset=100, repeats=3, warmup=12, seed=20261020):
    if repeats < 1 or warmup < 1:
        raise ValueError("Latency repeats and warm-up requests must be positive")
    clips = select_clips(rows, per_dataset, seed)
    warm = select_clips(rows, 2, seed)
    # Round-robin datasets for warm-up, including both short and long examples.
    groups = {ds: [r for r in warm if r["dataset"] == ds] for ds in DATASETS}
    warm = [groups[ds][i] for i in range(2) for ds in DATASETS if i < len(groups[ds])]
    return {
        "schema_version": 1,
        "manifest_digest": manifest_digest,
        "seed": seed,
        "per_dataset": per_dataset,
        "repeats": repeats,
        "warmup_requests": warmup,
        "batch_size": 1,
        "concurrency": 1,
        "replicas_per_gpu": 1,
        "gpu_memory_fraction": 0.9,
        "mode": "arabic",
        "precision": "bf16",
        "cpu_threads": 1,
        "clips": clips,
        "warmup_clips": warm,
        "selection": "duration-stratified midpoint sample per dataset; 0 means complete original membership",
        "measurement": "resident float32 16kHz mono waveform to final decoded text; synchronized CUDA wall time",
        "excluded": [
            "file I/O and checksum validation",
            "network upload",
            "server queue",
            "streaming TTFT",
        ],
        "percentile_method": "linear interpolation at (n-1)*p; request observations across repeated clips",
    }


def request_schedule(protocol):
    warm = protocol["warmup_clips"]
    yield "first_request", 0, warm[0]
    for i in range(protocol["warmup_requests"]):
        yield "warmup", i, warm[i % len(warm)]
    for repeat in range(protocol["repeats"]):
        clips = list(protocol["clips"])
        random.Random(protocol["seed"] + repeat).shuffle(clips)
        for clip in clips:
            yield "measured", repeat, clip


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {
            "n": 0,
            **dict.fromkeys(("mean", "min", "p50", "p90", "p95", "p99", "max", "stddev")),
        }
    if any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0
        for x in ordered
    ):
        raise ValueError("Latency measurements must be finite and nonnegative")

    def percentile(p):
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        **{f"p{p}": percentile(p / 100) for p in (50, 90, 95, 99)},
        "max": ordered[-1],
        "stddev": statistics.pstdev(ordered),
    }


def summarize_records(records):
    good = [r for r in records if r["status"] == "ok"]
    audio = sum(r["actual_duration"] for r in good)
    return {
        "attempted": len(records),
        "successful": len(good),
        "failed": len(records) - len(good),
        "unique_clips": len({r["id"] for r in records}),
        **{key: distribution([r[key] for r in good]) for key in TIMING_FIELDS},
        "rtf": distribution([r["request_seconds"] / r["actual_duration"] for r in good]),
        "aggregate_rtf": sum(r["request_seconds"] for r in good) / audio if audio else None,
        "peak_allocated_gib": max((r["peak_allocated_gib"] for r in good), default=None),
        "peak_reserved_gib": max((r["peak_reserved_gib"] for r in good), default=None),
        "generation_limit_hits": sum(r.get("generation_limit_hit") is True for r in good),
        "generation_limit_unobservable": sum(r.get("generation_limit_hit") is None for r in good),
        "empty_outputs": sum(r.get("empty_output", False) for r in good),
        "refusals": sum(r.get("refused", False) for r in good),
    }


def summarize_profile(records, protocol):
    schedule = list(request_schedule(protocol))
    if len(records) != len(schedule):
        raise ValueError("Incomplete latency request coverage")
    for index, (record, (phase, repeat, clip)) in enumerate(zip(records, schedule, strict=True)):
        if record["sequence"] != index or record["phase"] != phase or record["repeat"] != repeat:
            raise ValueError("Latency phase/order/repetition differs from frozen schedule")
        if any(record[k] != clip[k] for k in ("id", "dataset", "audio_sha256", "actual_duration")):
            raise ValueError("Latency clip identity differs")
        if not math.isfinite(clip["actual_duration"]) or clip["actual_duration"] <= 0:
            raise ValueError("Invalid latency audio duration")
        if record["status"] not in ("ok", "error"):
            raise ValueError("Invalid latency request status")
        distribution([record["request_seconds"]])
        if record["status"] == "ok":
            distribution(
                [record[k] for k in (*TIMING_FIELDS, "peak_allocated_gib", "peak_reserved_gib")]
            )
            if not math.isclose(
                record["request_seconds"],
                record["input_preparation_seconds"] + record["inference_decode_seconds"],
                abs_tol=1e-6,
            ):
                raise ValueError("Latency phase times do not sum to request time")
        elif phase != "measured":
            raise ValueError("First request and warm-up must succeed before measurement")
    measured = [r for r in records if r["phase"] == "measured"]
    return {
        "overall": summarize_records(measured),
        "by_dataset": {
            ds: summarize_records([r for r in measured if r["dataset"] == ds]) for ds in DATASETS
        },
        "by_duration": {
            group: summarize_records(
                [r for r in measured if duration_bin(r["actual_duration"]) == group]
            )
            for group in DURATION_BINS
        },
        "first_request_seconds": records[0]["request_seconds"],
        "warmup": summarize_records([r for r in records if r["phase"] == "warmup"]),
    }


def source_binding(run):
    return {
        "checkpoint_digest": digest(run["checkpoint"]),
        "backend": run.get("backend", "gqasr"),
        "precision": run["precision"],
        "max_new_tokens": run["max_new_tokens"],
        "public_model": run.get("public_model"),
        "benchmark_model": run.get("benchmark_model"),
    }


def read_latency_profiles(root, plan, rows, source_bindings):
    import json

    root = Path(root)
    manifest = json.loads((root / "profile.json").read_text())
    protocol = manifest["protocol"]
    expected = make_protocol(
        rows,
        plan["manifest_digest"],
        per_dataset=protocol["per_dataset"],
        repeats=protocol["repeats"],
        warmup=protocol["warmup_requests"],
        seed=protocol["seed"],
    )
    if protocol != expected or manifest["profile_sha256"] != digest(
        {k: v for k, v in manifest.items() if k != "profile_sha256"}
    ):
        raise ValueError("Latency protocol or frozen selection differs")
    results, hardware = {}, None
    for name, binding in manifest["models"].items():
        if name not in source_bindings or binding != source_bindings[name]:
            raise ValueError(
                f"Latency checkpoint/decoding source differs from ranked model: {name}"
            )
        report_path = root / name / "report.json"
        if not report_path.exists():
            results[name] = {
                "status": "unavailable",
                "reason": "Latency profile has not completed.",
            }
            continue
        report = json.loads(report_path.read_text())
        data_path = root / name / report["records_file"]
        if (
            Path(report["records_file"]).name != report["records_file"]
            or sha256(data_path) != report["records_sha256"]
        ):
            raise ValueError("Latency request records changed")
        if (
            report["profile_sha256"] != manifest["profile_sha256"]
            or report["model"] != name
            or report["source"] != binding
        ):
            raise ValueError("Latency report identity differs")
        if (
            report["runtime"] != manifest["runtime"]
            or report["hardware"]["identity"] != manifest["hardware"]["identity"]
        ):
            raise ValueError("Latency hardware/runtime differs from the profile")
        if hardware is not None and hardware != report["hardware"]["identity"]:
            raise ValueError("Latency models were not measured on the same GPU")
        hardware = report["hardware"]["identity"]
        summary = summarize_profile(list(read_jsonl(data_path)), protocol)
        if report["summary"] != summary or report["status"] != "complete":
            raise ValueError("Latency report summary or completion differs")
        distribution([report["model_load_seconds"], report["audio_preload_seconds"]])
        results[name] = {k: v for k, v in report.items() if k not in ("records_file",)}
        results[name]["report_sha256"] = sha256(report_path)
    return {
        "protocol": protocol,
        "profile_sha256": manifest["profile_sha256"],
        "profile_file_sha256": sha256(root / "profile.json"),
        "runtime": manifest["runtime"],
        "hardware": manifest["hardware"],
        "implementation": manifest["implementation"],
        "models": results,
    }


def write_latency_csv(path, profiles):
    fields = [
        "model",
        "group_type",
        "group",
        "attempted",
        "successful",
        "failed",
        "mean_seconds",
        "p50_seconds",
        "p90_seconds",
        "p95_seconds",
        "p99_seconds",
        "max_seconds",
        "aggregate_rtf",
        "peak_allocated_gib",
        "generation_limit_hits",
        "empty_outputs",
        "refusals",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, report in profiles.get("models", {}).items():
            if report["status"] != "complete":
                continue
            summary = report["summary"]
            groups = [("overall", "all", summary["overall"])]
            groups += [
                (kind, group, stats)
                for kind in ("by_dataset", "by_duration")
                for group, stats in summary[kind].items()
            ]
            for kind, group, stats in groups:
                writer.writerow(
                    {
                        "model": name,
                        "group_type": kind,
                        "group": group,
                        **stats,
                        **{
                            f"{k}_seconds": stats["request_seconds"][k]
                            for k in ("mean", "p50", "p90", "p95", "p99", "max")
                        },
                    }
                )


def write_request_csv(path, root, profiles):
    """Export validated observations, including cold/warm-up and failed requests."""
    import json

    fields = [
        "model",
        "sequence",
        "phase",
        "repeat",
        "id",
        "dataset",
        "actual_duration",
        "status",
        *TIMING_FIELDS,
        "peak_allocated_gib",
        "peak_reserved_gib",
        "generated_tokens",
        "generation_limit_hit",
        "empty_output",
        "refused",
        "output_sha256",
        "error",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, report in profiles["models"].items():
            if report["status"] != "complete":
                continue
            report_path = Path(root) / name / "report.json"
            if sha256(report_path) != report["report_sha256"]:
                raise ValueError("Latency report changed during export")
            data_path = report_path.parent / json.loads(report_path.read_text())["records_file"]
            before = sha256(data_path)
            if before != report["records_sha256"]:
                raise ValueError("Latency observations changed during export")
            for record in read_jsonl(data_path):
                writer.writerow({"model": name, **record})
            if sha256(data_path) != before:
                raise ValueError("Latency observations changed during export")
