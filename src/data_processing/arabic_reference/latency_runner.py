"""Serial, isolated batch-one ASR latency profiling on an allocated CUDA GPU."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from gqasr.arabic_benchmark.common import exclusive
from gqasr.arabic_benchmark.runner import backend_implementation_identity, checkpoint_identity

from . import local_checkpoint
from .core import digest, load_plan, read_jsonl, sha256, write_json, write_jsonl
from .latency import make_protocol, request_schedule, source_binding, summarize_profile
from .public_benchmark import REGISTRY, bind_prepared


def smi(query, gpu, *, apps=False):
    key = "--query-compute-apps=" if apps else "--query-gpu="
    value = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", key + ",".join(query), "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    return [
        dict(zip(query, row, strict=True))
        for row in csv.reader(io.StringIO(value), skipinitialspace=True)
        if row
    ]


def idle_gpu(gpu, allowed_pid=None):
    processes = smi(["pid", "gpu_uuid"], gpu, apps=True)
    foreign = [p for p in processes if p["pid"] != str(allowed_pid)]
    if foreign:
        raise ValueError(
            f"Latency GPU has other compute processes: {foreign}. Reserve an idle GPU; no processes were stopped."
        )


def hardware(gpu):
    rows = smi(
        [
            "uuid",
            "name",
            "memory.total",
            "driver_version",
            "power.limit",
            "clocks.max.sm",
            "mig.mode.current",
        ],
        gpu,
    )
    if len(rows) != 1 or rows[0]["mig.mode.current"].lower() == "enabled":
        raise ValueError("Latency profiling requires one full GPU, with MIG disabled")
    return {
        "identity": rows[0],
        "host": platform.node(),
        "cpu": platform.processor(),
        "cpu_cores_visible": os.cpu_count(),
        "system": platform.platform(),
    }


def runtime():
    return {
        "python": sys.version.split()[0],
        **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        "cpu_threads": 1,
        "tf32": False,
        "cudnn_benchmark": False,
    }


def telemetry(gpu):
    return smi(
        [
            "temperature.gpu",
            "power.draw",
            "clocks.current.sm",
            "clocks.current.memory",
            "utilization.gpu",
        ],
        gpu,
    )[0]


def model_sources(plan, names, public_root):
    result = {}
    for name in names:
        if name in ("qasr", "gqasr"):
            source = next((s for s in plan["sources"] if s["name"] == name), None)
            if source is None or not source.get("run"):
                raise ValueError(f"Frozen plan lacks {name}")
            result[name] = source["run"]
        else:
            if name not in REGISTRY:
                raise ValueError(f"Unknown latency model: {name}")
            path = Path(public_root) / name / "full"
            progress = json.loads((path / "progress.json").read_text())
            if (
                progress.get("event") != "benchmark_complete"
                or progress.get("completed") != sum(plan["input_counts"].values())
                or progress.get("total") != sum(plan["input_counts"].values())
            ):
                raise ValueError(f"Accuracy benchmark is not complete: {name}")
            result[name] = json.loads((path / "run.json").read_text())
    if len(set(names)) != len(names) or not names:
        raise ValueError("LATENCY_MODELS must contain distinct model names")
    for name, source in result.items():
        if source["precision"] != "bf16" or "arabic" not in source["modes"]:
            raise ValueError(f"Latency requires a BF16 Arabic source run: {name}")
    return result


def checkpoint_path(name):
    if name in local_checkpoint.ENTRIES:
        return local_checkpoint.checkpoint_path(name)
    if name in ("qasr", "gqasr"):
        key = f"LATENCY_{name.upper()}_CHECKPOINT"
        value = os.environ.get(key)
        if not value:
            raise ValueError(
                f"Set {key} to the checkpoint used for the completed {name.upper()} benchmark"
            )
        return Path(value)
    return Path(os.environ["PUBLIC_MODEL_CACHE"]) / name / "snapshot"


def implementation(sources):
    here = Path(__file__).parent
    return {
        "adapters": {
            name: backend_implementation_identity(source.get("backend", "gqasr"))
            for name, source in sources.items()
        },
        "profiler": {name: sha256(here / name) for name in ("latency.py", "latency_runner.py")},
    }


def measure_request(recognizer, clip, wave, budget, *, clock=time.perf_counter):
    """One in-process request: resident waveform to decoded text, no batch division."""
    cuda = recognizer.torch.cuda
    cuda.synchronize()
    cuda.reset_peak_memory_stats()
    started = clock()
    try:
        features = recognizer.features_from_audio([wave])
        cuda.synchronize()
        prepared = clock()
        # Never use safe_infer: batch one has no splitting/retry policy to hide.
        values = recognizer.infer([clip], features, "arabic", budget)
        cuda.synchronize()
        finished = clock()
        if len(values) != 1 or not isinstance(values[0].get("pred_text"), str):
            raise ValueError("Expected one final transcript for a single request")
        result = values[0]
        return {
            "status": "ok",
            "request_seconds": finished - started,
            "input_preparation_seconds": prepared - started,
            "inference_decode_seconds": finished - prepared,
            "peak_allocated_gib": cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": cuda.max_memory_reserved() / 2**30,
            "generated_tokens": result.get("generated_tokens"),
            "generation_limit_hit": result.get("generation_limit_hit"),
            "refused": result.get("refused", False),
            "empty_output": not result["pred_text"].strip(),
            "output_sha256": digest(result["pred_text"]),
        }
    except Exception as exc:
        # Failed requests remain in the record; summaries explicitly show their count.
        # Synchronization can itself fail after a CUDA error, so do not mask the cause.
        with contextlib.suppress(Exception):
            cuda.synchronize()
        elapsed = clock() - started
        with contextlib.suppress(Exception):
            cuda.empty_cache()
        return {
            "status": "error",
            "request_seconds": elapsed,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
        }


def run_worker(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job["output"])
    manifest = json.loads(Path(job["manifest"]).read_text())
    name, gpu = job["name"], job["gpu"]
    idle_gpu(gpu)
    if (
        hardware(gpu)["identity"] != manifest["hardware"]["identity"]
        or runtime() != manifest["runtime"]
    ):
        raise ValueError("Latency hardware or runtime changed; choose a new LATENCY_ROOT")
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Latency worker must see exactly one CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Latency GPU must support BF16")
    protocol = manifest["protocol"]
    paths = {r["id"]: r for r in read_jsonl(job["audio_manifest"])}
    clips = {r["id"]: r for r in [*protocol["clips"], *protocol["warmup_clips"]]}
    waves = {}
    started = time.perf_counter()
    for key, clip in clips.items():
        path = paths[key]["audio"]
        if sha256(path) != clip["audio_sha256"]:
            raise ValueError(f"Latency audio changed: {key}")
        wave = np.load(path, allow_pickle=False)
        if (
            wave.dtype != np.float32
            or wave.ndim != 1
            or not len(wave)
            or not np.isfinite(wave).all()
            or abs(len(wave) / 16000 - clip["actual_duration"]) > 1 / 16000
        ):
            raise ValueError(f"Invalid latency waveform: {key}")
        waves[key] = wave
    preload_seconds = time.perf_counter() - started
    backend = manifest["models"][name]["backend"]

    def verify_checkpoint():
        if (
            digest(checkpoint_identity(job["checkpoint"], backend))
            != manifest["models"][name]["checkpoint_digest"]
        ):
            raise ValueError(f"Latency checkpoint changed before/during profiling: {name}")

    verify_checkpoint()
    if backend == "gqasr":
        from gqasr.arabic_benchmark.worker import Recognizer as cls
    elif backend == "qasr":
        from gqasr.arabic_benchmark.qasr_backend import QASRRecognizer as cls
    else:
        from gqasr.arabic_benchmark.public_backend import recognizer_class

        cls = recognizer_class(backend)
    before = telemetry(gpu)
    started = time.perf_counter()
    recognizer = cls(job["checkpoint"], "cuda:0", "bf16", protocol["gpu_memory_fraction"])
    if manifest["models"][name].get("decoding_options"):
        recognizer.configure_decoding(manifest["models"][name]["decoding_options"])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("Latency adapters must use exactly one CPU/inter-op thread")
    records = []
    schedule = list(request_schedule(protocol))
    record_file = output / job["records_file"]
    with record_file.open("x") as stream:
        for sequence, (phase, repeat, clip) in enumerate(schedule):
            # Poll outside timed boundaries. This detects shared-GPU activity at
            # request boundaries; scheduler-level reservation is still required.
            idle_gpu(gpu, os.getpid())
            sample = measure_request(
                recognizer, clip, waves[clip["id"]], manifest["models"][name]["max_new_tokens"]
            )
            record = {**clip, "sequence": sequence, "phase": phase, "repeat": repeat, **sample}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if phase != "measured" and sample["status"] != "ok":
                raise ValueError(f"{name} failed {phase}: {sample['error']}")
            if sequence % 10 == 0 or sequence == len(schedule) - 1:
                print(
                    json.dumps(
                        {
                            "event": "latency_progress",
                            "model": name,
                            "phase": phase,
                            "completed_requests": sequence + 1,
                            "total_requests": len(schedule),
                        }
                    ),
                    flush=True,
                )
        os.fsync(stream.fileno())
    idle_gpu(gpu, os.getpid())
    if hardware(gpu)["identity"] != manifest["hardware"]["identity"]:
        raise ValueError("Latency GPU configuration changed during measurement")
    verify_checkpoint()
    result = {
        "schema_version": 1,
        "status": "complete",
        "model": name,
        "profile_sha256": manifest["profile_sha256"],
        "source": manifest["models"][name],
        "runtime": runtime(),
        "hardware": manifest["hardware"],
        "telemetry_before": before,
        "telemetry_after": telemetry(gpu),
        "gpu_isolation": "compute-process checks at every request boundary; exclusive GPU allocation required",
        "cuda_version": torch.version.cuda,
        "model_load_seconds": load_seconds,
        "audio_preload_seconds": preload_seconds,
        "first_token_latency_seconds": None,
        "first_token_latency_note": "Offline adapters return final text; streaming TTFT is not measured.",
        "records_file": record_file.name,
        "records_sha256": sha256(record_file),
        "summary": summarize_profile(records, protocol),
    }
    write_json(output / "report.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "run", "_worker"])
    parser.add_argument("--job")
    args = parser.parse_args(argv)
    if args.stage == "_worker":
        run_worker(args.job)
        return
    plan, rows = load_plan(os.environ["ARABIC_REF_PLAN"])
    names = os.environ["LATENCY_MODELS"].split()
    sources = model_sources(plan, names, os.environ["PUBLIC_BENCH_ROOT"])
    paths = {}
    # Hash model weights before timing. Model load time therefore is not a
    # guaranteed cold-disk measurement and is labelled accordingly.
    for name, source in sources.items():
        paths[name] = checkpoint_path(name).resolve()
        if checkpoint_identity(paths[name], source.get("backend", "gqasr")) != source["checkpoint"]:
            raise ValueError(f"Latency checkpoint differs from the accuracy benchmark: {name}")
        print(json.dumps({"event": "latency_checkpoint_verified", "model": name}), flush=True)
    prepared = bind_prepared({"work_dir": os.environ["PUBLIC_PREPARED_WORK"]}, rows)
    protocol = make_protocol(
        rows,
        plan["manifest_digest"],
        per_dataset=int(os.environ["LATENCY_PER_DATASET"]),
        repeats=int(os.environ["LATENCY_REPEATS"]),
        warmup=int(os.environ["LATENCY_WARMUP"]),
        seed=int(os.environ["LATENCY_SEED"]),
    )
    gpu = os.environ["LATENCY_GPU"]
    if not gpu.isdigit():
        raise ValueError("LATENCY_GPU must be one physical CUDA ordinal, such as 0")
    idle_gpu(gpu)
    manifest = {
        "protocol": protocol,
        "models": {n: source_binding(s) for n, s in sources.items()},
        "hardware": hardware(gpu),
        "runtime": runtime(),
        "implementation": implementation(sources),
    }
    manifest["profile_sha256"] = digest(manifest)
    root = Path(os.environ["LATENCY_ROOT"])
    selected_ids = {r["id"] for r in [*protocol["clips"], *protocol["warmup_clips"]]}
    selected_audio = [
        {"id": r["id"], "audio": r["audio"]} for r in prepared if r["id"] in selected_ids
    ]
    print(
        json.dumps(
            {
                "event": "latency_configuration",
                "models": names,
                "gpu": gpu,
                "clips": len(protocol["clips"]),
                "measured_requests_per_model": len(protocol["clips"]) * protocol["repeats"],
                "warmup": protocol["warmup_requests"],
                "batch_size": 1,
                "concurrency": 1,
                "output": str(root),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.stage == "check":
        for row in prepared:
            if row["id"] in selected_ids and sha256(row["audio"]) != row["audio_sha256"]:
                raise ValueError(f"Latency audio changed: {row['id']}")
        print("Latency preflight passed; no inference run.")
        return
    with exclusive(root):
        manifest_path = root / "profile.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Latency inputs, hardware, runtime or code changed; choose a new LATENCY_ROOT"
            )
        write_json(manifest_path, manifest)
        write_jsonl(root / "audio-paths.jsonl", selected_audio)
        for name in names:
            output = root / name
            output.mkdir(exist_ok=True)
            if (output / "report.json").exists():
                from .latency import read_latency_profiles

                read_latency_profiles(
                    root, plan, rows, {n: source_binding(s) for n, s in sources.items()}
                )
                print(json.dumps({"event": "latency_reused", "model": name}), flush=True)
                continue
            attempt = 1
            while (output / f"requests-{attempt:03d}.jsonl").exists():
                attempt += 1
            job = {
                "name": name,
                "gpu": gpu,
                "output": str(output.resolve()),
                "manifest": str(manifest_path.resolve()),
                "audio_manifest": str((root / "audio-paths.jsonl").resolve()),
                "checkpoint": str(paths[name]),
                "records_file": f"requests-{attempt:03d}.jsonl",
            }
            job_path = output / "job.json"
            write_json(job_path, job)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu,
                "HF_HUB_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
            # A fresh process per model gives it an empty CUDA allocator and avoids
            # stale model objects or repeated torch inter-op thread initialization.
            subprocess.run(
                [sys.executable, "-m", __spec__.name, "_worker", "--job", str(job_path)],
                env=env,
                check=True,
            )
        if implementation(sources) != manifest["implementation"]:
            raise ValueError("Profiler code changed during the run")
        from .latency import read_latency_profiles, write_latency_csv, write_request_csv

        report = read_latency_profiles(
            root, plan, rows, {n: source_binding(s) for n, s in sources.items()}
        )
        write_json(root / "latency.json", report)
        write_latency_csv(root / "latency.csv", report)
        write_request_csv(root / "latency-requests.csv", root, report)
        write_report(root / "REPORT.md", report)
        print("COMPLETE:", root / "REPORT.md", flush=True)


def write_report(path, profile):
    lines = [
        "# Arabic ASR single-request latency",
        "",
        "Batch 1 · concurrency 1 · one model on the same reserved GPU · BF16 · one CPU thread.",
        "Resident 16 kHz waveform to final text. No network, server queue or file I/O. Not streaming TTFT.",
        "Warm-up and first request are excluded from steady-state percentiles. Failed requests are counted separately; empty/refused/truncated outputs remain in successful-call timings.",
        "Model load follows checkpoint validation and may benefit from the OS file cache.",
        "Input preparation / inference breakdown follows adapter boundaries: public adapters perform acoustic preprocessing inside inference. Only total request latency is directly comparable.",
        "",
        f"Clips: {len(profile['protocol']['clips'])}; repeats: {profile['protocol']['repeats']}; warm-up: {profile['protocol']['warmup_requests']}.",
        "",
        "| Model | Mean s | Median s | p95 s | p99 s | RTF | Failed | Load s | First request s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in profile["models"].items():
        if report["status"] != "complete":
            continue
        s = report["summary"]["overall"]
        t = s["request_seconds"]
        values = [
            t["mean"],
            t["p50"],
            t["p95"],
            t["p99"],
            s["aggregate_rtf"],
            s["failed"],
            report["model_load_seconds"],
            report["summary"]["first_request_seconds"],
        ]
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join("N/A" if v is None else f"{v:.3f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "See latency.csv for dataset/duration breakdowns and latency.json for percentiles, memory, flags, hardware and provenance. Request records in each model directory retain all phases and repeated measurements.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
