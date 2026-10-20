"""Fresh, consistently configured accuracy runs, latency profiles and private publication."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .core import digest, write_json
from .public_benchmark import REGISTRY, selected_models


def validate_uniform_runs(root, names, *, gpus, workers, batch, diacritics, phase="full"):
    """Reject partial runs and configuration drift before scoring or publishing."""
    common = None
    bindings = {}
    for name in names:
        folder = Path(root) / name / phase
        run = json.loads((folder / "run.json").read_text())
        scores = json.loads((folder / "scores.json").read_text())
        progress = json.loads((folder / "progress.json").read_text())
        if scores["protocol"] != run or progress.get("event") != "benchmark_complete":
            raise ValueError(f"Incomplete or inconsistent uniform run: {name}")
        if progress.get("completed") != progress.get("total") or not progress.get("total"):
            raise ValueError(f"Incomplete uniform membership: {name}")
        signature = {
            k: run[k] for k in ("selection", "modes", "precision", "batch_size", "task_size", "runtime")
        }
        if common is None:
            common = signature
        if signature != common:
            raise ValueError(f"Uniform runtime, selection or batching differs: {name}")
        if run["modes"] != ["arabic"] or run["precision"] != "bf16" or run["batch_size"] != batch:
            raise ValueError(f"Uniform decoding policy differs: {name}")
        if run.get("gpu_isolation") != "checked_before_during_after":
            raise ValueError(f"Uniform run lacks GPU isolation checks: {name}")
        if run["backend"] != REGISTRY[name]["backend"] or run["max_new_tokens"] != REGISTRY[name]["max_new_tokens"]:
            raise ValueError(f"Unregistered backend or native token limit: {name}")
        expected_options = {"diacritics": diacritics, "code_switch": False} if run["backend"] == "jais" else {}
        if run.get("decoding_options", {}) != expected_options:
            raise ValueError(f"Uniform output format differs: {name}")
        performance = scores["performance"]
        if performance["oom_batch_splits"]:
            raise ValueError(
                f"{name} required OOM batch splits. Choose a smaller common batch/worker "
                "configuration and a new uniform run directory for a controlled throughput comparison."
            )
        attempts = performance["attempts"]
        if not attempts or any(
            a["gpus"] != gpus or a["workers_per_gpu"] != workers or a["batch_size"] != batch
            or a["actual_workers"] != workers * len(gpus)
            or a.get("ready_workers") != workers * len(gpus)
            or a.get("status") != "complete"
            or a.get("start_policy") != ("all_ready" if phase == "smoke" else "as_ready")
            for a in attempts
        ):
            raise ValueError(f"Uniform GPU/worker allocation differs: {name}")
        bindings[name] = digest(run)
    return {"common": common, "run_identities": bindings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "smoke", "run", "latency", "build", "publish", "all"])
    args = parser.parse_args(argv)
    names = selected_models(os.environ["PUBLIC_MODELS"])
    gpus = os.environ["PUBLIC_GPUS"].split(",")
    workers, batch = int(os.environ["PUBLIC_WORKERS_PER_GPU"]), int(os.environ["PUBLIC_BATCH_SIZE"])
    if workers != int(os.environ["PUBLIC_JAIS_WORKERS_PER_GPU"]):
        raise ValueError("Uniform runs require the same worker count for Jais and every other model")
    if os.environ.get("ARABIC_REF_SUBMISSIONS_ONLY") != "1":
        raise ValueError("Uniform releases must rank fresh submissions only")
    if os.environ["LATENCY_MODELS"].split() != names:
        raise ValueError("Uniform accuracy and latency must include the identical model list")
    value = os.environ["PUBLIC_JAIS_DIACRITICS"]
    if value not in ("0", "1"):
        raise ValueError("PUBLIC_JAIS_DIACRITICS must be 0 or 1")
    root = Path(os.environ["UNIFORM_BENCH_ROOT"])
    protocol = {
        "schema_version": 1,
        "models": names,
        "gpus": gpus,
        "workers_per_gpu": workers,
        "batch_size": batch,
        "smoke_per_dataset": int(os.environ["PUBLIC_SMOKE_PER_DATASET"]),
        "smoke_start_policy": "all_ready",
        "accuracy_start_policy": "as_ready",
        "jais_diacritics": value == "1",
        "precision": "bf16",
        "mode": "arabic",
        "decoding": "greedy; native saved frontends; preserve repetitions and empty outputs",
        "token_limits": {n: REGISTRY[n]["max_new_tokens"] for n in names},
        "token_limit_note": "768 generated tokens; Whisper uses its native 440-token per-chunk limit. Tokenizers differ.",
        "latency": {k: os.environ[k] for k in ("LATENCY_GPU", "LATENCY_PER_DATASET", "LATENCY_REPEATS", "LATENCY_WARMUP", "LATENCY_SEED")},
        "space_repo": os.environ["HF_SPACE_REPO"],
        "release_id": os.environ["ARABIC_REF_BOARD_RELEASE_ID"],
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Uniform protocol changed; select a new UNIFORM_RUN_ID")
    write_json(protocol_path, protocol)
    write_json(root / "status.json", {"stage": args.stage, "status": "running"})

    from .latency_runner import idle_gpu

    def stage(module, action):
        write_json(root / "status.json", {"stage": f"{module}/{action}", "status": "running"})
        subprocess.run(
            [sys.executable, "-m", f"data_processing.arabic_reference.{module}", action],
            check=True,
        )

    if args.stage in ("check", "all"):
        # Fail early on authentication/destination/dataset issues before GPU work.
        for gpu in gpus:
            idle_gpu(gpu)
        stage("cluster_leaderboard", "check")
        stage("public_benchmark", "check")
    if args.stage in ("smoke", "all"):
        stage("public_benchmark", "smoke")
        validate_uniform_runs(
            os.environ["PUBLIC_BENCH_ROOT"], names, gpus=gpus,
            workers=workers, batch=batch, diacritics=value == "1", phase="smoke",
        )
    if args.stage in ("run", "all"):
        stage("public_benchmark", "run")
    if args.stage in ("run", "latency", "build", "publish", "all"):
        verified = validate_uniform_runs(
            os.environ["PUBLIC_BENCH_ROOT"], names, gpus=gpus,
            workers=workers, batch=batch, diacritics=value == "1",
        )
        write_json(root / "accuracy-validation.json", verified)
    if args.stage in ("latency", "all"):
        stage("latency_runner", "run")
    if args.stage in ("build", "publish", "all"):
        # Public scoring imports profiles bound to these exact checkpoints/options
        # and validates every prediction, audio identity and timing task.
        stage("public_benchmark", "publish" if args.stage in ("publish", "all") else "score")
    write_json(root / "status.json", {"stage": args.stage, "status": "complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.environ.get("UNIFORM_BENCH_ROOT"):
            write_json(Path(os.environ["UNIFORM_BENCH_ROOT"]) / "status.json",
                       {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
