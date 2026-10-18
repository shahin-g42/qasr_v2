"""Command line for the pinned, six-dataset Arabic ASR benchmark."""
import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .common import DATASETS, load_config


def dataset_list(value):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or set(result) - set(DATASETS):
        raise argparse.ArgumentTypeError(f"Expected unique dataset names from: {', '.join(DATASETS)}")
    return result


def mode_list(value):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or set(result) - {"arabic", "auto"}:
        raise argparse.ArgumentTypeError("Modes must be unique values from: arabic,auto")
    return result


def doctor():
    import torch
    import transformers

    expected = int(os.environ.get("GQASR_BENCH_EXPECTED_GPUS", "8"))
    visible = torch.cuda.device_count()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    result = {"host": platform.node(), "python": sys.version.split()[0], "cpu_cores_visible": cpus,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_runtime": torch.version.cuda, "visible_gpus": visible,
              "expected_gpus": expected, "devices": []}
    if torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "5.19.0":
        raise RuntimeError(f"Wrong unified environment: {result}")
    if visible != expected:
        raise RuntimeError(f"Expected {expected} allocated GPUs, PyTorch sees {visible}: {result}")
    for index in range(visible):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            bf16 = torch.cuda.is_bf16_supported()
        result["devices"].append({"index": index, "name": props.name,
                                  "memory_gib": round(props.total_memory / 2**30, 2),
                                  "bf16": bf16})
    if not all(device["bf16"] for device in result["devices"]):
        raise RuntimeError(f"BF16 is required for the configured H100 benchmark: {result}")
    import importlib.util

    required_modules = ("rapidfuzz", "soundfile", "scipy", "pyarrow", "huggingface_hub")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    result["missing_modules"] = missing
    result["ffmpeg"] = shutil.which("ffmpeg")
    if missing or result["ffmpeg"] is None:
        raise RuntimeError(f"Install the benchmark dependencies and ffmpeg: {result}")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel GQASR Arabic leaderboard benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the single env, allocated GPUs and CPU affinity")
    p = sub.add_parser("source-smoke", help="stream small public test samples and verify audio locally; no GPU")
    p.add_argument("--output", required=True)
    p.add_argument("--per-dialect", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("references", help="download and verify the pinned official test manifests")
    p.add_argument("--config", required=True)
    p = sub.add_parser("download-casablanca", help="download only the pinned Casablanca test audio")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("preflight", help="check checkpoint, manifests and every test-audio path")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", type=dataset_list, default=list(DATASETS))
    p = sub.add_parser("prepare", help="decode, resample and verify test audio using CPU workers")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", type=dataset_list, default=list(DATASETS))
    p.add_argument("--workers", type=int, default=96)
    p = sub.add_parser("run", help="run inference with persistent model replicas and audio batches")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", type=dataset_list, default=list(DATASETS))
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers-per-gpu", type=int, default=11)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit-per-dataset", type=int, default=0,
                   help="deterministic smoke subset; zero runs every official utterance")
    p.add_argument("--modes", type=mode_list, default=["arabic", "auto"])
    p.add_argument("--output")
    p = sub.add_parser("tune", help="compare batch sizes at 11 workers/GPU across all eight GPUs")
    p.add_argument("--config", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--limit-per-dataset", type=int, default=64)
    p.add_argument("--output")

    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
        return
    if args.command == "source-smoke":
        from .source_smoke import source_smoke
        source_smoke(args.output, args.per_dialect, args.workers)
        return
    cfg = load_config(args.config)
    if args.command == "references":
        from .data import fetch_references
        fetch_references(cfg["work_dir"])
        return
    if args.command == "download-casablanca":
        from .data import download_casablanca, fetch_references
        fetch_references(cfg["work_dir"])
        download_casablanca(cfg, workers=args.workers)
        return
    if args.command == "preflight":
        from .data import check_data, fetch_references
        from .runner import checkpoint_identity

        fetch_references(cfg["work_dir"])
        weights = checkpoint_identity(cfg["checkpoint"])
        result = check_data(cfg, args.datasets)
        result["checkpoint"] = cfg["checkpoint"]
        result["checkpoint_files"] = len(weights)
        result["benchmark_datasets"] = list(args.datasets)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if not result["ready"]:
            raise SystemExit(2)
        return
    if args.command == "prepare":
        from .data import fetch_references, prepare
        fetch_references(cfg["work_dir"])
        prepare(cfg, args.datasets, workers=args.workers)
        return
    if args.command == "run":
        from .runner import run
        run(cfg, datasets=args.datasets, gpus=args.gpus, workers_per_gpu=args.workers_per_gpu,
            batch_size=args.batch_size, limit_per_dataset=args.limit_per_dataset,
            output=args.output, modes=args.modes)
        return
    from .runner import tune
    tune(cfg, gpus=args.gpus, output=args.output or str(Path(cfg["output_dir"]).with_name("tuning")),
         limit=args.limit_per_dataset)


if __name__ == "__main__":
    main()
