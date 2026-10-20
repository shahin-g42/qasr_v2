"""Public ASR benchmark stages on the existing frozen Arabic evaluation cache."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from gqasr.arabic_benchmark.common import atomic_json
from gqasr.arabic_benchmark.data import prepared_rows
from gqasr.arabic_benchmark.public_backend import checkpoint_identity
from gqasr.arabic_benchmark.runner import run

from .core import load_plan, sha256
from .hub_transport import check_hub_transport
from .leaderboard import additional_predictions
from .leaderboard_inputs import annotation_drafts
from . import local_checkpoint

REGISTRY = json.loads(Path(__file__).with_name("public_models.json").read_text())
REGISTRY.update(local_checkpoint.ENTRIES)


def selected_models(value):
    names = value.split()
    if not names or len(set(names)) != len(names) or set(names) - set(REGISTRY):
        raise ValueError(f"PUBLIC_MODELS must list distinct names from: {' '.join(REGISTRY)}")
    return names


def download_model(name, cache, *, api=None, download=None):
    if name in local_checkpoint.ENTRIES:
        return local_checkpoint.checkpoint_metadata(local_checkpoint.checkpoint_path(name), name)
    from huggingface_hub import HfApi, snapshot_download

    entry = REGISTRY[name]
    root = Path(cache) / name
    source = root / "source.json"
    if source.exists():
        info = json.loads(source.read_text())
        if any(
            info.get(k) != entry[k]
            for k in ("repository", "backend", "display_name", "max_new_tokens")
        ):
            raise ValueError(
                "Cached public model metadata changed; select a new PUBLIC_MODEL_CACHE"
            )
        if entry.get("revision") and info.get("revision") != entry["revision"]:
            raise ValueError("Cached Jais revision differs from the reviewed release")
        if checkpoint_identity(root / "snapshot", entry["backend"]) != info["files"]:
            raise ValueError(f"Cached public checkpoint changed: {name}")
        return info
    api, download = api or HfApi(), download or snapshot_download
    revision = entry.get("revision") or api.model_info(entry["repository"]).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not return an immutable public checkpoint revision")
    snapshot = root / "snapshot"
    print(
        json.dumps(
            {
                "event": "public_model_download",
                "name": name,
                "repository": entry["repository"],
                "revision": revision,
                "snapshot": str(snapshot),
            }
        ),
        flush=True,
    )
    # Jais alone uses hash-verified custom code from an immutable reviewed revision.
    download(
        entry["repository"],
        revision=revision,
        local_dir=str(snapshot),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.jinja",
            "chat_templates/*",
            *(["configuration_jais_asr.py", "modeling_jais_asr.py", "processing_jais_asr.py", "pipeline_jais_asr.py", "language.py", "postprocess.py"] if entry["backend"] == "jais" else []),
        ],
        max_workers=8,
    )
    info = {
        **entry,
        "revision": revision,
        "files": checkpoint_identity(snapshot, entry["backend"]),
        "postprocessing": {
            "jais": "official tag/refusal parser; preserve repetitions; Arabic family hint; default diacritics=false, code_switch=false",
            "hf-qwen": "raw token parsing; preserve repetitions; strip protocol delimiters",
            "cohere": "native Arabic punctuation prompt; reassemble all native audio chunks",
            "whisper": "native HF pipeline; 30s chunks with 5s overlap; token-cap diagnostics unavailable",
        }[entry["backend"]],
    }
    atomic_json(source, info)
    return info


def bind_prepared(cfg, rows, *, check_audio=False):
    actual = prepared_rows(cfg)
    expected = {r["id"]: r for r in rows}
    seen = set()
    for row in actual:
        key = row["id"]
        if key not in expected or key in seen:
            raise ValueError("Prepared audio membership does not match the frozen reference plan")
        for field in (
            "dataset",
            "ordinal",
            "text",
            "audio_sha256",
            "preparation_id",
            "actual_duration",
        ):
            if row[field] != expected[key][field]:
                raise ValueError(f"Prepared reference identity differs: {key}/{field}")
        if check_audio and sha256(row["audio"]) != row["audio_sha256"]:
            raise ValueError(f"Prepared waveform checksum differs: {key}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Prepared cache omits frozen reference-plan clips")
    return actual


def check_runtime(gpus):
    import torch
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.19.0"):
        raise ValueError(
            "Public baselines need native Qwen/Cohere support. Use the cluster benchmark environment with transformers==5.19.0; do not use the older training environment."
        )
    for name in (
        "Qwen3ASRForConditionalGeneration",
        "CohereAsrForConditionalGeneration",
        "WhisperForConditionalGeneration",
    ):
        getattr(transformers, name)
    ids = [s.strip() for s in gpus.split(",")]
    if not ids or len(set(ids)) != len(ids) or not all(s.isdigit() for s in ids):
        raise ValueError("PUBLIC_GPUS must contain distinct CUDA device ordinals")
    if not torch.cuda.is_available() or max(map(int, ids)) >= torch.cuda.device_count():
        raise ValueError(
            "Requested CUDA GPUs are not visible; run this stage on the allocated cluster node"
        )
    return {
        "transformers": transformers.__version__,
        "torch": str(torch.__version__),
        "devices": [{"index": i, "name": torch.cuda.get_device_name(int(i))} for i in ids],
    }


def workers_per_gpu(name):
    value = int(os.environ.get("PUBLIC_JAIS_WORKERS_PER_GPU", "4") if name == "jais-asr" else os.environ.get("PUBLIC_WORKERS_PER_GPU", "8"))
    if value < 1:
        raise ValueError("Workers per GPU must be positive")
    return value


def model_config(name, cache, work, output):
    if name in local_checkpoint.ENTRIES:
        return local_checkpoint.model_config(work, output, name)
    source = Path(cache) / name / "source.json"
    if not source.is_file():
        raise ValueError(f"Model {name} is not downloaded; run the download stage first")
    info = json.loads(source.read_text())
    entry = REGISTRY[name]
    if any(
        info.get(k) != entry[k] for k in ("repository", "backend", "display_name", "max_new_tokens")
    ):
        raise ValueError(f"Public model registration differs: {name}")
    if entry.get("revision") and info.get("revision") != entry["revision"]:
        raise ValueError("Jais revision differs from the reviewed release")
    checkpoint = Path(cache) / name / "snapshot"
    if checkpoint_identity(checkpoint, entry["backend"]) != info["files"]:
        raise ValueError(f"Public checkpoint files changed: {name}")
    options = {}
    if entry["backend"] == "jais":
        value = os.environ.get("PUBLIC_JAIS_DIACRITICS", "0")
        if value not in ("0", "1"):
            raise ValueError("PUBLIC_JAIS_DIACRITICS must be 0 or 1")
        options = {"diacritics": value == "1", "code_switch": False}
        info["postprocessing"] = (
            "official tag/refusal parser; preserve repetitions; Arabic family hint; "
            f"diacritics={str(options['diacritics']).lower()}, code_switch=false"
        )
    return {
        "backend": entry["backend"],
        "checkpoint": str(checkpoint.resolve()),
        "work_dir": str(Path(work).resolve()),
        "output_dir": str(Path(output).resolve()),
        "precision": "bf16",
        "max_new_tokens": entry["max_new_tokens"],
        "modes": ["arabic"],
        "task_size": 16,
        "simultaneous_model_loads": os.environ.get("PUBLIC_MODEL_LOAD_CONCURRENCY", "all"),
        "start_when_ready": True,
        "startup_timeout_seconds": 1800,
        "progress_timeout_seconds": 1800,
        "public_model": {k: v for k, v in info.items() if k != "files"},
        "decoding_options": options,
    }


def export_submissions(root, names, plan, rows):
    root = Path(root)
    models = []
    for name in names:
        folder = root / name / "full"
        progress = json.loads((folder / "progress.json").read_text())
        if (
            progress.get("event") != "benchmark_complete"
            or progress.get("completed") != len(rows)
            or progress.get("total") != len(rows)
        ):
            raise ValueError(
                f"Public run {name} is incomplete; smoke/subset scores cannot enter the full leaderboard"
            )
        prediction = folder / "predictions-arabic.jsonl"
        additional_predictions(rows, [(name, prediction)])
        run_file = folder / "run.json"
        identity = json.loads(run_file.read_text())
        public = identity.get("benchmark_model", identity.get("public_model", {}))
        local = name in local_checkpoint.ENTRIES
        fields = ("source_type", "checkpoint_digest", "training_step", "backend", "display_name") if local else ("repository", "revision", "backend", "display_name")
        if local:
            local_checkpoint.validate_registration(public, identity, name)
        if identity.get("modes") != ["arabic"] or identity.get("backend") != REGISTRY[name]["backend"] or any(
            public.get(k) != REGISTRY[name][k]
            for k in (("backend", "display_name", "max_new_tokens") if local else ("repository", "backend", "display_name", "max_new_tokens"))
        ):
            raise ValueError(f"Public run {name} has unexpected model/decoding provenance")
        if REGISTRY[name].get("revision") and public.get("revision") != REGISTRY[name]["revision"]:
            raise ValueError("Submitted Jais revision differs from the reviewed checkpoint")
        models.append(
            {
                "name": name,
                "status": "complete",
                "mode": "arabic",
                "predictions": prediction.relative_to(root).as_posix(),
                "predictions_sha256": sha256(prediction),
                "run_file": run_file.relative_to(root).as_posix(),
                "run_file_sha256": sha256(run_file),
                **{k: public[k] for k in fields},
            }
        )
    path = root / "submissions.json"
    atomic_json(
        path, {"schema_version": 1, "manifest_digest": plan["manifest_digest"], "models": models}
    )
    return path


def write_comparison_report(output):
    root = Path(output)
    data = json.loads((root / "leaderboard.json").read_text())
    panel = next(p for p in data["panels"] if p["id"] == "enhanced-full")
    lines = [
        "# Arabic ASR · enhanced-reference benchmark",
        "",
        f"Matched clips: **{panel['scored_clips']:,}**. Excluded: **{panel['excluded_clips']}**.",
        "References are frozen candidate-assisted ideal_text_draft values, not audio-verified gold.",
        ("Only fresh Arabic-mode runs are ranked; historical outputs remain reference-curation provenance."
         if data.get("ranking_source") == "fresh_submissions_only" else
         "Historical QASR/GQASR predictions are preserved. Each additional checkpoint is scored from its own completed Arabic-mode run."),
        "All paired original and enhanced scores below use exactly the same eligible clips.",
        "",
        "| Model | Original WER | Enhanced WER | Original CER | Enhanced CER |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in data["models"]:
        values = {
            r["track"]: r for r in panel["scores"] if r["model"] == name and r["dataset"] == "macro"
        }
        original, ideal = values["original_elm"], values["draft_elm"]
        label = data.get("model_details", {}).get(name, {}).get("display_name", name.upper())
        metrics = [original["wer"], ideal["wer"], original["cer"], ideal["cer"]]
        lines.append(
            f"| {label} | "
            + " | ".join("unavailable" if v is None else f"{100 * v:.2f}%" for v in metrics)
            + " |"
        )
    timing_note = (
        "Fresh source runs use the shared comparison protocol. Native token limits and output formats are recorded per model."
        if data.get("comparison_protocol") else
        "Full source runs; inspect each model's configuration and decoding modes."
    )
    lines += ["", "## Timing", "", timing_note + " Compute/output is amortized, not request latency.", "",
              "| Model | Mean batch (s) | Compute/output (s) | Outputs/wall second |", "|---|---:|---:|---:|"]
    for name in data["models"]:
        timing = data.get("performance", {}).get(name, {})
        values = [timing.get(k) for k in ("mean_batch_seconds", "amortized_compute_seconds_per_output", "outputs_per_wall_second")]
        lines.append(f"| {name} | " + " | ".join("unavailable" if v is None else f"{v:.3f}" for v in values) + " |")
    # Keep review/report artifacts outside the immutable upload bundle.
    destination = root.parent / f"{root.name}-comparison.md"
    destination.write_text("\n".join(lines) + "\n")
    print("Comparison report:", destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "transport",
            "download",
            "check",
            "smoke",
            "run",
            "rescore",
            "score",
            "publish",
            "all",
        ],
        nargs="?",
        default="all",
    )
    args = parser.parse_args(argv)
    if args.stage in {"transport", "download", "check", "publish", "all"}:
        print(json.dumps({"event": "public_hub_transport", **check_hub_transport()}), flush=True)
    if args.stage == "transport":
        return
    plan_root = Path(os.environ["ARABIC_REF_PLAN"])
    plan, rows = load_plan(plan_root)
    annotations = Path(os.environ["ARABIC_REF_ANNOTATIONS_BUNDLE"])
    annotation_drafts(annotations, plan, rows)
    if len(rows) != sum(plan["input_counts"].values()) or not all(
        "full" in r["cohorts"] for r in rows
    ):
        raise ValueError("This comparison requires the complete frozen six-dataset plan")
    names = selected_models(os.environ["PUBLIC_MODELS"])
    cache, root = Path(os.environ["PUBLIC_MODEL_CACHE"]), Path(os.environ["PUBLIC_BENCH_ROOT"])
    work = os.environ["PUBLIC_PREPARED_WORK"]
    gpus = os.environ["PUBLIC_GPUS"]
    print(
        json.dumps(
            {
                "event": "public_benchmark_configuration",
                "models": names,
                "prepared_cache": work,
                "run_root": str(root),
                "gpus": gpus,
                "workers_per_gpu": int(os.environ["PUBLIC_WORKERS_PER_GPU"]),
                "batch_size": int(os.environ["PUBLIC_BATCH_SIZE"]),
                "model_workers_per_gpu": {name: workers_per_gpu(name) for name in names},
                "model_load_concurrency": os.environ.get("PUBLIC_MODEL_LOAD_CONCURRENCY", "all"),
                "start_policy": "as_ready",
                "smoke_start_policy": "all_ready" if os.environ.get("UNIFORM_BENCH_ROOT") else "as_ready",
                "reference": "ideal_text_draft; paired original scores; speculative drafts excluded",
            },
            indent=2,
        )
    )
    if args.stage in {"check", "smoke", "run", "all"}:
        print(json.dumps({"event": "public_runtime", **check_runtime(gpus)}))
        bind_prepared({"work_dir": work}, rows, check_audio=True)
    if args.stage in {"download", "all"}:
        for name in names:
            info = download_model(name, cache)
            print(
                json.dumps(
                    {
                        "event": "public_model_pinned",
                        "name": name,
                        "repository": info.get("repository"),
                        "revision": info.get("revision"),
                        "checkpoint_digest": info.get("checkpoint_digest"),
                    }
                )
            )
    if args.stage in {"check", "smoke", "run", "all"}:
        for name in names:
            model_config(name, cache, work, root / name / "full")
        print(json.dumps({"event": "public_preflight_passed", "audio_hashes_checked": len(rows)}))
    if args.stage in {"smoke", "run", "all"}:
        for name in names:
            for phase in (
                ["smoke", "full"]
                if args.stage == "all"
                else ["full" if args.stage == "run" else "smoke"]
            ):
                output = root / name / phase
                cfg = model_config(name, cache, work, output)
                if os.environ.get("UNIFORM_BENCH_ROOT"):
                    cfg["require_idle_gpus"] = True
                    # A short smoke may finish before slower replicas load. Hold
                    # its work until all replicas occupy their memory share.
                    cfg["start_when_ready"] = phase != "smoke"
                print(
                    json.dumps({"event": "public_model_start", "name": name, "phase": phase}),
                    flush=True,
                )
                run(
                    cfg,
                    gpus=gpus,
                    workers_per_gpu=workers_per_gpu(name),
                    batch_size=int(os.environ["PUBLIC_BATCH_SIZE"]),
                    limit_per_dataset=int(os.environ.get("PUBLIC_SMOKE_PER_DATASET", "32")) if phase == "smoke" else 0,
                    modes=["arabic"],
                )
    if args.stage in {"score", "publish", "all"}:
        submissions = export_submissions(root, names, plan, rows)
        os.environ["ARABIC_REF_SUBMISSIONS"] = str(submissions)
    if args.stage in {"rescore", "score", "publish", "all"}:
        from .cluster_leaderboard import main as board_main

        if args.stage == "rescore":
            os.environ.pop("ARABIC_REF_SUBMISSIONS", None)
        board_main(["all" if args.stage in {"publish", "all"} else "build"])
        write_comparison_report(os.environ["ARABIC_REF_SPACE_BUNDLE"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
