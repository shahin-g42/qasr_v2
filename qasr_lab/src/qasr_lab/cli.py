"""Standalone command line interface; heavyweight imports stay inside commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import file_hash, fresh_directory, read_json, read_jsonl, write_json
from .panel import canonical_sample, load_panel, prepare_panel, save_panel
from .report import compare_reports, report_html, save_report, score_predictions


def import_legacy(path, output, mode="offline", model_id=None):
    source = read_json(path)
    try:
        results = source["modes"][mode]["results"]
    except KeyError as exc:
        raise ValueError(f"No results for legacy mode {mode}") from exc
    samples, predictions = [], []
    for row in results:
        sample = canonical_sample(
            row, base=Path(path).resolve().parent, language=source.get("language")
        )
        samples.append(sample)
        prediction = {
            "id": sample["id"],
            "hypothesis": row["hypothesis"],
            "status": "ok",
            "token_limit_observable": False,
        }
        for key in ("latency_seconds", "audio_seconds", "generated_tokens"):
            if key in row:
                prediction[key] = row[key]
        predictions.append(prediction)
    out = fresh_directory(output)
    panel = save_panel(
        samples,
        out / "panel",
        {
            "legacy_source": str(Path(path).resolve()),
            "sha256": file_hash(path),
            "original_manifest": source.get("manifest"),
            "leakage": {"status": "not_checked"},
            "warning": "Historical predictions only; held-out status and runtime provenance are unverified.",
        },
    )
    report = score_predictions(
        panel,
        samples,
        predictions,
        {
            "id": model_id or f"{Path(path).stem}:{mode}",
            "checkpoint": source.get("model"),
            "backend": f"imported:{mode}",
            "timing_protocol": None,
            "original_summary": source["modes"][mode].get("summary"),
        },
    )
    save_report(report, out / "report", predictions)
    return {"output": str(out), "samples": len(samples), "complete": report["complete"]}


def run_inference(config_path, panel_path, output):
    from .backends import HFBackend, evaluate_backend, validate_model_config

    config_path = Path(config_path).resolve()
    config = validate_model_config(read_json(config_path))
    for key in ("qasr_source", "checkpoint"):
        if config.get(key):
            value = Path(config[key]).expanduser()
            if (
                key == "qasr_source"
                or value.is_absolute()
                or str(value).startswith(".")
                or (config_path.parent / value).exists()
            ):
                config[key] = str((config_path.parent / value).resolve())
    panel, samples = load_panel(panel_path)
    out = fresh_directory(output)
    write_json(out / "config.json", config)
    try:
        backend = HFBackend(config)
        predictions = []
        with (out / "predictions.jsonl").open("x", encoding="utf-8") as journal:
            for prediction in evaluate_backend(backend, samples):
                predictions.append(prediction)
                journal.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                if len(predictions) % 10 == 0 or prediction["status"] != "ok":
                    print(
                        f"{len(predictions)}/{len(samples)}: {prediction['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
        backend.metadata["peak_allocated_gpu_gib"] = backend.peak_memory()
        report = score_predictions(panel, samples, predictions, backend.metadata)
        write_json(out / "report.json", report)
        (out / "report.html").write_text(report_html(report), encoding="utf-8")
        return {"output": str(out), "samples": len(samples), "complete": report["complete"]}
    except Exception as exc:
        write_json(out / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


def parser():
    root = argparse.ArgumentParser(
        description="Isolated QASR evaluation and adaptation experiments"
    )
    root.add_argument("--version", action="version", version="qasr-lab 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    panel = commands.add_parser("panel", help="Freeze a deterministic multilingual sample panel")
    panel.add_argument("--manifest", action="append", required=True)
    panel.add_argument("--output", required=True)
    panel.add_argument("--per-stratum", type=int, default=500)
    panel.add_argument("--seed", type=int, default=42)
    panel.add_argument("--language")
    panel.add_argument("--audio-root")
    panel.add_argument("--exclude-manifest", action="append", default=[])
    panel.add_argument("--require-audio", action="store_true")
    panel.add_argument(
        "--hash-audio",
        action="store_true",
        help="Hash selected audio bytes and verify them at inference",
    )
    run = commands.add_parser("run", help="Generate and score predictions with one checkpoint")
    run.add_argument("--model", required=True, help="Model JSON configuration")
    run.add_argument("--panel", required=True)
    run.add_argument("--output", required=True)
    score = commands.add_parser(
        "score", help="Score external predictions without model dependencies"
    )
    score.add_argument("--panel", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument("--output", required=True)
    compare = commands.add_parser("compare", help="Paired comparison, first report is the baseline")
    compare.add_argument("--report", action="append", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    legacy = commands.add_parser("import-legacy", help="Rescore existing eval_results JSON")
    legacy.add_argument("--input", required=True)
    legacy.add_argument("--mode", default="offline")
    legacy.add_argument("--model-id")
    legacy.add_argument("--output", required=True)
    audit = commands.add_parser(
        "audit-transfer", help="Compare converted tensors with their sources"
    )
    audit.add_argument("--target", required=True)
    audit.add_argument("--encoder")
    audit.add_argument("--decoder")
    audit.add_argument("--qasr-source")
    audit.add_argument("--output", required=True)
    plan = commands.add_parser(
        "plan", help="Write controlled pilot configurations without launching jobs"
    )
    plan.add_argument("--base", required=True)
    plan.add_argument("--non-speech-panel")
    plan.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Run one isolated, single-device pilot")
    train.add_argument("--config", required=True)
    source = commands.add_parser(
        "from-training-config", help="Prepare lab panels and pilots from an existing QASR YAML"
    )
    source.add_argument("--config", required=True)
    source.add_argument("--output")
    source.add_argument("--inspect", action="store_true")
    source.add_argument("--qasr-source")
    source.add_argument("--checkpoint", help="Explicitly override model_name_or_path")
    source.add_argument("--train-per-language", type=int, default=500)
    source.add_argument("--eval-per-language", type=int, default=50)
    source.add_argument("--non-speech-per-language", type=int, default=50)
    source.add_argument(
        "--musan",
        action="store_true",
        help="Prepare empty-target noise panels from the YAML noise_injection.noise_dir",
    )
    source.add_argument("--seeds", type=int, nargs="+")
    source.add_argument("--max-steps", type=int, default=200)
    source.add_argument("--hash-audio", action="store_true")
    return root


def dispatch(args):
    if args.command == "from-training-config":
        from .training_config import inspect_training_config, prepare_from_training_config

        if args.inspect:
            return inspect_training_config(
                args.config,
                qasr_source=args.qasr_source,
                checkpoint=args.checkpoint,
                musan=args.musan,
            )
        if not args.output:
            raise ValueError("--output is required unless --inspect is used")
        return prepare_from_training_config(
            args.config,
            args.output,
            qasr_source=args.qasr_source,
            checkpoint=args.checkpoint,
            train_per_language=args.train_per_language,
            eval_per_language=args.eval_per_language,
            non_speech_per_language=args.non_speech_per_language,
            seeds=args.seeds,
            max_steps=args.max_steps,
            hash_audio=args.hash_audio,
            musan=args.musan,
        )
    if args.command == "panel":
        result = prepare_panel(
            args.manifest,
            args.output,
            per_stratum=args.per_stratum,
            seed=args.seed,
            language=args.language,
            audio_root=args.audio_root,
            exclude_manifests=args.exclude_manifest,
            require_audio=args.require_audio,
            hash_audio=args.hash_audio,
        )
        return {"output": str(Path(args.output).resolve()), "samples": result["samples"]}
    if args.command == "run":
        return run_inference(args.model, args.panel, args.output)
    if args.command == "score":
        panel, samples = load_panel(args.panel)
        predictions = list(read_jsonl(args.predictions))
        report = score_predictions(
            panel, samples, predictions, {"id": args.model_id, "backend": "external"}
        )
        save_report(report, args.output, predictions)
        return {"output": str(Path(args.output).resolve()), "complete": report["complete"]}
    if args.command == "compare":
        result = compare_reports(
            args.report, args.output, iterations=args.bootstrap, seed=args.seed
        )
        return {"output": str(Path(args.output).resolve()), "candidates": len(result["candidates"])}
    if args.command == "import-legacy":
        return import_legacy(args.input, args.output, args.mode, args.model_id)
    if args.command == "audit-transfer":
        from .checkpoints import audit_transfer

        if not args.encoder and not args.decoder:
            raise ValueError("Supply --encoder and/or --decoder")
        result = audit_transfer(
            args.target, encoder=args.encoder, decoder=args.decoder, qasr_source=args.qasr_source
        )
        out = fresh_directory(args.output)
        write_json(out / "audit.json", result)
        return {
            "output": str(out),
            "passed": result["passed"],
            "verified_tensors": result["verified_tensors"],
        }
    if args.command == "plan":
        from .experiments import create_plan

        result = create_plan(args.base, args.output, non_speech_panel=args.non_speech_panel)
        return {
            "output": str(Path(args.output).resolve()),
            "experiments": len(result["experiments"]),
        }
    if args.command == "train":
        from .experiments import run_experiment

        return run_experiment(args.config)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"qasr-lab: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("complete", True) and result.get("passed", True) else 2
