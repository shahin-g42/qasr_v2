"""Independent GPU workers with exact-ID validation before pooled scoring."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .qwen_baseline import read_rows, review_rows, select_samples, select_training_samples, summary


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def validate_merge(samples, shards):
    expected = {r["id"]: r for r in samples}
    if len(expected) != len(samples):
        raise ValueError("Duplicate selected sample IDs")
    found = {}
    for rows in shards:
        for row in rows:
            key = row["id"]
            if key not in expected or key in found:
                raise ValueError(f"Unexpected or duplicate prediction ID: {key}")
            for field in ("reference", "audio", "language", "stratum"):
                if row.get(field) != expected[key].get(field):
                    raise ValueError(f"Prediction/sample {field} mismatch: {key}")
            found[key] = row
    if found.keys() != expected.keys():
        raise ValueError(f"Missing {len(expected.keys() - found.keys())} predictions")
    return [found[r["id"]] for r in samples]


def run_parallel(args):
    gpus = args.gpus.split(",")
    if not all(gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be nonempty and unique")
    if args.samples:
        raise ValueError("--samples is reserved for workers")
    cfg = json.loads(Path(args.run_config).read_text())
    if args.train:
        samples, population = select_training_samples(cfg["index_path"], args.per_group, args.seed)
    else:
        samples, population = select_samples(cfg["index_path"], args.predictions), None
    if len(gpus) > len(samples):
        raise ValueError("More GPUs than samples")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "samples.jsonl", samples)
    (output / "selection.json").write_text(json.dumps(
        {"seed": args.seed, "population": population, "gpus": gpus,
         "count": len(samples), "modes": args.modes}, indent=2))
    children, logs = [], []
    try:
        for rank, gpu in enumerate(gpus):
            shard = output / f"samples-{rank}.jsonl"
            write_rows(shard, samples[rank::len(gpus)])
            destination = output / f"worker-{rank}"
            cmd = [sys.executable, "-m", "gqasr.qwen_baseline",
                   "--run-config", args.run_config, "--model", args.model,
                   "--output", str(destination), "--samples", str(shard),
                   "--device", "cuda:0", "--batch-size", str(args.batch_size),
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--seed", str(args.seed), "--per-group", str(args.per_group),
                   "--modes", *args.modes]
            cmd += ["--train"] if args.train else ["--predictions", args.predictions]
            log = (output / f"worker-{rank}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                                                  "PYTHONUNBUFFERED": "1"}))
        started = time.monotonic()
        while True:
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Baseline worker failed: exit codes {codes}; inspect {output}/worker-*.log")
            progress = {}
            for mode in args.modes:
                count = 0
                for rank in range(len(gpus)):
                    path = output / f"worker-{rank}" / f"predictions-{mode}.jsonl"
                    if path.exists():
                        with path.open() as source:
                            count += sum(line.endswith("\n") for line in source)
                progress[mode] = {"done": count, "total": len(samples)}
            print(json.dumps({"event": "parallel_baseline_progress", "gpus": len(gpus),
                              "elapsed_seconds": round(time.monotonic() - started, 1),
                              "progress": progress}), flush=True)
            if all(code is not None for code in codes):
                break
            time.sleep(5)
        # Validate every mode before publishing any aggregate report.
        merged = {mode: validate_merge(samples, [
            read_rows(output / f"worker-{rank}" / f"predictions-{mode}.jsonl")
            for rank in range(len(gpus))
        ]) for mode in args.modes}
        scores = {} if args.train else {
            "gqasr": summary([{**r, "hypothesis": r["gqasr_hypothesis"]} for r in samples])
        }
        for mode, rows in merged.items():
            write_rows(output / f"predictions-{mode}.jsonl", rows)
            write_rows(output / f"review-{mode}.jsonl", review_rows(rows))
            scores[mode] = summary(rows)
            if args.train:
                scores[mode]["by_stratum"] = {
                    group: summary([r for r in rows if r["stratum"] == group])
                    for group in ("han_only", "han_latin")
                }
                scores[mode]["sampling_note"] = (
                    "Balanced script strata, not population-weighted. "
                    "Use auto mode for mixed-language label review; baseline is not ground truth."
                )
        (output / "scores.json").write_text(json.dumps(scores, indent=2))
        (output / "provenance.json").write_text(json.dumps({
            "selection": json.loads((output / "selection.json").read_text()),
            "workers": [json.loads((output / f"worker-{rank}" / "provenance.json").read_text())
                        for rank in range(len(gpus))]
        }, indent=2))
        print(f"Baseline complete: {output}/scores.json", flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
