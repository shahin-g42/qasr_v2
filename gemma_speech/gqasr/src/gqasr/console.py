"""Readable console metrics; the full JSONL remains the machine-readable record."""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def format_record(record, config=None):
    step = record["step"]
    elapsed = record.get("elapsed_seconds", 0)
    config = config or {}
    total = config.get("max_steps")
    progress = f"{step:,}"
    if total and total > 0:
        progress += f" / {total:,} ({100 * step / total:.2f}%) | remaining {max(0, total - step):,}"
    lines = [
        "",
        f"=== GQASR | update {progress} | elapsed {elapsed / 3600:.2f} h ===",
    ]
    if "loss" in record:
        lines.append(
            f"Training loss (all ranks, {record['loss_window_updates']} updates, "
            f"{record['loss_microbatches']:,} microbatches): {record['loss']:.5f}"
        )
    if "rank0_last_microbatch_loss" in record:
        lines.append(f"Training loss (rank 0, last microbatch): {record['rank0_last_microbatch_loss']:.5f}")
    elif record.get("event") == "initial_validation":
        lines.append("Initial validation: loaded checkpoint, before any optimizer update.")
    upcoming = []
    if total and step < total:
        for key, label in (("eval_steps", "validation"), ("save_steps", "checkpoint")):
            interval = config.get(key)
            if interval and interval > 0:
                upcoming.append(f"{label}: update {min(total, (step // interval + 1) * interval):,}")
    if upcoming:
        lines.append("Next " + " | ".join(upcoming))
    if "rank0_gradient_norm" in record:
        lines.append(f"Gradient norm (rank 0, before clipping): {record['rank0_gradient_norm']:.5g}")
    if record.get("learning_rates"):
        lines.append("Learning rates: " + " | ".join(
            f"{group}={lr:.3g}" for group, lr in record["learning_rates"].items()
        ))
    rows = record.get("rank0_last_microbatch_augmentation", [])
    timing = record.get("performance", {})
    if "seconds_per_update" in timing:
        lines.append(
            f"Throughput (recent window): {timing['seconds_per_update']:.2f} s/update"
            f" | {timing['examples_per_second']:.1f} global anchors/s"
        )
    if "rank0_data_wait_seconds_per_update" in timing:
        lines.append(
            f"DataLoader wait (rank 0): {timing['rank0_data_wait_seconds_per_update']:.3f} s/update"
            f" | {timing['rank0_data_wait_seconds_per_microbatch']:.3f} s/microbatch"
        )
    if rows:
        n = len(rows)
        lines.append("Augmentation (rank 0): " + " | ".join(
            f"{label}={sum(bool(r.get(key)) for r in rows)}/{n}" for key, label in (
                ("triggered", "code-switch triggered"), ("applied", "mixed"),
                ("waveform_changed", "waveform changed"), ("specaugment_changed", "SpecAugment"),
            )
        ))
        reasons = Counter(r.get("reason", "unspecified") for r in rows)
        pairs = Counter(r["pair"] for r in rows if r.get("applied") and r.get("pair"))
        lines.append("Code-switch outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
        if pairs:
            lines.append("Mixed pairs: " + ", ".join(f"{k}={v}" for k, v in sorted(pairs.items())))
    if record.get("rank0_last_microbatch_data"):
        sources = Counter(r.get("used", "unspecified") for r in record["rank0_last_microbatch_data"])
        lines.append("Data sources (rank 0): " + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())))
    scores = record.get("validation")
    if scores is not None:
        lines.extend([
            "",
            f"VALIDATION COMPLETE | {scores.get('strict/all', {}).get('utterances', '?')} utterances | gathered across ranks",
            f"{'Metric group':<38} {'N':>6} {'WER':>9} {'CER':>9} {'Word errors/ref':>17} {'Char errors/ref':>19}",
            "-" * 103,
        ])
        for group, values in sorted(scores.items(), key=lambda item: (not item[0].startswith("strict/"), item[0])):
            words = f"{values.get('word_errors', '?')}/{values.get('words', '?')}"
            chars = f"{values.get('character_errors', '?')}/{values.get('characters', '?')}"
            lines.append(
                f"{group:<38} {values.get('utterances', 0):>6} "
                f"{percent(values.get('wer')):>9} {percent(values.get('cer')):>9} "
                f"{words:>17} {chars:>19}"
            )
            if values.get("empty_references") or values.get("nonempty_on_empty"):
                lines.append(f"  Empty references: {values.get('empty_references', 0)}; "
                             f"nonempty predictions on empty references: {values.get('nonempty_on_empty', 0)}")
        lines.append("WER uses whitespace words; for Chinese, prioritize CER. Rates can exceed 100%.")
        lines.append(f"Predictions: predictions-{step}.jsonl (in the training output directory)")
    lines.append("Full per-example diagnostics are retained in metrics.jsonl.")
    return "\n".join(lines)


def print_record(record, config=None):
    if os.environ.get("GQASR_CONSOLE_FORMAT", "pretty") == "json":
        print(json.dumps(record), flush=True)
    else:
        print(format_record(record, config), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--from-step", type=int, default=0)
    args = parser.parse_args()
    metrics = Path(args.metrics)
    config_path = metrics.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    with metrics.open() as source:
        while True:
            position = source.tell()
            line = source.readline()
            if not line.endswith("\n"):
                source.seek(position)
                if not args.follow:
                    break
                time.sleep(2)
                continue
            record = json.loads(line)
            if record["step"] >= args.from_step:
                print_record(record, config)


if __name__ == "__main__":
    main()
