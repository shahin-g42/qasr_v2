"""Readable console metrics; the full JSONL remains the machine-readable record."""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def format_record(record):
    step = record["step"]
    elapsed = record.get("elapsed_seconds", 0)
    lines = [
        "",
        f"=== GQASR | update {step:,} | elapsed {elapsed / 3600:.2f} h ===",
        f"Training loss (rank 0, last microbatch): {record['rank0_last_microbatch_loss']:.5f}",
    ]
    if "rank0_gradient_norm" in record:
        lines.append(f"Gradient norm (rank 0, before clipping): {record['rank0_gradient_norm']:.5g}")
    if record.get("learning_rates"):
        lines.append("Learning rates: " + " | ".join(
            f"{group}={lr:.3g}" for group, lr in record["learning_rates"].items()
        ))
    rows = record.get("rank0_last_microbatch_augmentation", [])
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


def print_record(record):
    if os.environ.get("GQASR_CONSOLE_FORMAT", "pretty") == "json":
        print(json.dumps(record), flush=True)
    else:
        print(format_record(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--from-step", type=int, default=0)
    args = parser.parse_args()
    with Path(args.metrics).open() as source:
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
                print_record(record)


if __name__ == "__main__":
    main()
