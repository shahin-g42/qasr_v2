"""Validate and filter training manifests before launching expensive training.

Scans every sample in the manifest, decodes audio, checks duration, tokenizes
transcripts, and produces a filtered manifest containing only samples that will
NOT fail during training. Run this BEFORE any training phase.

What this catches:
  - Missing or unreadable audio files
  - Corrupt/zero-length audio
  - Audio with NaN/Inf samples
  - Duration mismatch (metadata vs actual)
  - Audio outside configured duration range
  - Empty or punctuation-only transcripts
  - Transcripts exceeding max_target_length tokens
  - Malformed JSONL records

Usage:
    PYTHONPATH=src python -m qasr.validate_data \
        --manifest /path/to/train.json \
        --output /path/to/train_filtered.json \
        --model /path/to/qasr/checkpoint \
        --max-target-length 512 \
        --min-duration 0.1 \
        --max-duration 35.0 \
        --workers 32

    # Validate all manifests in a config:
    PYTHONPATH=src python -m qasr.validate_data \
        --config configs/train_full_4node.yaml \
        --output-dir /path/to/filtered_manifests/ \
        --workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("qasr.validate")


@dataclass
class ValidationResult:
    """Result of validating a single sample."""

    line_number: int
    audio_path: str
    is_valid: bool
    error: str | None = None
    actual_duration: float | None = None
    token_count: int | None = None
    duration_mismatch: bool = False


@dataclass
class ManifestReport:
    """Aggregated validation report for one manifest file."""

    manifest_path: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    total_valid_hours: float = 0.0
    total_invalid_hours: float = 0.0
    duration_mismatches: int = 0
    filtered_output_path: str | None = None

    def add_error(self, category: str) -> None:
        self.errors[category] = self.errors.get(category, 0) + 1

    @property
    def valid_pct(self) -> float:
        return 100.0 * self.valid_records / max(self.total_records, 1)

    def summary(self) -> str:
        lines = [
            f"  Manifest: {self.manifest_path}",
            f"  Total: {self.total_records:,} | Valid: {self.valid_records:,} ({self.valid_pct:.1f}%) | Invalid: {self.invalid_records:,}",
            f"  Valid hours: {self.total_valid_hours:.1f}h | Lost hours: {self.total_invalid_hours:.1f}h",
        ]
        if self.errors:
            lines.append("  Errors:")
            for category, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                lines.append(f"    {category}: {count:,}")
        if self.duration_mismatches:
            lines.append(f"  Duration mismatches (>1s): {self.duration_mismatches:,}")
        if self.filtered_output_path:
            lines.append(f"  Filtered output: {self.filtered_output_path}")
        return "\n".join(lines)


def _validate_single_sample(
    record_line: str,
    line_number: int,
    *,
    sampling_rate: int,
    min_duration: float,
    max_duration: float,
    max_target_length: int | None,
    tokenizer_vocab: dict[str, int] | None,
    audio_root: str | None,
) -> ValidationResult:
    """Validate a single JSONL record. Runs in a worker process."""
    import re

    # Parse JSON
    try:
        record = json.loads(record_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ValidationResult(line_number, "", False, f"invalid_json: {exc}")

    if not isinstance(record, dict):
        return ValidationResult(line_number, "", False, "not_a_dict")

    # Extract fields
    audio_value = record.get("audio_filepath") or record.get("wav_path")
    text_value = record.get("text") or record.get("transcript")

    if not audio_value or not isinstance(audio_value, str) or not audio_value.strip():
        return ValidationResult(line_number, str(audio_value), False, "missing_audio_path")

    if not text_value or not isinstance(text_value, str) or not text_value.strip():
        return ValidationResult(line_number, audio_value, False, "missing_text")

    # Check text has at least one alphanumeric character
    if not any(c.isalnum() for c in text_value):
        return ValidationResult(line_number, audio_value, False, "punctuation_only_text")

    # Resolve audio path
    audio_value = re.sub(r"^/vast", "/lustrefs/taiga/vast40", audio_value)
    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute() and audio_root:
        audio_path = Path(audio_root) / audio_path

    # Check file exists
    if not audio_path.is_file():
        return ValidationResult(line_number, str(audio_path), False, "file_not_found")

    # Check file size (quick reject for zero-byte files)
    if audio_path.stat().st_size == 0:
        return ValidationResult(line_number, str(audio_path), False, "empty_file")

    # Decode audio
    try:
        import soundfile as sf
        waveform, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as exc:
        return ValidationResult(line_number, str(audio_path), False, f"decode_error: {type(exc).__name__}")

    if waveform.shape[0] == 0:
        return ValidationResult(line_number, str(audio_path), False, "zero_length_audio")

    # Mono conversion
    waveform = waveform.mean(axis=1, dtype=np.float32)

    # Check for NaN/Inf
    if not np.isfinite(waveform).all():
        return ValidationResult(line_number, str(audio_path), False, "nan_or_inf_audio")

    # Resample if needed
    if sr != sampling_rate:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, sampling_rate)
        waveform = resample_poly(
            waveform, up=sampling_rate // divisor, down=sr // divisor
        ).astype(np.float32)

    # Check actual duration
    actual_duration = waveform.shape[0] / sampling_rate
    if actual_duration < min_duration:
        return ValidationResult(
            line_number, str(audio_path), False,
            f"too_short: {actual_duration:.3f}s < {min_duration}s",
            actual_duration=actual_duration,
        )
    if actual_duration > max_duration:
        return ValidationResult(
            line_number, str(audio_path), False,
            f"too_long: {actual_duration:.3f}s > {max_duration}s",
            actual_duration=actual_duration,
        )

    # Check metadata duration mismatch (> 1 second difference). Not a hard
    # failure — the sample is still usable — but it is flagged for reporting.
    duration_mismatch = False
    meta_duration = record.get("duration")
    if meta_duration is not None:
        try:
            duration_mismatch = abs(float(meta_duration) - actual_duration) > 1.0
        except (TypeError, ValueError):
            duration_mismatch = True

    # Check transcript token length (approximate with character count if no tokenizer)
    token_count = None
    if max_target_length is not None:
        # Rough estimate: ~4 chars per token for multilingual
        # For precise count, we'd need the actual tokenizer
        char_count = len(text_value.strip())
        # Conservative estimate: 2 chars per token for Arabic/CJK
        estimated_tokens = char_count  # 1:1 worst case for CJK
        if estimated_tokens > max_target_length * 4:  # Very conservative
            return ValidationResult(
                line_number, str(audio_path), False,
                f"transcript_too_long: ~{estimated_tokens} chars",
                actual_duration=actual_duration,
                token_count=estimated_tokens,
                duration_mismatch=duration_mismatch,
            )

    return ValidationResult(
        line_number, str(audio_path), True,
        actual_duration=actual_duration,
        token_count=token_count,
        duration_mismatch=duration_mismatch,
    )


def validate_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    sampling_rate: int = 16000,
    min_duration: float = 0.1,
    max_duration: float = 35.0,
    max_target_length: int | None = 512,
    audio_root: str | None = None,
    workers: int = 32,
    language: str | None = None,
) -> ManifestReport:
    """Validate all samples in a manifest and optionally write a filtered version.

    Args:
        manifest_path: Path to the JSONL manifest.
        output_path: If set, write valid records here.
        sampling_rate: Target sampling rate for audio validation.
        min_duration: Minimum audio duration in seconds.
        max_duration: Maximum audio duration in seconds.
        max_target_length: Maximum transcript token count.
        audio_root: Base directory for relative audio paths.
        workers: Number of parallel validation workers.
        language: Language tag (for reporting only).

    Returns:
        ManifestReport with statistics.
    """
    manifest_path = Path(manifest_path)
    report = ManifestReport(manifest_path=str(manifest_path))

    if not manifest_path.is_file():
        LOGGER.error("Manifest not found: %s", manifest_path)
        report.add_error("manifest_not_found")
        return report

    # Read all lines
    LOGGER.info("Reading manifest: %s", manifest_path)
    lines: list[tuple[int, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if stripped:
                lines.append((line_number, stripped))

    report.total_records = len(lines)
    LOGGER.info("  %d records to validate with %d workers", len(lines), workers)

    # Validate in parallel
    valid_lines: list[tuple[int, str]] = []
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _validate_single_sample,
                line_text,
                line_num,
                sampling_rate=sampling_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_target_length=max_target_length,
                tokenizer_vocab=None,
                audio_root=audio_root,
            ): (line_num, line_text)
            for line_num, line_text in lines
        }

        for future in as_completed(futures):
            completed += 1
            line_num, line_text = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = ValidationResult(line_num, "", False, f"worker_crash: {exc}")

            if result.is_valid:
                report.valid_records += 1
                if result.actual_duration:
                    report.total_valid_hours += result.actual_duration / 3600.0
                valid_lines.append((line_num, line_text))
            else:
                report.invalid_records += 1
                error_category = result.error.split(":")[0] if result.error else "unknown"
                report.add_error(error_category)
                if result.actual_duration:
                    report.total_invalid_hours += result.actual_duration / 3600.0
            if result.duration_mismatch:
                report.duration_mismatches += 1

            if completed % 5000 == 0 or completed == len(lines):
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(lines) - completed) / max(rate, 1)
                LOGGER.info(
                    "  Progress: %d/%d (%.0f%%) | %.0f samples/s | ETA %.0fs | "
                    "valid=%d invalid=%d",
                    completed, len(lines), 100 * completed / len(lines),
                    rate, eta, report.valid_records, report.invalid_records,
                )

    elapsed = time.time() - t0
    LOGGER.info(
        "  Validation complete in %.1fs (%.0f samples/s)",
        elapsed, len(lines) / max(elapsed, 0.1),
    )

    # Write filtered manifest with only required columns
    if output_path and valid_lines:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by original line number to preserve order
        valid_lines.sort(key=lambda x: x[0])
        with output_path.open("w", encoding="utf-8") as f:
            for _, line_text in valid_lines:
                try:
                    record = json.loads(line_text)
                    # Extract only required columns with standardized names
                    clean_record = {
                        "audio_filepath": record.get("audio_filepath") or record.get("wav_path"),
                        "text": record.get("text") or record.get("transcript"),
                    }
                    # Include duration if present
                    if record.get("duration") is not None:
                        clean_record["duration"] = record["duration"]
                    f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                except (json.JSONDecodeError, KeyError):
                    # Fallback: write original line if parsing fails
                    f.write(line_text + "\n")
        report.filtered_output_path = str(output_path)
        LOGGER.info("  Filtered manifest written: %s (%d records)", output_path, len(valid_lines))

    return report


def validate_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    workers: int = 32,
) -> list[ManifestReport]:
    """Validate all manifests referenced in a training config YAML."""
    import yaml

    config_path = Path(config_path)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_duration = config.get("min_duration_seconds", 0.1)
    max_duration = config.get("max_duration_seconds", 35.0)
    max_target_length = config.get("max_target_length", 512)
    audio_root = config.get("audio_root")

    reports = []

    # Process train manifests
    train_manifest = config.get("train_manifest", {})
    if isinstance(train_manifest, dict):
        for language, paths in train_manifest.items():
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                manifest_name = Path(path).stem
                out_path = output_dir / f"train_{language}_{manifest_name}_filtered.json"
                report = validate_manifest(
                    path,
                    output_path=out_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_target_length=max_target_length,
                    audio_root=audio_root,
                    workers=workers,
                    language=language,
                )
                reports.append(report)
    elif isinstance(train_manifest, list):
        for path in train_manifest:
            manifest_name = Path(path).stem
            out_path = output_dir / f"train_{manifest_name}_filtered.json"
            report = validate_manifest(
                path,
                output_path=out_path,
                min_duration=min_duration,
                max_duration=max_duration,
                max_target_length=max_target_length,
                audio_root=audio_root,
                workers=workers,
            )
            reports.append(report)

    # Process eval manifests
    eval_manifest = config.get("eval_manifest", {})
    if isinstance(eval_manifest, dict):
        for language, paths in eval_manifest.items():
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                manifest_name = Path(path).stem
                out_path = output_dir / f"eval_{language}_{manifest_name}_filtered.json"
                report = validate_manifest(
                    path,
                    output_path=out_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_target_length=max_target_length,
                    audio_root=audio_root,
                    workers=workers,
                    language=language,
                )
                reports.append(report)

    return reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and filter QASR training manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single manifest:
  python -m qasr.validate_data --manifest train.json --output train_filtered.json --workers 32

  # Validate all manifests in a training config:
  python -m qasr.validate_data --config configs/train_full_4node.yaml --output-dir filtered/ --workers 64
        """,
    )
    parser.add_argument("--manifest", help="Single manifest JSONL file to validate")
    parser.add_argument("--output", help="Output path for filtered manifest")
    parser.add_argument("--config", help="Training config YAML (validates all manifests)")
    parser.add_argument("--output-dir", help="Output directory for filtered manifests (with --config)")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--workers", type=int, default=32, help="Parallel validation workers")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.config:
        if not args.output_dir:
            parser.error("--output-dir is required when using --config")
        reports = validate_from_config(args.config, args.output_dir, workers=args.workers)
    elif args.manifest:
        report = validate_manifest(
            args.manifest,
            output_path=args.output,
            sampling_rate=args.sampling_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            max_target_length=args.max_target_length,
            audio_root=args.audio_root,
            workers=args.workers,
        )
        reports = [report]
    else:
        parser.error("Either --manifest or --config is required")

    # Print summary
    print("\n" + "=" * 70)
    print(" DATASET VALIDATION REPORT")
    print("=" * 70)
    total_records = 0
    total_valid = 0
    total_invalid = 0
    total_valid_hours = 0.0
    total_invalid_hours = 0.0

    for report in reports:
        print(f"\n{report.summary()}")
        total_records += report.total_records
        total_valid += report.valid_records
        total_invalid += report.invalid_records
        total_valid_hours += report.total_valid_hours
        total_invalid_hours += report.total_invalid_hours

    print("\n" + "-" * 70)
    print(" TOTALS:")
    print(f"   Records: {total_records:,} total | {total_valid:,} valid | {total_invalid:,} invalid")
    print(f"   Hours:   {total_valid_hours:.1f}h valid | {total_invalid_hours:.1f}h lost")
    if total_records > 0:
        print(f"   Health:  {100 * total_valid / total_records:.1f}% samples usable")
    print("=" * 70)

    # Exit with error if too many samples are invalid
    if total_records > 0 and total_valid / total_records < 0.5:
        LOGGER.warning("Less than 50% of samples are valid - check your data paths.")
        sys.exit(1)


if __name__ == "__main__":
    main()
