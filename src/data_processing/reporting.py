"""QA reporting: statistics, dialect distribution, and sample dumps."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest_io import read_cleaned_manifest

LOGGER = logging.getLogger("data_processing.reporting")


@dataclass
class ManifestReport:
    """QA report for a single processed manifest."""

    source_manifest: str
    total_records: int = 0
    processed_records: int = 0
    rejected_records: int = 0
    dialect_distribution: dict[str, float] = field(default_factory=dict)
    change_distribution: dict[str, float] = field(default_factory=dict)
    avg_confidence: float = 0.0
    validation_pass_rate: float = 0.0
    processing_time_hours: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_manifest": self.source_manifest,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "rejected_records": self.rejected_records,
            "dialect_distribution": self.dialect_distribution,
            "change_distribution": self.change_distribution,
            "avg_confidence": round(self.avg_confidence, 4),
            "validation_pass_rate": round(self.validation_pass_rate, 4),
            "processing_time_hours": round(self.processing_time_hours, 2),
            "samples": self.samples,
        }


def generate_manifest_report(
    source_path: str | Path,
    cleaned_path: str | Path,
    rejected_path: str | Path | None = None,
    num_samples: int = 10,
    processing_start_time: float | None = None,
) -> ManifestReport:
    """Generate a QA report for a processed manifest."""
    report = ManifestReport(source_manifest=str(source_path))

    cleaned_path = Path(cleaned_path)
    if not cleaned_path.is_file():
        LOGGER.warning("Cleaned manifest not found: %s", cleaned_path)
        return report

    # Read cleaned records
    dialects: Counter[str] = Counter()
    changes: Counter[str] = Counter()
    confidences: list[float] = []
    records: list[dict[str, Any]] = []

    for record in read_cleaned_manifest(cleaned_path):
        records.append(record)
        report.processed_records += 1

        dialect = record.get("dialect", "unknown")
        dialects[dialect] += 1

        for change in record.get("changes", []):
            changes[change] += 1

        confidence = record.get("confidence", 0.0)
        confidences.append(confidence)

    # Count rejected records
    if rejected_path and Path(rejected_path).is_file():
        for _ in read_cleaned_manifest(rejected_path):
            report.rejected_records += 1

    report.total_records = report.processed_records + report.rejected_records

    # Compute distributions
    total = max(report.processed_records, 1)
    report.dialect_distribution = {
        k: round(v / total, 4) for k, v in dialects.most_common()
    }
    report.change_distribution = {
        k: round(v / total, 4) for k, v in changes.most_common()
    }

    # Average confidence
    if confidences:
        report.avg_confidence = sum(confidences) / len(confidences)

    # Validation pass rate
    if report.total_records > 0:
        report.validation_pass_rate = report.processed_records / report.total_records

    # Processing time
    if processing_start_time:
        report.processing_time_hours = (time.time() - processing_start_time) / 3600

    # Sample records
    if records and num_samples > 0:
        sample_records = random.sample(records, min(num_samples, len(records)))
        report.samples = [
            {
                "original": r.get("original_text", ""),
                "cleaned": r.get("text", ""),
                "dialect": r.get("dialect", "unknown"),
                "changes": r.get("changes", []),
                "confidence": r.get("confidence", 0.0),
            }
            for r in sample_records
        ]

    return report


def save_report(report: ManifestReport, output_path: str | Path) -> None:
    """Save a manifest report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved report to %s", output_path)


def generate_summary_report(
    reports: list[ManifestReport],
    output_path: str | Path,
) -> dict[str, Any]:
    """Generate a summary report across all manifests."""
    summary = {
        "total_manifests": len(reports),
        "total_records": sum(r.total_records for r in reports),
        "total_processed": sum(r.processed_records for r in reports),
        "total_rejected": sum(r.rejected_records for r in reports),
        "avg_confidence": (
            sum(r.avg_confidence for r in reports) / len(reports)
            if reports else 0.0
        ),
        "avg_validation_pass_rate": (
            sum(r.validation_pass_rate for r in reports) / len(reports)
            if reports else 0.0
        ),
        "total_processing_hours": sum(r.processing_time_hours for r in reports),
        "manifests": [
            {
                "source": r.source_manifest,
                "records": r.total_records,
                "processed": r.processed_records,
                "rejected": r.rejected_records,
            }
            for r in reports
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved summary report to %s", output_path)
    return summary


__all__ = [
    "ManifestReport",
    "generate_manifest_report",
    "generate_summary_report",
    "save_report",
]
