"""Bind exploratory reference scores to the exact frozen annotation release."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .core import DATASETS
from .release import validate_bundle


def dataset_pin(repo_id, revision):
    if not repo_id and not revision:
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id or ""):
        raise ValueError("Dataset repository must be OWNER/NAME")
    if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise ValueError("Dataset revision must be an immutable 40-character commit SHA")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "url": f"https://huggingface.co/datasets/{repo_id}/tree/{revision}",
    }


def annotation_drafts(root, plan, rows):
    """Reject missing, reordered-content, or substituted prediction/reference inputs."""
    import pyarrow.parquet as pq

    root = Path(root)
    manifest = validate_bundle(root)
    if manifest["repo_type"] != "dataset" or manifest["manifest_digest"] != plan["manifest_digest"]:
        raise ValueError("Annotation bundle does not match the frozen leaderboard plan")
    expected = {r["id"]: r for r in rows}
    seen, drafts, annotations = set(), [], []
    exclusions = Counter()
    for ds in manifest["datasets"]:
        for item in pq.read_table(root / "data" / ds / "test.parquet").to_pylist():
            key = item["id"]
            if key not in expected or key in seen:
                raise ValueError("Annotation membership must exactly match the plan")
            source = expected[key]
            for field in ("dataset", "text", "audio_sha256", "binding", "policy_sha256", "cohorts"):
                if item[field] != source[field]:
                    raise ValueError(f"Annotation {field} mismatch: {key}")
            if item["dataset"] != ds or item["duration_seconds"] != source["actual_duration"]:
                raise ValueError(f"Annotation dataset/duration mismatch: {key}")
            if json.loads(item["candidate_texts_json"]) != {
                c["name"]: c["text"] for c in source["candidates"]
            }:
                raise ValueError(f"Annotation predictions mismatch: {key}")
            seen.add(key)
            annotations.append(item)
            if item["speculative"] or item["proposal_status"] == "speculative_draft":
                exclusions["speculative"] += 1
            elif (
                item["proposal_status"] != "proposed"
                or not isinstance(item["ideal_text_draft"], str)
                or not item["ideal_text_draft"].strip()
            ):
                exclusions["missing_completed_draft"] += 1
            else:
                drafts.append({**source, "ideal_text_draft": item["ideal_text_draft"]})
    if seen != set(expected):
        raise ValueError("Annotation membership must exactly match the plan")
    # Ensure the inputs did not change while the tables were being read.
    validate_bundle(root)
    return (
        drafts,
        annotations,
        {
            "bundle_sha256": manifest["bundle_sha256"],
            "release_id": manifest["release_id"],
            "eligible_clips": len(drafts),
            "excluded_clips": len(rows) - len(drafts),
            "exclusions": dict(exclusions),
            "reference_column": "ideal_text_draft",
            "status": "candidate_assisted_unverified",
        },
    )


def transcript_examples(rows, drafts, annotations, additions=None):
    """Three hash-ranked clips per source; selection never uses a model's score."""
    by_id = {r["id"]: r for r in annotations}
    examples = []
    pool = drafts if annotations else rows
    for ds in DATASETS:
        candidates = [r for r in pool if r["dataset"] == ds]
        candidates.sort(
            key=lambda r: hashlib.sha256(f"board-example-v1:{r['id']}".encode()).hexdigest()
        )
        for row in candidates[:3]:
            item = by_id.get(row["id"], {})
            examples.append(
                {
                    "id": row["id"],
                    "dataset": ds,
                    "duration_seconds": row["actual_duration"],
                    "dialect": row.get("metadata", {}).get("dialect"),
                    "original": row["text"],
                    "ideal_draft": row.get("ideal_text_draft"),
                    "predictions": {
                        c["name"]: c["text"]
                        for c in [*row["candidates"], *(additions or {}).get(row["id"], [])]
                    },
                    "flags": row["flags"],
                    "editorial_flags": json.loads(item.get("checks_json") or "{}").get(
                        "editorial_flags", []
                    ),
                }
            )
    return examples
