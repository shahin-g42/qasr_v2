"""Portable listening packs, attested human review, and paired-subset scoring."""

from __future__ import annotations

import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

from .core import (
    MARKED_UNCLEAR_RE,
    atomic_text,
    digest,
    load_plan,
    read_jsonl,
    review_issues,
    sha256,
    validate_ideal_pair,
    validate_issue_notes,
    write_json,
    write_jsonl,
)
from .prompts import blinded
from .runtime import load_proposals


def review_pack(plan_root, output, proposals=None, audio_root=None, metadata_only=False):
    plan, rows = load_plan(plan_root)
    proposals_by_id = load_proposals(proposals, plan, rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        name = row["id"].replace(":", "-") + ".wav"
        audio = (
            Path(audio_root) / row["dataset"] / f"{row['ordinal']:06d}.npy"
            if audio_root
            else Path(row["audio"])
        )
        audio_url = None
        if not metadata_only:
            import numpy as np

            if sha256(audio) != row["audio_sha256"]:
                raise ValueError(f"Audio hash mismatch: {row['id']}")
            data = np.load(audio, allow_pickle=False)
            if (
                data.ndim != 1
                or not np.isfinite(data).all()
                or abs(len(data) / 16000 - row["actual_duration"]) > 2 / 16000
            ):
                raise ValueError(f"Unexpected waveform: {row['id']}")
            target = output / "audio" / name
            target.parent.mkdir(exist_ok=True)
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes((np.clip(data, -1, 1) * 32767).round().astype("<i2").tobytes())
            audio_url = "audio/" + name
        proposal = proposals_by_id.get(row["id"], {}).get("proposal")
        jobs.append(
            {
                "id": row["id"],
                "binding": row["binding"],
                "policy_sha256": row["policy_sha256"],
                "audio_sha256": row["audio_sha256"],
                "audio_url": audio_url,
                "review_audio_sha256": sha256(output / audio_url) if audio_url else None,
                "dataset": row["dataset"],
                "dialect": row["metadata"].get("dialect"),
                "duration": row["actual_duration"],
                "candidates": blinded(row),
                "review_issues": review_issues(row),
                "proposal": proposal,
                "validation_adjustments": proposals_by_id.get(row["id"], {}).get(
                    "validation_adjustments", []
                ),
            }
        )
    write_json(
        output / "jobs.json",
        {
            "policy": plan["policy"],
            "manifest_digest": plan["manifest_digest"],
            "metadata_only": metadata_only,
            "jobs": jobs,
        },
    )
    template = Path(__file__).with_name("review.html").read_text()
    encoded = json.dumps(jobs, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    policy = (
        json.dumps(plan["policy"], ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    atomic_text(
        output / "index.html",
        template.replace("__JOBS_JSON__", encoded).replace("__POLICY_JSON__", policy),
    )
    return {
        "clips": len(jobs),
        "audio_exported": not metadata_only,
        "index": str(output / "index.html"),
    }


def validate_review(review, row):
    required = {
        "id",
        "binding",
        "policy_sha256",
        "audio_sha256",
        "reviewer_id",
        "audio_listened",
        "decision",
        "verified_text",
        "ideal_text",
        "reason",
        "issue_resolutions",
        "ideal_spelling_reason",
    }
    if set(review) != required:
        raise ValueError("Review schema mismatch")
    for field in ("id", "binding", "policy_sha256", "audio_sha256"):
        if review[field] != row[field]:
            raise ValueError(f"Review {field} mismatch")
    if not isinstance(review["ideal_spelling_reason"], str):
        raise ValueError("Ideal spelling explanation must be text")
    for field in ("reviewer_id", "reason"):
        if not isinstance(review[field], str) or not review[field].strip():
            raise ValueError(f"Review requires {field}")
    if review["audio_listened"] is not True:
        raise ValueError("Audio listening attestation required")
    if review["decision"] not in ("verified", "unresolved"):
        raise ValueError("Review decision must be verified or unresolved")
    validate_issue_notes(review["issue_resolutions"], row)
    a, b = review["verified_text"], review["ideal_text"]
    if review["decision"] == "unresolved":
        if a is not None or b is not None:
            raise ValueError("Unresolved texts must be null")
    else:
        if not all(isinstance(t, str) and t.strip() for t in (a, b)):
            raise ValueError("Verified texts must be nonempty")
        validate_ideal_pair(a, b, review["ideal_spelling_reason"])
        if any(MARKED_UNCLEAR_RE.search(t) for t in (a, b)):
            raise ValueError("Marked unclear span cannot be scored as fully verified")


def check_duplicate_review(seen, row):
    texts = (row["verified_text"], row["ideal_text"])
    previous = seen.setdefault(row["audio_sha256"], texts)
    if previous != texts:
        raise ValueError(
            "Conflicting verified/ideal transcripts for identical audio; adjudication required"
        )


def import_reviews(plan_root, review_files, output):
    plan, rows = load_plan(plan_root)
    by_id = {r["id"]: r for r in rows}
    submissions = {}
    for path in review_files:
        for review in read_jsonl(path):
            identifier = review.get("id")
            if identifier not in by_id or identifier in submissions:
                raise ValueError(f"Unknown or duplicate reviewed ID: {identifier}")
            validate_review(review, by_id[identifier])
            submissions[identifier] = review
    reviewed_hashes = {}
    output_rows = []
    for row in rows:
        updated = dict(row)
        review = submissions.get(row["id"])
        if review:
            updated.update(
                {
                    "review_status": review["decision"],
                    "review": review,
                    "verified_text": review["verified_text"],
                    "ideal_text": review["ideal_text"],
                }
            )
            if review["decision"] == "verified":
                check_duplicate_review(reviewed_hashes, updated)
        updated["reviewed_sha256"] = digest(updated)
        output_rows.append(updated)
    output = Path(output)
    if output.exists() and list(read_jsonl(output)) != output_rows:
        raise ValueError("Reviewed version already exists; write a new version")
    write_jsonl(output, output_rows)
    return {
        "rows": len(rows),
        "statuses": dict(Counter(r["review_status"] for r in output_rows)),
        "manifest_digest": plan["manifest_digest"],
        "reviewed_digest": digest(output_rows),
        "review_level": "single human attestation; public benchmark freeze requires independent audit",
    }


def score(plan_root, reviewed_path, output):
    from gqasr.arabic_benchmark.scoring import counts, rates

    plan, originals = load_plan(plan_root)
    by_id = {r["id"]: r for r in originals}
    reviewed = list(read_jsonl(reviewed_path))
    if len(reviewed) != len(originals) or {r["id"] for r in reviewed} != set(by_id):
        raise ValueError("Reviewed manifest coverage differs from pilot")
    eligible = []
    reviewed_hashes = {}
    for row in reviewed:
        hashed = {k: v for k, v in row.items() if k != "reviewed_sha256"}
        if digest(hashed) != row.get("reviewed_sha256"):
            raise ValueError("Reviewed manifest hash mismatch")
        source = by_id[row["id"]]
        for key, value in source.items():
            if key not in ("review_status", "verified_text", "ideal_text") and row[key] != value:
                raise ValueError("Original benchmark data changed")
        if row["review_status"] == "verified":
            validate_review(row["review"], source)
            if row["review"]["decision"] != "verified" or any(
                row[k] != row["review"][k] for k in ("verified_text", "ideal_text")
            ):
                raise ValueError("Reviewed text does not match attestation")
            check_duplicate_review(reviewed_hashes, row)
            eligible.append(row)
    if not eligible:
        raise ValueError(
            "No audio-reviewed verified references; proposals cannot be scored as gold"
        )
    groups = defaultdict(Counter)
    for row in eligible:
        for candidate in row["candidates"]:
            for track, key, normalized in [
                ("original_elm", "text", True),
                ("verified_elm", "verified_text", True),
                ("ideal_strict", "ideal_text", False),
            ]:
                c = counts(row[key], candidate["text"], normalized=normalized)
                for cohort in row["cohorts"]:
                    groups[(cohort, candidate["name"], track, row["dataset"])].update(c)
    results = {"/".join(key): rates(value) for key, value in groups.items()}
    for cohort, model, track, _ in groups:
        prefix = f"{cohort}/{model}/{track}"
        data = [
            results[prefix + "/" + ds]
            for ds in plan["input_counts"]
            if prefix + "/" + ds in results
        ]
        results[prefix + "/macro"] = {
            "datasets": len(data),
            "full_suite": False,
            "wer": sum(r["wer"] for r in data) / 6 if len(data) == 6 else None,
            "cer": sum(r["cer"] for r in data) / 6 if len(data) == 6 else None,
        }
    result = {
        "schema_version": 1,
        "manifest_digest": plan["manifest_digest"],
        "reviewed_sha256": sha256(reviewed_path),
        "verified_clips": len(eligible),
        "selected_clips": len(originals),
        "pending_or_unresolved": len(originals) - len(eligible),
        "coverage": {
            cohort: {
                "selected": dict(
                    Counter(r["dataset"] for r in originals if cohort in r["cohorts"])
                ),
                "verified": dict(Counter(r["dataset"] for r in eligible if cohort in r["cohorts"])),
            }
            for cohort in ("pilot", "diagnostic")
        },
        "note": "Paired verified subsets only; diagnostics are outcome-selected. Not an official leaderboard result. ideal_strict measures full-string agreement, not an isolated punctuation/diacritic quality metric.",
        "scores": results,
    }
    write_json(output, result)
    return result
