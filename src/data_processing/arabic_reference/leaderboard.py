"""Reproducible private leaderboard bundles; no model-derived draft scoring as gold."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .core import DATASETS, load_plan, read_jsonl, sha256, validate_source, write_json
from .release import finish_bundle, valid_release_id
from .review import validated_reviews

TRACKS = {
    "original_elm": ("Original reference · normalized", "text", True),
    "original_strict": ("Original reference · strict", "text", False),
    "verified_elm": ("Audio-reviewed wording · normalized", "verified_text", True),
    "verified_strict": ("Audio-reviewed wording · strict", "verified_text", False),
    "ideal_elm": ("Audio-reviewed ideal · normalized", "ideal_text", True),
    "ideal_strict": ("Audio-reviewed ideal · strict", "ideal_text", False),
}


def additional_predictions(rows, predictions):
    """Exact frozen membership and audio identity; no text-match or ordinal alignment."""
    by_id = {r["id"]: r for r in rows}
    names = {c["name"] for r in rows for c in r["candidates"]}
    additions, sources = defaultdict(list), []
    for name, path in predictions:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in names:
            raise ValueError("Submission model names must be distinct filename-safe labels")
        names.add(name)
        before = sha256(path)
        seen = set()
        for prediction in read_jsonl(path):
            validate_source(prediction)
            identifier = prediction["id"]
            if identifier not in by_id or identifier in seen:
                raise ValueError(
                    "Submission IDs must exactly match the frozen plan, without duplicates"
                )
            row = by_id[identifier]
            for key in (
                "dataset",
                "ordinal",
                "text",
                "audio_sha256",
                "preparation_id",
                "actual_duration",
            ):
                if prediction[key] != row[key]:
                    raise ValueError(f"Submission {key} mismatch: {identifier}")
            if not isinstance(prediction.get("pred_text"), str):
                raise ValueError(
                    "Submission must provide text predictions, including explicit empty strings"
                )
            additions[identifier].append({"name": name, "text": prediction["pred_text"]})
            seen.add(identifier)
        if seen != set(by_id) or before != sha256(path):
            raise ValueError("Incomplete or changed submission")
        sources.append({"name": name, "predictions_sha256": before})
    return additions, sources


def calculate_panel(rows, tracks, additions):
    from gqasr.arabic_benchmark.scoring import counts, rates

    groups = defaultdict(Counter)
    for row in rows:
        for candidate in [*row["candidates"], *additions.get(row["id"], [])]:
            for track in tracks:
                _, field, normalized = TRACKS[track]
                groups[(candidate["name"], track, row["dataset"])].update(
                    counts(row[field], candidate["text"], normalized=normalized)
                )
    result = [
        {"model": model, "track": track, "dataset": ds, **rates(value)}
        for (model, track, ds), value in sorted(groups.items())
    ]
    for model, track in sorted({(r["model"], r["track"]) for r in result}):
        subset = [r for r in result if r["model"] == model and r["track"] == track]
        complete = {r["dataset"] for r in subset} == set(DATASETS)
        result.append(
            {
                "model": model,
                "track": track,
                "dataset": "macro",
                "datasets": len(subset),
                "utterances": sum(r["utterances"] for r in subset),
                "wer": sum(r["wer"] for r in subset) / 6 if complete else None,
                "cer": sum(r["cer"] for r in subset) / 6 if complete else None,
            }
        )
    return result


def build_leaderboard(
    plan_root, output, *, reviewed=None, predictions=(), release_id="v4-board-001"
):
    from gqasr.arabic_benchmark import scoring

    valid_release_id(release_id)
    plan, rows = load_plan(plan_root)
    eligible = validated_reviews(rows, reviewed)[1] if reviewed else []
    additions, sources = additional_predictions(rows, predictions)
    cohorts = sorted({c for r in rows for c in r["cohorts"]})
    panels = []
    for cohort in cohorts:
        selected = [r for r in rows if cohort in r["cohorts"]]
        verified = [r for r in eligible if cohort in r["cohorts"]]
        panels.append(
            {
                "id": f"original-{cohort}",
                "label": f"Original references · {cohort} cohort",
                "cohort": cohort,
                "reference_status": "original",
                "selected_clips": len(selected),
                "scored_clips": len(selected),
                "full_suite": cohort == "full"
                and len(selected) == sum(plan["input_counts"].values()),
                "scores": calculate_panel(selected, ("original_elm", "original_strict"), additions),
            }
        )
        if verified:
            panels.append(
                {
                    "id": f"reviewed-{cohort}",
                    "label": f"Audio-reviewed comparison · {cohort} cohort",
                    "cohort": cohort,
                    "reference_status": "single_reviewer_attested",
                    "selected_clips": len(selected),
                    "scored_clips": len(verified),
                    "full_suite": cohort == "full"
                    and len(verified) == sum(plan["input_counts"].values()),
                    "scores": calculate_panel(verified, tuple(TRACKS), additions),
                }
            )
    payload = {
        "schema_version": 1,
        "release_id": release_id,
        "manifest_digest": plan["manifest_digest"],
        "policy_sha256": plan["policy_sha256"],
        "selected_clips": len(rows),
        "source_clips": sum(plan["input_counts"].values()),
        "verified_clips": len(eligible),
        "duration_hours": sum(r["actual_duration"] for r in rows) / 3600,
        "datasets": list(DATASETS),
        "tracks": {k: v[0] for k, v in TRACKS.items()},
        "panels": panels,
        "prediction_sources": [
            {k: s[k] for k in ("name", "predictions_sha256")} for s in plan["sources"]
        ]
        + sources,
        "reviewed_manifest_sha256": sha256(reviewed) if reviewed else None,
        "scorer_sha256": sha256(scoring.__file__),
        "independently_audited": False,
        "limitations": [
            "Private research comparison, not an official Open Universal Arabic ASR leaderboard rank.",
            "QASR/GQASR outputs informed reference curation; this is not a model-independent test-set audit.",
            "Original and reviewed scores are paired only within the same panel; do not compare different subsets as improvement.",
            "Diagnostic cohorts are outcome-selected. Incomplete six-dataset macros are absent, not zero.",
            "Strict WER/CER measures whole-transcript agreement, not isolated punctuation or diacritic quality.",
            "Training overlap and independent reference review remain unaudited. A fresh holdout is needed for general-quality claims.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-board-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        write_json(stage / "leaderboard.json", payload)
        encoded = (
            json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
        )
        template = Path(__file__).with_name("leaderboard.html").read_text()
        (stage / "index.html").write_text(template.replace("__DATA_JSON__", encoded))
        (stage / "README.md").write_text("""---
title: Arabic ASR Benchmark
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
---
# Arabic ASR Benchmark

Private research comparison generated from checksum-bound source predictions and frozen annotation plans. The static page embeds its data and requires no external service. Normalized and strict WER/CER are separated by reference track and cohort. Reviewed comparisons use the identical attested subset for every model and reference track; LLM drafts are never treated as verified references. A missing six-dataset macro remains unavailable.

`leaderboard.json` contains metric counts, coverage and source/scorer digests. `release.json` enumerates exactly the files allowed for upload. This is not an independently audited public ranking or an official ELM rank. See the limitations displayed in the page.
""")
        return finish_bundle(
            stage,
            output,
            {
                "repo_type": "space",
                "release_id": release_id,
                "manifest_digest": plan["manifest_digest"],
                "selected_clips": len(rows),
                "verified_clips": len(eligible),
            },
        )
