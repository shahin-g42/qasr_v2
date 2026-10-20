"""Reproducible private leaderboard bundles; no model-derived draft scoring as gold."""

from __future__ import annotations

import csv
import html
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .core import DATASETS, load_plan, read_jsonl, sha256, validate_source, write_json
from .leaderboard_inputs import annotation_drafts, dataset_pin, transcript_examples
from .leaderboard_reference import reference_guide, reference_guide_markdown
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
DRAFT_TRACKS = {
    "draft_elm": ("Enhanced draft · normalized", "ideal_text_draft", True),
    "draft_strict": ("Enhanced draft · strict", "ideal_text_draft", False),
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
            validate_source(prediction, allow_empty_prediction=True)
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
                _, field, normalized = (TRACKS | DRAFT_TRACKS)[track]
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
                **{
                    metric: sum(r[metric] for r in subset) / 6
                    if complete and all(r[metric] is not None for r in subset)
                    else None
                    for metric in ("wer", "cer")
                },
            }
        )
    return result


def build_leaderboard(
    plan_root,
    output,
    *,
    reviewed=None,
    predictions=(),
    release_id="v4-board-001",
    title="Arabic ASR Benchmark",
    annotations=None,
    dataset_repo=None,
    dataset_revision=None,
    submissions=None,
    timing_runs=(),
    latency_root=None,
    submissions_only=False,
    comparison_protocol=None,
):
    from gqasr.arabic_benchmark import scoring

    valid_release_id(release_id)
    if (
        not isinstance(title, str)
        or not title.strip()
        or len(title) > 120
        or any(unicodedata.category(c) in {"Cc", "Zl", "Zp"} for c in title)
    ):
        raise ValueError("Space title must be a nonempty single line of at most 120 characters")
    title = title.strip()
    plan, rows = load_plan(plan_root)
    model_details, submission_sha256 = {}, None
    if submissions:
        from .submissions import read_submissions

        public_predictions, model_details, submission_sha256 = read_submissions(submissions, plan)
        predictions = [*predictions, *public_predictions]
    if submissions_only and (not submissions or not model_details):
        raise ValueError("Fresh-only ranking requires complete, provenance-bound submissions")
    from .benchmark_performance import read_performance, performance_csv

    performance = {name: detail["performance"] for name, detail in model_details.items()}
    for name, folder in (() if submissions_only else timing_runs):
        source = next((s for s in plan["sources"] if s["name"] == name), None)
        if source is None or name in performance:
            raise ValueError(f"Unknown or duplicate timing source: {name}")
        performance[name] = read_performance(folder, source["predictions_sha256"], expected_run=source["run"])
    from .latency import read_latency_profiles, source_binding, write_latency_csv, write_request_csv
    latency = None
    if latency_root:
        bindings = {s["name"]: source_binding(s["run"]) for s in plan["sources"] if (s.get("run") or {}).get("checkpoint")}
        bindings.update({name: detail["latency_binding"] for name, detail in model_details.items()})
        latency = read_latency_profiles(latency_root, plan, rows, bindings)
    pin = dataset_pin(dataset_repo, dataset_revision)
    drafts, annotation_rows, annotation_source = (
        annotation_drafts(annotations, plan, rows) if annotations else ([], [], None)
    )
    eligible = validated_reviews(rows, reviewed)[1] if reviewed else []
    additions, sources = additional_predictions(rows, predictions)
    if submissions_only:
        if {s["name"] for s in sources} != set(model_details):
            raise ValueError("Fresh-only ranking cannot include unbound prediction files")
        # Keep the frozen plan and annotation source untouched. Historical outputs
        # remain curation provenance, but only fresh runs participate in this release.
        rows = [{**r, "candidates": []} for r in rows]
        drafts = [{**r, "candidates": []} for r in drafts]
        eligible = [{**r, "candidates": []} for r in eligible]
        if latency and set(latency["models"]) != set(model_details):
            raise ValueError("Fresh-only latency must cover exactly the ranked models")
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
        proposed = [r for r in drafts if cohort in r["cohorts"]]
        if proposed:
            panels.append(
                {
                    "id": f"enhanced-{cohort}",
                    "label": f"Enhanced drafts · matched {cohort} cohort",
                    "cohort": cohort,
                    "reference_status": "candidate_assisted_unverified",
                    "selected_clips": len(selected),
                    "scored_clips": len(proposed),
                    "excluded_clips": len(selected) - len(proposed),
                    "full_suite": cohort == "full"
                    and len(proposed) == sum(plan["input_counts"].values()),
                    "scores": calculate_panel(
                        proposed, ("original_elm", "original_strict", *DRAFT_TRACKS), additions
                    ),
                }
            )
    payload = {
        "schema_version": 2,
        "title": title,
        "release_id": release_id,
        "ranking_source": "fresh_submissions_only" if submissions_only else "historical_and_submissions",
        "manifest_digest": plan["manifest_digest"],
        "policy_sha256": plan["policy_sha256"],
        "selected_clips": len(rows),
        "source_clips": sum(plan["input_counts"].values()),
        "verified_clips": len(eligible),
        "duration_hours": sum(r["actual_duration"] for r in rows) / 3600,
        "datasets": list(DATASETS),
        "tracks": {k: v[0] for k, v in (TRACKS | (DRAFT_TRACKS if annotations else {})).items()},
        "dataset_release": pin,
        "annotation_source": annotation_source,
        "models": sorted(
            {c["name"] for r in rows for c in [*r["candidates"], *additions.get(r["id"], [])]}
        ),
        "model_details": model_details,
        "performance": performance,
        "latency": latency,
        "submission_manifest_sha256": submission_sha256,
        "dataset_summary": [
            {
                "dataset": ds,
                "clips": sum(r["dataset"] == ds for r in rows),
                "hours": sum(r["actual_duration"] for r in rows if r["dataset"] == ds) / 3600,
            }
            for ds in DATASETS
        ],
        "examples": transcript_examples(rows, drafts, annotation_rows, additions),
        "example_sampling": "Three SHA256-ranked IDs per dataset, independent of scores; non-speculative completed drafts when available. Illustrations, not an accuracy audit.",
        "reference_guide": reference_guide(drafts),
        "panels": panels,
        "prediction_sources": [
            {k: s[k] for k in ("name", "predictions_sha256")} for s in plan["sources"]
        ] * (not submissions_only)
        + sources,
        "reference_curation_sources": [
            {k: s[k] for k in ("name", "predictions_sha256")} for s in plan["sources"]
        ],
        "reviewed_manifest_sha256": sha256(reviewed) if reviewed else None,
        "scorer_sha256": sha256(scoring.__file__),
        "builder_sha256": sha256(__file__),
        "independently_audited": False,
        "limitations": [
            "Private research comparison, not an official Open Universal Arabic ASR leaderboard rank.",
            "QASR/GQASR outputs informed reference curation; this is not a model-independent test-set audit.",
            "Enhanced draft scores measure agreement with candidate-assisted ideal_text_draft, not verified transcription accuracy. Speculative drafts are excluded; paired original scores use exactly the same clips.",
            "Original and reviewed scores are paired only within the same panel; do not compare different subsets as improvement.",
            "Diagnostic cohorts are outcome-selected. Incomplete six-dataset macros are absent, not zero.",
            "Strict WER/CER measures whole-transcript agreement, not isolated punctuation or diacritic quality.",
            "Training overlap and independent reference review remain unaudited. A fresh holdout is needed for general-quality claims.",
        ],
    }
    protocol = json.loads(Path(comparison_protocol).read_text()) if comparison_protocol else None
    if protocol and (not submissions_only or set(protocol["models"]) != set(payload["models"])):
        raise ValueError("Comparison protocol must describe exactly the fresh ranked models")
    payload["comparison_protocol"] = protocol
    payload["comparison_protocol_sha256"] = sha256(comparison_protocol) if comparison_protocol else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-board-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        write_json(stage / "leaderboard.json", payload)
        if protocol:
            (stage / "protocol.json").write_bytes(Path(comparison_protocol).read_bytes())
        performance_csv(stage / "performance.csv", payload["models"], performance)
        if latency:
            write_json(stage / "latency.json", latency)
            write_latency_csv(stage / "latency.csv", latency)
            write_request_csv(stage / "latency-requests.csv", latency_root, latency)
        with (stage / "scores.csv").open("w", newline="") as handle:
            fields = [
                "panel",
                "reference_status",
                "model",
                "track",
                "dataset",
                "utterances",
                "wer",
                "cer",
                "word_errors",
                "words",
                "character_errors",
                "characters",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for panel in panels:
                for score in panel["scores"]:
                    writer.writerow(
                        {
                            "panel": panel["id"],
                            "reference_status": panel["reference_status"],
                            **score,
                        }
                    )
        encoded = (
            json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
        )
        template = Path(__file__).with_name("leaderboard.html").read_text()
        values = {"SPACE_TITLE": html.escape(title), "DATA_JSON": encoded}
        rendered = re.sub(r"__(SPACE_TITLE|DATA_JSON)__", lambda m: values[m[1]], template)
        (stage / "index.html").write_text(rendered)
        for filename in ("leaderboard.css", "leaderboard.js", "latency.js"):
            (stage / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
        (stage / "README.md").write_text(f"""---
title: {json.dumps(title, ensure_ascii=True)}
emoji: 🎙️
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
---
<h1>{html.escape(title)}</h1>

Arabic speech recognition, measured with context. Explore model rankings, six dataset comparisons, paired reference tracks, and Arabic transcript examples in an accessible, responsive static dashboard. No inference server, external scripts, font requests, or tokens are needed by the page.

{"This release ranks fresh predictions only. Historical QASR/GQASR outputs remain reference-curation provenance. The frozen protocol.json records the common accuracy configuration and model output settings, including Jais diacritics." if submissions_only else "This release includes historical source predictions and completed additional model submissions."}

The default ranking uses **original source references**. The optional **enhanced comparison** evaluates `ideal_text_draft` alongside originals on the identical eligible clips, excluding every speculative or incomplete draft. These references were informed by QASR/GQASR outputs; their scores are exploratory agreement measurements, not independent evidence of model accuracy. Final reviewed tracks appear only after listening attestations are imported.

The six-dataset macro gives each dataset equal weight. Within each dataset, WER/CER are total edit distance divided by total reference length; CER includes spaces. Normalized scoring uses the pinned ELM protocol. Strict scoring preserves spelling, punctuation and diacritics and is not an isolated formatting-quality measure. Missing denominators or dataset coverage produce unavailable values, never zeros.

Dataset: {pin["url"] if pin else "No published dataset revision supplied."}

The page records the immutable dataset revision, annotation bundle, prediction sources, reference policy and scorer hashes. The transcript browser uses stable ID hashes; the separate editorial guide uses explicitly curated, exact-text-bound examples. Full-set ranking retains the original dataset membership, including any duplicate audio; enhanced coverage and exclusions are explicit.

{reference_guide_markdown(payload["reference_guide"])}

The speed table reports mean batch service time when recorded, amortized worker compute per output, and end-to-end throughput including startup and recorded attempts. These are different measurements: compute per output is not request latency. Timing covers the complete source run, including every decoding mode, independently of accuracy filters. Compare the recorded hardware, replicas, batch sizes, output settings and OOM splits. Older runs have no individual batch measurements. `performance.csv` exports the timing values and configurations.

When supplied, the separate **single-request latency profile** measures resident 16 kHz waveform to final text, with batch 1, concurrency 1, BF16, one CPU thread and the same reserved GPU. All models receive the same duration-stratified clips, order, warm-up count and repetitions. First request and warm-up are excluded from steady-state percentiles. Network upload, file I/O, server queueing and streaming first-token latency are outside this measurement. Model load is measured separately after checksum validation, so it may benefit from the OS cache. Native decoding/token limits are retained and recorded; partial, empty or refused outputs and failures are visible. Input-preparation and inference phase boundaries vary by adapter; total request time is the comparable measure. GPU clocks are observed, not locked. CPU contention remains a possible source of variation.

`latency.csv` exports dataset/duration summaries, `latency-requests.csv` contains every measured, warm-up and first-request observation, and `latency.json` records distributions, protocol, hardware and provenance. These files appear only after a profile is imported; missing timings are never inferred from throughput. A sampled profile is not a full-suite measurement. Historical GQASR scores stay separate from the fresh step-7,500 checkpoint.

`leaderboard.json` contains metric counts, coverage, examples and provenance. `scores.csv` exports every panel (rates are fractions; multiply by 100 for percentages). `release.json` enumerates exactly the files allowed for upload. Private visibility is enforced by the publisher. This is not an independently audited public ranking or an official ELM rank.
""")
        return finish_bundle(
            stage,
            output,
            {
                "repo_type": "space",
                "title": title,
                "release_id": release_id,
                "manifest_digest": plan["manifest_digest"],
                "selected_clips": len(rows),
                "verified_clips": len(eligible),
                "dataset_release": pin,
                "annotations_bundle_sha256": annotation_source["bundle_sha256"]
                if annotation_source
                else None,
                "submission_manifest_sha256": submission_sha256,
                "ranking_source": payload["ranking_source"],
                "comparison_protocol_sha256": payload["comparison_protocol_sha256"],
                "latency_profile_file_sha256": latency["profile_file_sha256"] if latency else None,
            },
        )
