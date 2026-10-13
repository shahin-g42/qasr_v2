"""Scored reports and paired comparisons; no model dependencies."""

from __future__ import annotations

import html
import random
import statistics
from collections import defaultdict

from .io import environment, fresh_directory, positive_number, read_json, write_json, write_jsonl
from .metrics import POLICY, SCORER_VERSION, aggregate, normalize, score_pair

SCHEMA = "qasr-lab-report-v1"


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def score_predictions(panel, samples, predictions, model):
    by_id = {}
    for row in predictions:
        if row.get("id") in by_id:
            raise ValueError(f"Duplicate prediction id: {row.get('id')}")
        by_id[row.get("id")] = row
    expected = {row["id"] for row in samples}
    if set(by_id) != expected:
        raise ValueError(
            f"Prediction IDs must exactly match the panel; "
            f"missing={sorted(expected - set(by_id))[:10]}, "
            f"extra={list(set(by_id) - expected)[:10]}"
        )
    scored = []
    for sample in samples:
        prediction = by_id[sample["id"]]
        status = prediction.get("status", "ok")
        if status not in ("ok", "error"):
            raise ValueError(f"Invalid prediction status: {status}")
        hypothesis = prediction.get("hypothesis")
        if status == "ok" and not isinstance(hypothesis, str):
            raise ValueError("Successful predictions need a string hypothesis")
        duration = positive_number(
            prediction.get("audio_seconds", sample["duration"]), "audio_seconds"
        )
        row = {
            **sample,
            "status": status,
            "audio_seconds": duration,
            "is_speech": bool(sample["text"].strip()),
            "hypothesis": hypothesis if status == "ok" else None,
            "error": prediction.get("error") if status == "error" else None,
            "hit_token_limit": bool(prediction.get("hit_token_limit", False)),
            "token_limit_observable": bool(
                prediction.get("token_limit_observable", "hit_token_limit" in prediction)
            ),
        }
        if status == "ok":
            row["scores"] = score_pair(sample["text"], hypothesis, sample["language"])
            stripped = normalize(hypothesis, sample["language"], "strict")
            row.update(
                non_speech_output=bool(stripped),
                output_words=len(stripped.split()),
                output_characters=len(stripped.replace(" ", "")),
            )
            if prediction.get("latency_seconds") is not None:
                row["latency_seconds"] = positive_number(
                    prediction["latency_seconds"], "latency_seconds"
                )
        scored.append(row)
    languages = sorted({row["language"] for row in scored})
    slices = sorted({tag for row in scored for tag in row["tags"]})
    measured = [row for row in scored if row["status"] == "ok" and "latency_seconds" in row]
    latencies = [row["latency_seconds"] for row in measured]
    performance = {
        "timed_samples": len(measured),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "rtf_corpus": (
            sum(latencies) / sum(row["audio_seconds"] for row in measured) if measured else None
        ),
    }
    return {
        "schema": SCHEMA,
        "scorer_version": SCORER_VERSION,
        "normalization": POLICY,
        "panel_sha256": panel["panel_sha256"],
        "panel_provenance": panel.get("provenance", {}),
        "model": model,
        "environment": environment(),
        "complete": all(row["status"] == "ok" for row in scored),
        "token_limit_samples": sum(row["hit_token_limit"] for row in scored),
        "token_limit_unobserved_samples": sum(not row["token_limit_observable"] for row in scored),
        "overall": aggregate(scored),
        "languages": {
            lang: aggregate([r for r in scored if r["language"] == lang], lang)
            for lang in languages
        },
        "slices": {tag: aggregate([r for r in scored if tag in r["tags"]]) for tag in slices},
        "performance": performance,
        "samples": scored,
    }


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def report_html(report):
    rows = []
    for language, metrics in report["languages"].items():
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    language,
                    metrics["samples"],
                    _percent(metrics["strict"]["wer"]),
                    _percent(metrics["recognition"]["wer"]),
                    _percent(metrics["strict"]["cer"]),
                    _percent(metrics["recognition"]["cer"]),
                    metrics["failures"],
                )
            )
            + "</tr>"
        )
    errors = sorted(
        (r for r in report["samples"] if r["status"] == "ok" and r["is_speech"]),
        key=lambda r: r["scores"]["strict"]["char"]["errors"],
        reverse=True,
    )[:30]
    details = "".join(
        f"<article><b>{html.escape(row['id'])} · {html.escape(row['language'])}</b>"
        f"<p dir='auto'>Reference: {html.escape(row['text'])}</p>"
        f"<p dir='auto'>Prediction: {html.escape(row['hypothesis'])}</p></article>"
        for row in errors
    )
    status = "Complete" if report["complete"] else "INCOMPLETE — do not rank this run"
    silence = report["overall"]["non_speech"]
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>QASR Lab evaluation</title><style>
body{{font:16px system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#172536}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ccd}}
article{{padding:16px 0;border-bottom:1px solid #dde}}p{{line-height:1.6}}code{{overflow-wrap:anywhere}}
</style><h1>QASR Lab evaluation</h1><p>{status}</p>
<p>Model: {html.escape(str(report["model"].get("id", "unnamed")))}</p>
<p>Panel: <code>{report["panel_sha256"]}</code></p>
<table><tr><th>Language</th><th>Clips</th><th>Strict WER</th><th>Recognition WER</th>
<th>Strict CER</th><th>Recognition CER</th><th>Failures</th></tr>{"".join(rows)}</table>
<p>Chinese/Japanese: use CER. Language-specific metrics should determine decisions;
no mixed-language WER/CER average is used.</p>
<p>Non-speech clips: {silence["samples"]}; false-transcription rate:
{_percent(silence["false_transcription_rate"])}. Known token-limit hits: {report["token_limit_samples"]};
unobserved: {report["token_limit_unobserved_samples"]}.</p>
<h2>Scoring policy</h2><p>{html.escape(POLICY["recognition"])}</p>
<h2>Largest character-error counts</h2>{details}
<p>The JSON report contains every prediction, slice metric, error count and provenance record.</p></html>"""


def save_report(report, output, predictions=None):
    out = fresh_directory(output)
    write_json(out / "report.json", report)
    if predictions is not None:
        write_jsonl(out / "predictions.jsonl", predictions)
    (out / "report.html").write_text(report_html(report), encoding="utf-8")
    return out


def paired_interval(base, candidate, *, language, mode, iterations, seed):
    unit = "char" if language in ("zh", "ja") else "word"
    left = {row["id"]: row for row in base if row["language"] == language and row["is_speech"]}
    groups = defaultdict(list)
    for row in candidate:
        if row["id"] not in left:
            continue
        reference = left[row["id"]]
        # Recordings are the resampling unit when annotated; otherwise utterances.
        group = (
            (row.get("source", ""), row["recording_id"])
            if row.get("recording_id")
            else ("utterance", row["id"])
        )
        groups[group].append(
            (
                reference["scores"][mode][unit]["errors"],
                row["scores"][mode][unit]["errors"],
                reference["scores"][mode][unit]["reference_units"],
            )
        )
    totals = [tuple(sum(item[i] for item in group) for i in range(3)) for group in groups.values()]
    denominator = sum(item[2] for item in totals)
    if not denominator:
        return {"delta": None, "ci95": None, "resampling_groups": len(totals)}
    delta = sum(item[1] - item[0] for item in totals) / denominator
    rng = random.Random(seed)
    draws = []
    if len(totals) >= 2:
        for _ in range(iterations):
            selected = rng.choices(totals, k=len(totals))
            denom = sum(item[2] for item in selected)
            if denom:
                draws.append(sum(item[1] - item[0] for item in selected) / denom)
    return {
        "delta": delta,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)] if draws else None,
        "resampling_groups": len(totals),
        "primary_metric": "cer" if unit == "char" else "wer",
    }


def compare_reports(paths, output, *, iterations=1000, seed=42):
    if len(paths) < 2 or iterations < 1:
        raise ValueError("Need at least two reports and positive bootstrap iterations")
    reports = [read_json(path) for path in paths]
    base = reports[0]
    for report in reports:
        if report.get("schema") != SCHEMA or not report["complete"]:
            raise ValueError("Comparison requires complete QASR Lab reports")
        for key in ("panel_sha256", "scorer_version", "normalization"):
            if report[key] != base[key]:
                raise ValueError(f"Cannot compare reports with different {key}")
        if {r["id"] for r in report["samples"]} != {r["id"] for r in base["samples"]}:
            raise ValueError("Reports contain different sample IDs")
    candidates = []
    for report in reports[1:]:
        comparison = {
            "model": report["model"],
            "token_limit_samples": report["token_limit_samples"],
            "languages": {},
            "latency_comparable": (
                bool(base["model"].get("timing_protocol"))
                and base["model"].get("timing_protocol") == report["model"].get("timing_protocol")
                and base["model"].get("hardware") == report["model"].get("hardware")
                and base["model"].get("runtime_packages") == report["model"].get("runtime_packages")
                and base["model"].get("decode_protocol") == report["model"].get("decode_protocol")
                and base["performance"]["timed_samples"] == len(base["samples"])
                and report["performance"]["timed_samples"] == len(report["samples"])
            ),
        }
        for language in base["languages"]:
            comparison["languages"][language] = {
                mode: paired_interval(
                    base["samples"],
                    report["samples"],
                    language=language,
                    mode=mode,
                    iterations=iterations,
                    seed=seed,
                )
                for mode in ("strict", "recognition")
            }
        comparison["non_speech"] = report["overall"]["non_speech"]
        baseline_latency = base["performance"]["latency_mean_s"]
        candidate_latency = report["performance"]["latency_mean_s"]
        comparison["mean_latency_speedup"] = (
            baseline_latency / candidate_latency
            if comparison["latency_comparable"] and baseline_latency and candidate_latency
            else None
        )
        candidates.append(comparison)
    result = {
        "schema": "qasr-lab-comparison-v1",
        "baseline": base["model"],
        "panel_sha256": base["panel_sha256"],
        "scorer_version": base["scorer_version"],
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "confidence": 0.95,
            "unit": "recording when supplied, otherwise utterance",
            "multiple_comparisons_corrected": False,
        },
        "interpretation": "Negative candidate-minus-baseline error-rate deltas favor the candidate.",
        "candidates": candidates,
        "decision": "Review per-language, non-speech and latency results.",
    }
    out = fresh_directory(output)
    write_json(out / "comparison.json", result)
    lines = [
        "<!doctype html><meta charset='utf-8'><title>QASR Lab comparison</title>",
        "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto}td,th{padding:12px;text-align:left}</style>",
        "<h1>Paired model comparison</h1><p>Negative deltas favor the candidate. "
        "Values below are percentage points, with paired 95% bootstrap intervals.</p>",
        "<table><tr><th>Candidate</th><th>Language</th><th>Metric</th><th>Strict delta</th><th>Recognition delta</th></tr>",
    ]
    for candidate in candidates:
        for language, modes in candidate["languages"].items():
            cells = [
                candidate["model"].get("id", "?"),
                language,
                modes["recognition"].get("primary_metric", "n/a"),
            ]
            for mode in ("strict", "recognition"):
                value = modes[mode]
                interval = value["ci95"]
                cells.append(
                    "n/a"
                    if value["delta"] is None
                    else (
                        f"{100 * value['delta']:+.2f}"
                        + (
                            f" [{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}]"
                            if interval
                            else " [insufficient groups]"
                        )
                    )
                )
            lines.append(
                "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"
            )
    lines.append(
        "</table><p>Inspect comparison.json for non-speech metrics and timing comparability. "
        "Intervals are not corrected for multiple comparisons.</p>"
    )
    (out / "comparison.html").write_text("\n".join(lines), encoding="utf-8")
    return result
