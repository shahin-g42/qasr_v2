import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qasr_lab.cli import import_legacy, main
from qasr_lab.io import read_json, write_json, write_jsonl
from qasr_lab.metrics import edit_counts, normalize
from qasr_lab.panel import canonical_sample, load_panel, prepare_panel, save_panel
from qasr_lab.report import compare_reports, report_html, save_report, score_predictions


def sample(index, text="hello world", language="en", **extra):
    return canonical_sample(
        {
            "id": str(index),
            "audio_filepath": f"/data/clip{index}.wav",
            "text": text,
            "language": language,
            "duration": 3,
            **extra,
        },
        base="/",
    )


def report(samples, hypotheses, model_id="test"):
    return score_predictions(
        {"panel_sha256": "same"},
        samples,
        [
            {"id": row["id"], "hypothesis": hyp}
            for row, hyp in zip(samples, hypotheses, strict=True)
        ],
        {"id": model_id},
    )


def test_lightweight_import_and_help_without_site_packages():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "qasr_lab", "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "audit-transfer" in result.stdout


def test_normalization_preserves_indic_marks_and_arabic_letter_distinctions():
    assert normalize("മലയാളം", "ml", "recognition") == "മലയാളം"
    assert normalize("हिंदी", "hi", "recognition") == "हिंदी"
    assert normalize("أَهْلاً ١٢", "ar", "recognition") == "أهلا 12"
    assert normalize("Hello, DON'T!", "en", "recognition") == "hello dont"
    assert normalize("Hello, DON'T!", "en", "strict") == "Hello, DON'T!"


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("a b c", "a x c", (1, 0, 0)),
        ("a b c", "a c", (0, 1, 0)),
        ("a c", "a b c", (0, 0, 1)),
        ("", "a b", (0, 0, 2)),
    ],
)
def test_alignment_counts(reference, hypothesis, expected):
    counts = edit_counts(reference.split(), hypothesis.split())
    assert tuple(counts[key] for key in ("substitutions", "deletions", "insertions")) == expected
    assert counts["errors"] == sum(expected)


def test_frozen_selection_is_order_independent_and_detects_tampering(tmp_path):
    rows = [sample(index) for index in range(30)]
    left, right = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(left, rows)
    write_jsonl(right, rows[::-1])
    a = prepare_panel([left], tmp_path / "a", per_stratum=5)
    b = prepare_panel([right], tmp_path / "b", per_stratum=5)
    assert a["panel_sha256"] == b["panel_sha256"]
    assert a["samples"] == 5
    file = tmp_path / "a" / "samples.jsonl"
    file.write_text(file.read_text().replace("hello", "changed"))
    with pytest.raises(ValueError, match="changed"):
        load_panel(tmp_path / "a")


def test_speaker_leakage_and_duplicate_ids_rejected(tmp_path):
    selected = tmp_path / "selected.jsonl"
    training = tmp_path / "train.jsonl"
    write_jsonl(selected, [sample(1, source="corpus", speaker_id="person")])
    write_jsonl(training, [sample(2, source="corpus", speaker_id="person")])
    with pytest.raises(ValueError, match="overlap"):
        prepare_panel([selected], tmp_path / "panel", exclude_manifests=[training])
    with pytest.raises(ValueError, match="Duplicate"):
        save_panel([sample(1), sample(1)], tmp_path / "duplicate")


def test_silence_is_scored_separately_and_chinese_uses_cer():
    result = report(
        [sample(1), sample(2, text=""), sample(3, text="你好世界", language="zh")],
        ["hello world", "invented words", "你好世"],
    )
    assert result["languages"]["en"]["strict"]["wer"] == 0
    assert result["overall"]["non_speech"]["false_transcription_rate"] == 1
    assert result["overall"]["non_speech"]["generated_words_per_minute"] == 40
    assert result["languages"]["zh"]["primary_metric"] == "cer"
    assert result["languages"]["zh"]["strict"]["cer"] == 0.25
    assert result["token_limit_unobserved_samples"] == 3


def test_predictions_must_cover_panel_and_errors_are_not_hidden(tmp_path):
    samples = [sample(1), sample(2)]
    with pytest.raises(ValueError, match="exactly match"):
        score_predictions({"panel_sha256": "same"}, samples, [{"id": "1", "hypothesis": ""}], {})
    result = score_predictions(
        {"panel_sha256": "same"},
        samples,
        [{"id": "1", "hypothesis": ""}, {"id": "2", "status": "error", "error": "OOM"}],
        {},
    )
    assert not result["complete"]
    assert result["overall"]["failures"] == 1
    save_report(result, tmp_path / "run")
    with pytest.raises(ValueError, match="complete"):
        compare_reports([tmp_path / "run/report.json"] * 2, tmp_path / "comparison")


def test_paired_comparison_and_timing_provenance(tmp_path):
    samples = [sample(1), sample(2), sample(3, text="")]
    base = report(samples, ["hello wrong", "hello wrong", "noise"], "baseline")
    better = report(samples, ["hello world", "hello world", ""], "better")
    save_report(base, tmp_path / "base")
    save_report(better, tmp_path / "better")
    result = compare_reports(
        [tmp_path / "base/report.json", tmp_path / "better/report.json"],
        tmp_path / "comparison",
        iterations=100,
    )
    candidate = result["candidates"][0]
    assert candidate["languages"]["en"]["strict"]["delta"] == -0.5
    assert candidate["languages"]["en"]["strict"]["ci95"] == [-0.5, -0.5]
    assert candidate["mean_latency_speedup"] is None
    assert candidate["non_speech"]["false_transcription_rate"] == 0
    assert (tmp_path / "comparison/comparison.html").is_file()


def test_mismatched_panels_and_existing_outputs_rejected(tmp_path):
    left = report([sample(1)], ["hello world"])
    right = {**left, "panel_sha256": "different"}
    save_report(left, tmp_path / "left")
    save_report(right, tmp_path / "right")
    with pytest.raises(ValueError, match="panel_sha256"):
        compare_reports(
            [tmp_path / "left/report.json", tmp_path / "right/report.json"], tmp_path / "compare"
        )
    with pytest.raises(FileExistsError):
        save_report(left, tmp_path / "left")


def test_html_escapes_transcript():
    result = report([sample(1, text="<script>alert(1)</script>")], ["<img src=x>"])
    output = report_html(result)
    assert "<script>" not in output and "<img" not in output
    assert "&lt;script&gt;" in output


def test_legacy_import_does_not_claim_verified_timing_or_holdout(tmp_path):
    source = tmp_path / "legacy.json"
    write_json(
        source,
        {
            "language": "en",
            "model": "/unknown",
            "manifest": "train.jsonl",
            "modes": {
                "offline": {
                    "results": [
                        {
                            "audio_filepath": "/a.wav",
                            "reference": "hello",
                            "hypothesis": "hello",
                            "audio_seconds": 2,
                            "latency_seconds": 0.1,
                        }
                    ]
                }
            },
        },
    )
    import_legacy(source, tmp_path / "import")
    result = read_json(tmp_path / "import/report/report.json")
    assert result["model"]["timing_protocol"] is None
    assert result["panel_provenance"]["leakage"]["status"] == "not_checked"
    assert (
        main(
            [
                "score",
                "--panel",
                str(tmp_path / "import/panel"),
                "--predictions",
                str(tmp_path / "import/report/predictions.jsonl"),
                "--model-id",
                "rescored",
                "--output",
                str(tmp_path / "rescored"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "rescored/report.json").read_text())["complete"]
