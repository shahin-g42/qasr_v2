import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audm import prepare_non_speech as prep
from audm.robustness import NonSpeechPool, validate_negative_record


def source_row(identity=1, kind="noise", seconds=2, sr=16000):
    wave = np.random.default_rng(identity).normal(0, 0.05, round(seconds * sr)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, wave, sr, format="FLAC", subtype="PCM_16")
    return dict(
        audio={"bytes": buffer.getvalue(), "path": "ignored.flac"},
        freesound_id=identity,
        labels=["Rain"] if kind == "noise" else ["Music", "Piano"],
        title="test clip",
        description="",
        tags=[],
        license="https://creativecommons.org/licenses/by/4.0/",
        uploader="example",
        freesound_url=f"https://freesound.org/s/{identity}/",
        source_partition="dev",
    )


@pytest.fixture
def snapshot(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "data"
    raw = root / "raw" / "FSD50K-16k" / prep.REVISION
    (raw / "data").mkdir(parents=True)
    entries = []
    for index, split in enumerate(prep.SPLITS):
        path = raw / "data" / f"{split}-00000-of-00001.parquet"
        rows = [source_row(index * 10 + 1), source_row(index * 10 + 2, kind="music", seconds=2.5)]
        if split == "train":
            duplicate = source_row(1)
            duplicate["freesound_id"] = 999  # Same held-out waveform, different source ID.
            rows.extend([duplicate, {**source_row(998), "labels": ["Rain", "Speech"]}])
        pq.write_table(pa.Table.from_pylist(rows), path)
        entries.append(
            dict(
                path=str(path.relative_to(raw)),
                size=path.stat().st_size,
                sha256=prep.sha256(path),
                rows=len(rows),
            )
        )
    prep.dump(raw / "checksums.json", entries)
    prep.dump(raw / "aud_download.json", dict(repo=prep.REPO, revision=prep.REVISION))
    return root, raw


@pytest.mark.parametrize(
    "license,allowed",
    [
        ("https://creativecommons.org/publicdomain/zero/1.0/", True),
        ("http://creativecommons.org/licenses/by/3.0/", True),
        ("https://creativecommons.org/licenses/by-nc/4.0/", False),
        ("https://creativecommons.org/licenses/sampling+/1.0/", False),
        ("https://creativecommons.org.evil.test/licenses/by/4.0/", False),
        ("CC-BY", False),
        (None, False),
    ],
)
def test_license_filter(license, allowed):
    assert prep.permitted_license(license) is allowed


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "noise"),
        ({"labels": ["Music", "Piano"]}, "music"),
        ({"labels": ["Music", "Rain"]}, None),
        ({"labels": ["Music", "Female_singing"]}, None),
        ({"labels": ["Human_voice", "Rain"]}, None),
        ({"labels": ["Human_group_actions"]}, None),
        ({"labels": ["Speech_synthesizer"]}, None),
        ({"labels": ["UnseenLabel"]}, None),
        ({"description": "faint voices in background"}, None),
        ({"tags": ["talking"]}, None),
        ({"tags": "rain"}, None),
    ],
)
def test_conservative_category_filter(changes, expected):
    assert prep.category({**source_row(), **changes})[0] == expected


def test_real_parquet_pipeline_and_manifest_contract(snapshot, tmp_path):
    root, _ = snapshot
    candidates, output = tmp_path / "review", tmp_path / "corpus"
    report = prep.prepare(root, candidates, min_seconds=1, max_seconds=1.5)
    assert report["counts"]["excluded:duplicate_audio"] == 1
    assert report["counts"]["excluded:possible_vocals"] == 1
    rows = list(prep.jsonl(candidates / "candidates.jsonl"))
    assert len(rows) == 9  # 1 noise + 2 music chunks per split; short tails discarded.
    assert all(1 <= r["duration"] <= 1.5 and Path(r["audio_filepath"]).is_file() for r in rows)
    for row in rows:
        with pytest.raises(ValueError, match="empty text"):
            validate_negative_record(row, require_audio_type=True)
    assert "audio/test/" in (candidates / "review.html").read_text()
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "".join(
            json.dumps({"candidate_id": r["candidate_id"], "decision": "accept"}) + "\n"
            for r in rows
        )
    )
    result = prep.finalize(candidates / "candidates.jsonl", decisions, output, silence_per_split=2)
    assert result["counts"]["train:silence"] == 2
    pool = NonSpeechPool(
        dict(probability=0.03, manifests=[str(output / "train.jsonl")]),
        1,
        1.5,
        require_audio_type=True,
    )
    assert pool.scan_audio()["categories"] == {"music": 2, "noise": 1, "silence": 2}
    all_rows = {s: list(prep.jsonl(output / f"{s}.jsonl")) for s in prep.SPLITS}
    hashes = {}
    for split, records in all_rows.items():
        assert {r["audio_type"] for r in records} == {"music", "noise", "silence"}
        hashes[split] = {r["audio_sha256"] for r in records if r["audio_type"] != "silence"}
        for row in records:
            assert validate_negative_record(row, require_audio_type=True) == row["audio_type"]
            wave, sr = sf.read(row["audio_filepath"])
            assert len(wave) / sr == row["duration"]
            assert prep.sha256(row["audio_filepath"]) == row["audio_sha256"]
    assert not hashes["train"] & (hashes["test"] | hashes["validation"])
    # Corpus copies survive later modifications to the review files.
    original = next(
        r for r in all_rows[rows[0]["split"]] if r.get("candidate_id") == rows[0]["candidate_id"]
    )
    Path(rows[0]["audio_filepath"]).write_bytes(b"changed")
    assert prep.sha256(original["audio_filepath"]) == original["audio_sha256"]
    with pytest.raises(ValueError, match="audio changed"):
        prep.finalize(candidates / "candidates.jsonl", decisions, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))
    with pytest.raises(FileExistsError):
        prep.prepare(root, candidates)


def test_checksum_and_missing_shard_fail(snapshot):
    _, raw = snapshot
    shard = next((raw / "data").glob("*.parquet"))
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        prep.verify_snapshot(raw)
    shard.unlink()
    with pytest.raises(ValueError, match="Missing"):
        prep.verify_snapshot(raw)


@pytest.mark.parametrize(
    "decision",
    [
        {"candidate_id": "not-real", "decision": "accept"},
        {"candidate_id": "fsd-1-0", "decision": "pending"},
        {"candidate_id": "fsd-1-0", "decision": "accept", "audio_type": "silence"},
    ],
)
def test_bad_decisions_rejected(snapshot, tmp_path, decision):
    root, _ = snapshot
    review = tmp_path / "review"
    prep.prepare(root, review)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n")
    with pytest.raises(ValueError, match="invalid decision"):
        prep.finalize(review / "candidates.jsonl", decisions, tmp_path / "corpus")


def test_missing_categories_and_mutated_manifest(snapshot, tmp_path):
    root, _ = snapshot
    review = tmp_path / "review"
    prep.prepare(root, review)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("")
    with pytest.raises(ValueError, match="each source split"):
        prep.finalize(review / "candidates.jsonl", decisions, tmp_path / "corpus")
    with (review / "candidates.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="manifest changed"):
        prep.finalize(review / "candidates.jsonl", decisions, tmp_path / "corpus")


def test_audio_schema_and_bad_values():
    with pytest.raises(ValueError, match="embedded"):
        prep.read_audio({"audio": {"path": "/etc/passwd"}})
    with pytest.raises(ValueError, match="16 kHz"):
        prep.read_audio(source_row(sr=8000))
    for low, high in [(0, 30), (1, 31), (float("nan"), 30), (10, 5)]:
        with pytest.raises(ValueError, match="Require"):
            prep.prepare("unused", "unused", min_seconds=low, max_seconds=high)


def test_review_page_escapes_metadata(tmp_path):
    path = tmp_path / "review.html"
    prep.review_page(path, [{"title": "</script><img src=x onerror=alert(1)>"}])
    assert "</script><img" not in path.read_text()
    assert "\\u003c/script>" in path.read_text()


def test_download_uses_pinned_revision_and_resumable_directory(snapshot, monkeypatch):
    root, raw = snapshot
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **kw: calls.append((a, kw))
    )
    assert prep.download(root)["shard_checksums"] == "verified"
    assert calls[0][1]["revision"] == prep.REVISION
    assert calls[0][1]["local_dir"] == str(raw)
    assert "*.py" not in calls[0][1]["allow_patterns"]
    assert prep.download(root)["snapshot"] == str(raw)


def test_metadata_only_finalize_without_review(snapshot, tmp_path):
    root, _ = snapshot
    review = tmp_path / "candidates"
    prep.prepare(root, review)
    output = tmp_path / "corpus"
    result = prep.finalize(
        review / "candidates.jsonl", output=output, metadata_only=True, silence_per_split=2
    )
    assert result["selection_mode"] == "metadata_only"
    assert result["manually_reviewed"] is False
    assert result["decisions_sha256"] is None
    assert not (output / "decisions.jsonl").exists()
    for split in prep.SPLITS:
        rows = list(prep.jsonl(output / f"{split}.jsonl"))
        assert len(rows) == 4
        assert {r["review_status"] for r in rows} == {"metadata_only", "synthetic"}
        assert all(r["text"] == "" for r in rows)
        for row in rows:
            validate_negative_record(row, require_audio_type=True)
    pool = NonSpeechPool(
        dict(probability=0.03, manifests=[str(output / "train.jsonl")]),
        1,
        30,
        require_audio_type=True,
    )
    assert pool.scan_audio()["categories"] == {"noise": 1, "music": 1, "silence": 2}
    with pytest.raises(FileExistsError):
        prep.finalize(review / "candidates.jsonl", output=output, metadata_only=True)
    original = next(prep.jsonl(review / "candidates.jsonl"))
    Path(original["audio_filepath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        prep.finalize(review / "candidates.jsonl", output=tmp_path / "bad", metadata_only=True)
    assert not (tmp_path / "bad").exists()


def test_finalize_requires_explicit_selection_mode(tmp_path):
    with pytest.raises(ValueError, match="either"):
        prep.finalize("missing.jsonl", output=tmp_path / "out")
    with pytest.raises(ValueError, match="either"):
        prep.finalize(
            "missing.jsonl",
            decisions="decisions.jsonl",
            output=tmp_path / "out",
            metadata_only=True,
        )


def test_finalize_wrapper_selects_metadata_mode(tmp_path):
    import os
    import subprocess

    executable = tmp_path / "capture-python"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/non_speech.sh"
    env = dict(os.environ, AUD_PYTHON=str(executable), AUD_NON_SPEECH_ROOT=str(tmp_path))
    for flags in ([], ["--metadata-only"]):
        result = subprocess.run(
            ["bash", str(script), "finalize", *flags],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        args = result.stdout.splitlines()
        assert ("--decisions" in args) == (not flags)
        assert ("--metadata-only" in args) == bool(flags)
        assert str(tmp_path / "corpus-v1") in args
