"""Parallel corpus checks with tiny audio/tokenizer fixtures; no model execution."""

import hashlib
import json
import sqlite3

import numpy as np
import pytest
import soundfile as sf
from gqasr.data import scan
from gqasr.processing import GQASRProcessor
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def setup_scan(tmp_path):
    backend = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "language": 1,
                "Arabic": 2,
                "English": 3,
                "<asr_text>": 4,
                "مرحبا": 5,
                "hello": 6,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    processor = GQASRProcessor.__new__(GQASRProcessor)
    processor.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", additional_special_tokens=["<asr_text>"]
    )
    processor.config = {"eos_id": 7}
    cfg = dict(
        index_path=str(tmp_path / "index.sqlite"),
        audio_root=None,
        audio_path_rewrites={},
        min_duration_seconds=0.1,
        max_duration_seconds=1.0,
        max_target_length=8,
        language_batch_counts={"ar": 1, "en": 1},
        train_manifest={},
        eval_manifest={},
    )
    for split in ("train", "eval"):
        for lang, text in [("ar", "مرحبا"), ("en", "hello")]:
            rows = []
            for i in range(4):
                audio = tmp_path / f"{split}-{lang}-{i}.wav"
                sf.write(audio, np.zeros(3200), 16000)
                rows.append(
                    dict(
                        audio_filepath=str(audio),
                        text=text,
                        duration=0.2,
                        recording_id=f"{split}-{lang}-{i}",
                    )
                )
            rows += [rows[0], dict(rows[1], text="!"), dict(rows[2], text=" ".join([text] * 20))]
            manifest = tmp_path / f"{split}-{lang}.jsonl"
            manifest.write_text(
                "\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
            )
            cfg[split + "_manifest"][lang] = [str(manifest)]
    return cfg, processor


def records(path):
    with sqlite3.connect(path) as db:
        return db.execute("SELECT * FROM samples ORDER BY split,language,ordinal").fetchall()


def test_parallel_matches_serial_and_reports_progress(tmp_path, capsys):
    cfg, processor = setup_scan(tmp_path)
    serial = scan(cfg, processor, True, workers=1, chunk_size=2, temp_dir=tmp_path)
    expected = records(cfg["index_path"])
    cfg["index_path"] = str(tmp_path / "parallel.sqlite")
    parallel = scan(
        cfg, processor, True, workers=2, chunk_size=2, progress_seconds=0.05, temp_dir=tmp_path
    )
    assert serial == parallel
    assert records(cfg["index_path"]) == expected
    assert serial["skipped"] == {
        "identical_duplicate": 4,
        "unusable_transcript": 4,
        "overlong_target": 4,
    }
    assert all(count == 4 for split in parallel["counts"].values() for count in split.values())
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(e["event"] == "scan_progress" for e in events)
    assert events[-1]["event"] == "scan_complete"
    assert events[-1]["manifest_bytes_percent"] == 100
    assert events[-1]["workers"] == 2
    for source in parallel["sources"]:
        from pathlib import Path

        assert source["sha256"] == hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.scan.lock"))
    assert not list(tmp_path.glob("gqasr-index-*"))


@pytest.mark.parametrize("failure", ["leakage", "malformed", "missing_audio"])
def test_parallel_errors_are_contextual_and_not_published(tmp_path, failure):
    from pathlib import Path

    cfg, processor = setup_scan(tmp_path)
    if failure == "leakage":
        cfg["eval_manifest"]["ar"] = cfg["train_manifest"]["ar"]
        match = "leakage"
    elif failure == "malformed":
        path = Path(cfg["train_manifest"]["ar"][0])
        path.write_text("{invalid json}\n")
        match = r"train-ar.jsonl:1:"
    else:
        (tmp_path / "train-ar-0.wav").unlink()
        match = r"train-ar.jsonl:2:"
    with pytest.raises(ValueError, match=match):
        scan(cfg, processor, True, workers=2, chunk_size=2, temp_dir=tmp_path)
    assert not Path(cfg["index_path"]).exists()
    assert not list(tmp_path.glob("*.scan.lock"))
    assert not list(tmp_path.glob("gqasr-index-*"))


def test_parallel_target_lengths_match_processor(tmp_path):
    from gqasr.scan_workers import check_scan_batch, initialize_scan_worker

    cfg, processor = setup_scan(tmp_path)
    text = "مرحبا hello"
    exact = len(processor.target(text, "ar"))
    row = json.dumps(dict(audio="unused.wav", text=text, duration=0.2)).encode()
    for limit, reason in [(exact, None), (exact - 1, "overlong_target")]:
        cfg["max_target_length"] = limit
        initialize_scan_worker(processor.tokenizer.backend_tokenizer.to_str(), cfg, False)
        assert check_scan_batch("/tmp/manifest.jsonl", "ar", [(1, row)])[0][2] == reason


def test_existing_scan_lock_is_preserved(tmp_path):
    from pathlib import Path

    cfg, processor = setup_scan(tmp_path)
    lock = Path(cfg["index_path"] + ".scan.lock")
    lock.write_text("owned by another scan\n")
    with pytest.raises(FileExistsError, match="scan lock exists"):
        scan(cfg, processor, workers=2, temp_dir=tmp_path)
    assert lock.read_text() == "owned by another scan\n"
    assert not Path(cfg["index_path"]).exists()


def test_worker_autodetection_respects_allocation(monkeypatch):
    import gqasr.data as data

    monkeypatch.setattr(data.os, "sched_getaffinity", lambda _: set(range(96)), raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert data.available_scan_workers() == 96
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    assert data.available_scan_workers() == 24
