"""Offline protocol and orchestration tests; no pretrained weights or GPU required."""
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from gqasr.arabic_benchmark.common import atomic_json, fingerprint, read_rows, sha256
from gqasr.arabic_benchmark.scoring import aggregate, counts, normalize_elm

from gqasr.arabic_benchmark import data, runner, worker


@pytest.mark.parametrize(("raw", "normalized"), [
    ("  أَهْلًا، و سَهْلًا! ١٢٣؟", "اهلا وسهلا 123"),
    ("آ أ إ ؤ ئ ء پ ڤ", "ا ا ا وي ب ف"),
    ("ا\tب\nو  ج", "ا ب وج"),
    # These characters are deliberately NOT normalized by the official recipe.
    ("ى ة ـ … ٰ ABC", "ى ة ـ … ٰ ABC"),
])
def test_elm_normalization_contract(raw, normalized):
    assert normalize_elm(raw) == normalized


def test_word_character_denominators_and_empty_references():
    result = counts("ا ب", "اب", normalized=True)
    assert (result["word_errors"], result["words"]) == (2, 2)
    assert (result["character_errors"], result["characters"]) == (1, 3)
    assert counts("", "اضافة", normalized=True)["word_errors"] == 1
    assert counts("أَهْلًا!", "اهلا", normalized=True)["word_errors"] == 0
    assert counts("أَهْلًا!", "اهلا")["word_errors"] == 1


def test_macro_is_not_pooled_or_mean_utterance_wer():
    rows = [
        {"id": "1", "dataset": "sada", "text": "ا", "pred_text": "ب"},
        {"id": "2", "dataset": "sada", "text": "ا ب ج", "pred_text": "ا ب ج"},
        {"id": "3", "dataset": "commonvoice", "text": "ا", "pred_text": "ا"},
    ]
    result = aggregate(rows)
    assert result["elm/sada"]["wer"] == .25
    assert result["elm/pooled"]["wer"] == .2
    assert result["elm/macro"]["wer"] == .125
    assert not result["elm/macro"]["full_suite"]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(rows + rows[:1])


def _reference(index, dataset="casablanca", duration=.1):
    return {"id": f"{dataset}:{index:06d}", "dataset": dataset, "ordinal": index,
            "relative_audio": f"clips/segment_{index}.wav", "text": "مرجع رسمي",
            "duration": duration, "upstream_id": None, "metadata": {}}


def _wav_bytes():
    stream = io.BytesIO()
    sf.write(stream, np.linspace(-.2, .2, 1600), 16000, format="WAV")
    return stream.getvalue()


@pytest.mark.parametrize("missing", [False, True])
def test_casablanca_real_parquet_extraction(tmp_path, monkeypatch, missing):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    official = [_reference(0), _reference(1)]
    for index, dialect in enumerate(("Egypt", "Morocco")):
        if missing and index == 1:
            continue
        folder = snapshot / dialect
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([
            {"seg_id": f"segment_{index}", "audio": {"bytes": _wav_bytes()},
             "gender": "male", "transcription": "Do not replace the official reference"},
        ]), folder / "test-00000-of-00001.parquet")
    requests = []

    def snapshot_download(repository, **kwargs):
        requests.append((repository, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(data, "official_rows", lambda *args: official)
    root = tmp_path / "audio"
    cfg = {"work_dir": str(tmp_path), "datasets": {"casablanca": {"root": str(root)}}}
    if missing:
        with pytest.raises(ValueError, match="missing 1 official test IDs"):
            data.download_casablanca(cfg, workers=2)
        assert not (root / "download-provenance.json").exists()
    else:
        data.download_casablanca(cfg, workers=2)
        assert json.loads((root / "download-provenance.json").read_text())["matched"] == 2
        metadata = read_rows(root / "metadata.jsonl")
        assert {m["relative_audio"] for m in metadata} == {r["relative_audio"] for r in official}
        assert {m["metadata"]["dialect"] for m in metadata} == {"Egypt", "Morocco"}
        for index in range(2):
            assert len(data.decode({"path": str(root / f"segment_{index}.wav")})) == 1600
    assert requests[0][1]["revision"] == data.CASABLANCA_REVISION
    assert requests[0][1]["allow_patterns"] == ["*/test-*.parquet"]


def test_resolver_rejects_ambiguous_encodings(tmp_path):
    for suffix in ("wav", "mp3"):
        (tmp_path / f"segment_0.{suffix}").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        data.resolve_audio(_reference(0), tmp_path, data.inventory(tmp_path), {})


@pytest.mark.parametrize(("dataset", "filename", "recording", "start", "duration"), [
    ("sada", "6k_SBA_107_0-seg_3_990-14_890.wav", "6k_SBA_107_0", 3.99, 10.9),
    ("mgb2", "03BD00C0_spk-0001_seg-0001671:0002252.wav", "03BD00C0", 16.71, 5.81),
])
def test_original_recording_segment_boundaries(tmp_path, dataset, filename, recording, start, duration):
    source = tmp_path / f"{recording}.wav"
    source.touch()
    row = {**_reference(0, dataset=dataset), "relative_audio": filename}
    resolved = data.resolve_audio(row, tmp_path, data.inventory(tmp_path), {})
    assert resolved["path"] == str(source.resolve())
    assert resolved["offset_seconds"] == pytest.approx(start)
    assert resolved["duration_seconds"] == pytest.approx(duration)


def test_ffmpeg_mp3_segment_fallback(tmp_path, monkeypatch):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required by the cluster doctor")
    wav, mp3 = tmp_path / "source.wav", tmp_path / "source.mp3"
    wav.write_bytes(_wav_bytes())
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(wav), str(mp3)], check=True)

    def unsupported(*args):
        raise RuntimeError("simulate libsndfile without mp3 support")

    monkeypatch.setattr(sf, "SoundFile", unsupported)
    wave = data.decode({"path": str(mp3), "offset_seconds": .02, "duration_seconds": .05})
    assert wave.shape == (800,)
    assert np.isfinite(wave).all() and np.max(np.abs(wave)) > .01


def test_audio_segment_resample_and_cached_repair(tmp_path):
    path = tmp_path / "source.wav"
    sr = 48000
    wave = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    sf.write(path, np.column_stack((wave, wave)), sr, subtype="FLOAT")
    source = {"path": str(path), "offset_seconds": .2, "duration_seconds": .25}
    args = (_reference(0, duration=.25), source, {"dialect": "Egypt"}, str(tmp_path / "prepared"), .01)
    result = data.prepare_one(args)
    audio = np.load(result["audio"])
    assert audio.shape == (4000,)
    assert np.isfinite(audio).all()
    assert result["text"] == "مرجع رسمي"
    assert result["metadata"]["dialect"] == "Egypt"
    assert data.prepare_one(args) == result
    Path(result["audio"]).write_bytes(b"damaged")
    assert data.prepare_one(args)["audio_sha256"] == result["audio_sha256"]
    with pytest.raises(ValueError, match="no sample was trimmed or dropped"):
        data.prepare_one((_reference(1, duration=1), source, {}, args[3], .01))


def test_parallel_audio_preparation_and_membership_validation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(4):
        (source / f"segment_{i}.wav").write_bytes(_wav_bytes())
    official = [_reference(i) for i in range(4)]
    monkeypatch.setattr(data, "official_rows", lambda *args: official)
    cfg = {"work_dir": str(tmp_path), "datasets": {"casablanca": {"root": str(source)}}}
    data.prepare(cfg, datasets=["casablanca"], workers=2)
    rows = data.prepared_rows(cfg, datasets=["casablanca"])
    assert [r["id"] for r in rows] == [r["id"] for r in official]
    assert all(np.load(r["audio"]).shape == (1600,) for r in rows)
    manifest = tmp_path / "prepared" / "casablanca.jsonl"
    manifest.write_text(manifest.read_text().replace("مرجع رسمي", "changed text"))
    with pytest.raises(ValueError, match="Changed prepared manifest"):
        data.prepared_rows(cfg, datasets=["casablanca"])


def test_streaming_smoke_stops_after_requested_samples(tmp_path, monkeypatch):
    import huggingface_hub

    from gqasr.arabic_benchmark import source_smoke

    dialects = ("Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen")
    paths = [f"{dialect}/test-00000-of-00001.parquet" for dialect in dialects]
    closed = []

    class API:
        def list_repo_files(self, *args, **kwargs):
            assert kwargs["revision"] == data.CASABLANCA_REVISION
            return paths

    def stream(paths):
        index = dialects.index(Path(paths[0]).parent.name)
        try:
            yield paths[0], {"seg_id": f"segment_{index}", "audio": {"bytes": _wav_bytes()}}
            pytest.fail("The smoke test read past the requested sample count")
        finally:
            closed.append(paths[0])

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(source_smoke, "stream_rows", stream)
    monkeypatch.setattr(source_smoke, "fetch_references", lambda *args: None)
    monkeypatch.setattr(source_smoke, "official_rows", lambda *args: [_reference(i) for i in range(8)])
    result = source_smoke.source_smoke(tmp_path, per_dialect=1, workers=2)
    assert result["samples"] == 8
    assert result["all_ids_match_official_references"]
    assert not result["model_weights_loaded"]
    assert set(closed) == set(paths)
    assert {r["text"] for r in read_rows(tmp_path / "samples.jsonl")} == {"مرجع رسمي"}


def _fake_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    """Real spawned processes and queue/atomic-file protocol, deterministic dummy recognition."""
    events.put({"event": "ready", "slot": slot, "gpu": gpu})
    begin.wait()
    while (task := jobs.get()) is not None:
        predictions = [{**row, "mode": mode, "pred_text": row["text"], "generated_tokens": 1,
                        "generation_limit_hit": False}
                       for row in task["rows"] for mode in task["modes_by_id"][row["id"]]]
        atomic_json(task["output"], {"run_id": task["run_id"], "task_id": task["task_id"],
                                    "predictions": predictions, "stats": {
                                        "inference_seconds": .01, "audio_seconds": 1,
                                        "oom_batch_splits": 0, "peak_allocated_gib": 0,
                                        "peak_reserved_gib": 0}})
        events.put({"event": "task_complete", "slot": slot, "task_id": task["task_id"]})


def _idle_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    events.put({"event": "ready", "slot": slot, "gpu": gpu})
    begin.wait()
    time.sleep(60)


def _scheduler_fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "tokenizer").mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "tokenizer/tokenizer.json"):
        (checkpoint / name).write_text("test fixture; never loaded as a real model")
    rows = [{**_reference(i, dataset="sada"), "audio_sha256": f"hash-{i}", "actual_duration": .1}
            for i in range(12)]
    monkeypatch.setattr(runner, "official_rows", lambda *args: rows)
    monkeypatch.setattr(runner, "prepared_rows", lambda *args: rows)
    monkeypatch.setattr(worker, "worker_main", _fake_worker)
    cfg = {"checkpoint": str(checkpoint), "work_dir": str(tmp_path), "output_dir": str(tmp_path / "results"),
           "precision": "fp32", "task_size": 2, "simultaneous_model_loads": 2}
    return cfg, Path(cfg["output_dir"])


def test_spawned_scheduler_exact_coverage_resume_and_identity(tmp_path, monkeypatch):
    cfg, output = _scheduler_fixture(tmp_path, monkeypatch)
    runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=2)
    first = json.loads((output / "scores.json").read_text())
    assert first["performance"]["total_workers"] == 4
    assert len(first["performance"]["attempts"][0]["workers"]) == 4
    assert not first["leaderboard_context"]["comparable_to_leaderboard"]
    assert json.loads((output / "progress.json").read_text())["percent"] == 100
    for mode in ("arabic", "auto"):
        assert len(read_rows(output / f"predictions-{mode}.jsonl")) == 12
    runner.run(cfg, datasets=["sada"], gpus="0", workers_per_gpu=1)
    assert len(json.loads((output / "attempts.json").read_text())) == 1
    # Simulate an interrupted task: only these four mode/example pairs should run again.
    next((output / "tasks").glob("*.json")).unlink()
    runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=2)
    attempt = json.loads((output / "attempts.json").read_text())[-1]
    assert attempt["resumed_predictions"] == 20
    assert attempt["new_predictions"] == 4
    assert attempt["actual_workers"] == 1
    identity = json.loads((output / "run.json").read_text())
    assert {"modeling.py", "processing.py"} <= identity["implementation"].keys()
    changed_code = {**identity["implementation"], "modeling.py": "changed implementation"}
    with monkeypatch.context() as scoped:
        scoped.setattr(runner, "implementation_identity", lambda: changed_code)
        with pytest.raises(ValueError, match="different checkpoint/selection/protocol/code"):
            runner.run(cfg, datasets=["sada"], gpus="0", workers_per_gpu=1)
    (Path(cfg["checkpoint"]) / "model.safetensors").write_text("changed weights")
    with pytest.raises(ValueError, match="different checkpoint"):
        runner.run(cfg, datasets=["sada"], gpus="0", workers_per_gpu=1)


def test_scheduler_stall_exits_and_preserves_resume_metadata(tmp_path, monkeypatch):
    cfg, output = _scheduler_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "worker_main", _idle_worker)
    cfg["progress_timeout_seconds"] = .05
    with pytest.raises(TimeoutError, match="No inference task completed"):
        runner.run(cfg, datasets=["sada"], gpus="0", workers_per_gpu=1)
    assert json.loads((output / "attempts.json").read_text())[-1]["status"] == "interrupted"


def _wait_for_marker(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"Early-start dependency was not satisfied: {path.name}")
        time.sleep(.01)


def _staggered_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    """The second loader cannot finish until the first worker completes a task."""
    root = Path(checkpoint)
    try:
        with load_slots:
            if slot == 0:
                # Both must be allowed inside the loading semaphore concurrently.
                _wait_for_marker(root / "second-loading.json")
            else:
                atomic_json(root / "second-loading.json", {})
                _wait_for_marker(root / "early-task.json")
                if (root / "fail-late.json").exists():
                    events.put({"event": "worker_error", "slot": slot,
                                "traceback": "late loader failed after early task"})
                    time.sleep(60)
                atomic_json(root / "early-work-observed.json", {})
        events.put({"event": "ready", "slot": slot, "gpu": gpu})
        begin.wait()
        while (task := jobs.get()) is not None:
            predictions = [{**row, "mode": mode, "pred_text": row["text"], "generated_tokens": 1,
                            "generation_limit_hit": False}
                           for row in task["rows"] for mode in task["modes_by_id"][row["id"]]]
            atomic_json(task["output"], {"run_id": task["run_id"], "task_id": task["task_id"],
                                        "predictions": predictions, "stats": {
                                            "inference_seconds": .01, "audio_seconds": 1,
                                            "oom_batch_splits": 0, "peak_allocated_gib": 0,
                                            "peak_reserved_gib": 0}})
            events.put({"event": "task_complete", "slot": slot, "task_id": task["task_id"]})
            if slot == 0:
                atomic_json(root / "early-task.json", {})
                _wait_for_marker(root / "early-work-observed.json")
                # Let the second worker deliver readiness before the remaining tasks.
                time.sleep(60)
    except BaseException as exc:
        events.put({"event": "worker_error", "slot": slot, "traceback": str(exc)})
        time.sleep(60)


@pytest.mark.parametrize("fail_late", [False, True])
def test_as_ready_processing_overlaps_loading_and_late_failure_resumes(tmp_path, monkeypatch, fail_late):
    cfg, output = _scheduler_fixture(tmp_path, monkeypatch)
    cfg.update(start_when_ready=True, simultaneous_model_loads="all")
    checkpoint = Path(cfg["checkpoint"])
    if fail_late:
        atomic_json(checkpoint / "fail-late.json", {})
    monkeypatch.setattr(worker, "worker_main", _staggered_worker)
    if fail_late:
        with pytest.raises(RuntimeError, match="late loader failed after early task"):
            runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=1)
        assert list((output / "tasks").glob("*.json"))
        attempt = json.loads((output / "attempts.json").read_text())[-1]
        assert attempt["status"] == "interrupted" and attempt["ready_workers"] == 1
        assert attempt["first_ready_seconds"] is not None and attempt["all_ready_seconds"] is None
        monkeypatch.setattr(worker, "worker_main", _fake_worker)
        runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=1)
        assert json.loads((output / "attempts.json").read_text())[-1]["resumed_predictions"] > 0
    else:
        runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=1)
        assert (checkpoint / "early-work-observed.json").is_file()
        attempt = json.loads((output / "attempts.json").read_text())[-1]
        assert attempt["first_ready_seconds"] < attempt["all_ready_seconds"]
    performance = json.loads((output / "scores.json").read_text())["performance"]
    assert performance["start_policy"] == "as_ready" and performance["simultaneous_model_loads"] == 2
    assert performance["processing_includes_worker_startup"] is True
    for mode in ("arabic", "auto"):
        predictions = read_rows(output / f"predictions-{mode}.jsonl")
        assert len(predictions) == len({r["id"] for r in predictions}) == 12


def _partly_ready_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    if slot == 0:
        events.put({"event": "ready", "slot": slot, "gpu": gpu})
    time.sleep(60)


def _unused_loader_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    if slot == 0:
        _fake_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path)
    else:
        time.sleep(60)


def test_as_ready_finishes_complete_coverage_without_waiting_for_unused_loader(tmp_path, monkeypatch):
    cfg, output = _scheduler_fixture(tmp_path, monkeypatch)
    cfg.update(start_when_ready=True, startup_timeout_seconds=5)
    monkeypatch.setattr(worker, "worker_main", _unused_loader_worker)
    runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=1)
    attempt = json.loads((output / "attempts.json").read_text())[-1]
    assert attempt["status"] == "complete" and attempt["ready_workers"] == 1
    assert attempt["all_ready_seconds"] is None
    assert json.loads((output / "progress.json").read_text())["completed"] == 24


def test_as_ready_still_times_out_a_worker_that_never_loads(tmp_path, monkeypatch):
    cfg, output = _scheduler_fixture(tmp_path, monkeypatch)
    cfg.update(start_when_ready=True, startup_timeout_seconds=3, progress_timeout_seconds=30)
    monkeypatch.setattr(worker, "worker_main", _partly_ready_worker)
    with pytest.raises(TimeoutError, match=r"Model startup timeout \(1/2 ready\)"):
        runner.run(cfg, datasets=["sada"], gpus="0,1", workers_per_gpu=1)
    attempt = json.loads((output / "attempts.json").read_text())[-1]
    assert attempt["status"] == "interrupted" and attempt["ready_workers"] == 1


@pytest.mark.parametrize("value", [0, -1, "bad", True, 1.5])
def test_invalid_model_loading_limit_fails_before_spawning(tmp_path, monkeypatch, value):
    cfg, _ = _scheduler_fixture(tmp_path, monkeypatch)
    cfg["simultaneous_model_loads"] = value
    with pytest.raises(ValueError, match="simultaneous_model_loads"):
        runner.run(cfg, datasets=["sada"], gpus="0", workers_per_gpu=1)


def test_prediction_coverage_rejects_missing_duplicate_and_altered_references():
    expected = [{**_reference(0), "audio_sha256": "hash"}]
    prediction = {**expected[0], "mode": "arabic", "pred_text": "نص"}
    for found in ([], [prediction, prediction], [{**prediction, "text": "changed"}]):
        with pytest.raises(ValueError):
            runner.validate_predictions(expected, found, "arabic")


def test_full_suite_never_asserts_unverified_leaderboard_equivalence(tmp_path, monkeypatch):
    from copy import deepcopy

    catalog = deepcopy(runner.CATALOG)
    for ds in catalog["datasets"].values():
        ds["rows"] = 1
    monkeypatch.setattr(runner, "CATALOG", catalog)
    expected = [{**_reference(i, dataset=ds), "audio_sha256": "hash"}
                for i, ds in enumerate(runner.DATASETS)]
    identity = {"modes": ["arabic"], "precision": "fp32"}
    predictions = [{**r, "mode": "arabic", "pred_text": r["text"], "generated_tokens": 1,
                    "generation_limit_hit": False} for r in expected]
    atomic_json(tmp_path / "tasks" / "test.json", {
        "run_id": fingerprint(identity), "predictions": predictions,
        "stats": {"inference_seconds": 1, "audio_seconds": 1, "oom_batch_splits": 0,
                  "peak_allocated_gib": 0, "peak_reserved_gib": 0},
    })
    runner.report(tmp_path, expected, identity, 1, [])
    result = json.loads((tmp_path / "scores.json").read_text())
    assert result["full_suite"]
    assert result["leaderboard_context"]["complete_pinned_suite_scored"]
    assert not result["leaderboard_context"]["comparable_to_leaderboard"]
    assert result["leaderboard_context"]["unresolved_requirements"]


def test_audio_integrity_check_before_frontend(tmp_path):
    target = tmp_path / "audio.npy"
    np.save(target, np.ones(1600))
    recognizer = object.__new__(worker.Recognizer)
    expected_hash = sha256(target)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prepared audio changed"):
        recognizer.features([{"id": "test", "audio": str(target), "audio_sha256": expected_hash}])


def test_worker_preserves_alignment_for_partial_modes_and_batches(tmp_path, monkeypatch):
    import os
    import queue
    import threading

    import torch

    class Recognizer:
        def __init__(self, *args):
            self.torch = torch

        def features(self, rows):
            return {"sample_ids": torch.tensor([r["ordinal"] for r in rows])}

        def safe_infer(self, rows, features, mode, budget):
            assert features["sample_ids"].tolist() == [r["ordinal"] for r in rows]
            return [{**r, "mode": mode} for r in rows], 0

    monkeypatch.setattr(worker, "Recognizer", Recognizer)
    monkeypatch.setattr(worker.os, "environ", dict(os.environ))
    rows = [{**_reference(i), "actual_duration": .1} for i in range(3)]
    task = {"task_id": "test", "rows": rows, "run_id": "test-run", "modes": ["arabic", "auto"],
            "modes_by_id": {rows[0]["id"]: ["arabic"], rows[1]["id"]: ["auto"],
                            rows[2]["id"]: ["arabic", "auto"]}, "output": str(tmp_path / "task.json")}
    jobs, events, begin = queue.Queue(), queue.Queue(), threading.Event()
    jobs.put(task)
    jobs.put(None)
    begin.set()
    worker.worker_main(0, "cpu", "unused", {"precision": "fp32", "memory_fraction": 1,
                       "batch_size": 2, "max_new_tokens": 4}, jobs, events, begin,
                       threading.Semaphore(1), str(tmp_path / "worker.log"))
    predictions = json.loads((tmp_path / "task.json").read_text())["predictions"]
    assert {(r["id"], r["mode"]) for r in predictions} == {
        (row_id, mode) for row_id, modes in task["modes_by_id"].items() for mode in modes}
    assert len(predictions) == 4


def test_oom_batch_splitting_preserves_every_example():
    from types import SimpleNamespace

    class OutOfMemoryError(Exception):
        pass

    recognizer = object.__new__(worker.Recognizer)
    recognizer.torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError,
                                                           empty_cache=lambda: None))

    def infer(rows, *args):
        if len(rows) > 1:
            raise OutOfMemoryError()
        return rows

    recognizer.infer = infer
    recognizer.features = lambda rows: None
    rows = [_reference(i) for i in range(4)]
    predictions, retries = recognizer.safe_infer(rows, None, "arabic", 4)
    assert predictions == rows
    assert retries == 3


@pytest.mark.parametrize("stage", ["doctor", "smoke", "run"])
def test_cluster_launcher_gpu_visibility_and_decode_options(tmp_path, stage):
    import os
    import shutil
    import subprocess
    import sys

    root = tmp_path / "gemma_speech"
    (root / "cluster").mkdir(parents=True)
    script = root / "cluster" / "gqasr_arabic_leaderboard.sh"
    shutil.copyfile(Path(__file__).parents[2] / "cluster" / script.name, script)
    env_bin = root / ".venv" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "activate").write_text("export CUDA_VISIBLE_DEVICES=1\n")
    executable = env_bin / "gqasr-arabic-benchmark"
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                          "print(json.dumps({'args': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}))\n")
    executable.chmod(0o755)
    env = {**os.environ, "DECODE_MODES": "auto", "GQASR_BENCH_OUTPUT": str(tmp_path / "output"),
           "GEMMA_SPEECH_ENV": str(root / ".venv")}
    for name in ("GPU_IDS", "WORKERS_PER_GPU", "BATCH_SIZE"):
        env.pop(name, None)
    result = subprocess.run(["bash", str(script), stage], env=env, text=True, capture_output=True, check=True)
    record = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))
    assert record["gpus"] == "0,1,2,3,4,5,6,7"
    if stage != "doctor":
        args = record["args"]
        assert args[args.index("--workers-per-gpu") + 1] == "11"
        assert args[args.index("--modes") + 1] == "auto"
