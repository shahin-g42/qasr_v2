"""Controlled latency boundaries, sampling, failures, provenance and release imports."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_public_asr_benchmark import make_public_run, public_plan  # noqa: F401

from data_processing.arabic_reference import latency_runner as runner
from data_processing.arabic_reference import local_checkpoint
from data_processing.arabic_reference.core import (
    DATASETS,
    digest,
    load_plan,
    sha256,
    write_json,
    write_jsonl,
)
from data_processing.arabic_reference.latency import (
    distribution,
    make_protocol,
    read_latency_profiles,
    request_schedule,
    select_clips,
    source_binding,
    summarize_profile,
    write_latency_csv,
    write_request_csv,
)


def input_rows():
    return [
        {
            "id": f"{ds}:{i:06d}",
            "dataset": ds,
            "actual_duration": float(i + 1),
            "audio_sha256": digest([ds, i]),
        }
        for ds in DATASETS
        for i in range(100)
    ]


def observations(protocol):
    records = []
    for i, (phase, repeat, clip) in enumerate(request_schedule(protocol)):
        seconds = 50.0 if phase != "measured" else clip["actual_duration"] / 2
        records.append(
            {
                **clip,
                "sequence": i,
                "phase": phase,
                "repeat": repeat,
                "status": "ok",
                "request_seconds": seconds,
                "input_preparation_seconds": seconds / 10,
                "inference_decode_seconds": seconds * 0.9,
                "peak_allocated_gib": 4.0,
                "peak_reserved_gib": 5.0,
                "generation_limit_hit": False,
                "refused": False,
                "empty_output": False,
                "output_sha256": digest("مرحبا"),
            }
        )
    return records


def profile_fixture(root, rows=None, plan=None, source=None, name="qasr"):
    rows = rows or input_rows()
    plan = plan or {"manifest_digest": "plan"}
    source = source or {
        "checkpoint": {"weights": "a" * 64},
        "precision": "bf16",
        "max_new_tokens": 768,
        "backend": "qasr",
    }
    protocol = make_protocol(rows, plan["manifest_digest"], per_dataset=3, repeats=2, warmup=2)
    binding = source_binding(source)
    manifest = {
        "protocol": protocol,
        "models": {name: binding},
        "runtime": {"torch": "fixture"},
        "hardware": {"identity": {"uuid": "GPU-fixture"}},
        "implementation": {"profiler": "fixture"},
    }
    manifest["profile_sha256"] = digest(manifest)
    records = observations(protocol)
    folder = root / name
    folder.mkdir(parents=True)
    write_jsonl(folder / "requests-001.jsonl", records)
    report = {
        "status": "complete",
        "model": name,
        "profile_sha256": manifest["profile_sha256"],
        "source": binding,
        "runtime": manifest["runtime"],
        "hardware": manifest["hardware"],
        "records_file": "requests-001.jsonl",
        "records_sha256": sha256(folder / "requests-001.jsonl"),
        "model_load_seconds": 12.0,
        "audio_preload_seconds": 2.0,
        "summary": summarize_profile(records, protocol),
    }
    write_json(folder / "report.json", report)
    write_json(root / "profile.json", manifest)
    return plan, rows, {name: binding}, report


def test_duration_stratified_selection_is_order_independent_and_full_means_all():
    rows = input_rows()
    a = select_clips(rows, 4)
    assert a == select_clips(list(reversed(rows)), 4)
    assert len(a) == 24
    assert [r["actual_duration"] for r in a[:4]] == [13.0, 38.0, 63.0, 88.0]
    assert len(select_clips(rows, 0)) == len(rows)
    assert not any("text" in r for r in a)
    with pytest.raises(ValueError, match="missing dataset"):
        select_clips([r for r in rows if r["dataset"] != "sada"])


def test_same_schedule_cold_and_warm_excluded_and_duration_weighted_rtf():
    protocol = make_protocol(input_rows(), "plan", per_dataset=3, repeats=2, warmup=3)
    records = observations(protocol)
    assert list(request_schedule(protocol)) == list(request_schedule(protocol))
    result = summarize_profile(records, protocol)
    s = result["overall"]
    assert s["successful"] == 36 and s["unique_clips"] == 18
    assert s["aggregate_rtf"] == 0.5 and s["rtf"]["mean"] == 0.5
    assert result["first_request_seconds"] == 50.0
    assert result["warmup"]["successful"] == 3
    assert sum(v["attempted"] for v in result["by_dataset"].values()) == 36
    assert sum(v["attempted"] for v in result["by_duration"].values()) == 36
    assert result["by_duration"]["0–5 s"]["request_seconds"]["mean"] is None
    assert s["request_seconds"]["max"] < 50.0


def test_percentiles_and_invalid_times():
    stats = distribution([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == stats["p50"] == 2.5
    assert stats["p95"] == pytest.approx(3.85)
    assert stats["p99"] == pytest.approx(3.97)
    assert distribution([])["mean"] is None
    for value in (float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError):
            distribution([value])


def test_failures_are_counted_empty_and_truncated_outputs_stay_in_timings():
    protocol = make_protocol(input_rows(), "plan", per_dataset=1, repeats=1, warmup=1)
    records = observations(protocol)
    records[-1] = {**records[-1], "status": "error", "request_seconds": 123.0, "error": "OOM"}
    records[-2].update(empty_output=True, generation_limit_hit=True, refused=True)
    records[-3]["generation_limit_hit"] = None
    s = summarize_profile(records, protocol)["overall"]
    assert s["attempted"] == 6 and s["successful"] == 5 and s["failed"] == 1
    assert s["request_seconds"]["max"] < 123
    assert (
        s["generation_limit_hits"]
        == s["generation_limit_unobservable"]
        == s["empty_outputs"]
        == s["refusals"]
        == 1
    )
    for r in records:
        if r["phase"] == "measured":
            r["status"] = "error"
    s = summarize_profile(records, protocol)["overall"]
    assert s["successful"] == 0 and s["request_seconds"]["mean"] is None


@pytest.mark.parametrize("mutation", ["missing", "repeat", "phase", "audio", "time", "cold_error"])
def test_invalid_or_incomplete_request_coverage_rejected(mutation):
    p = make_protocol(input_rows(), "plan", per_dataset=1)
    records = observations(p)
    if mutation == "missing":
        records.pop()
    if mutation == "repeat":
        records[-1]["repeat"] = 999
    if mutation == "phase":
        records[0]["phase"] = "measured"
    if mutation == "audio":
        records[-1]["audio_sha256"] = "bad"
    if mutation == "time":
        records[-1]["request_seconds"] += 1
    if mutation == "cold_error":
        records[0]["status"] = "error"
    with pytest.raises(ValueError):
        summarize_profile(records, p)


def test_import_recomputes_summary_and_exports_every_observation(tmp_path):
    plan, rows, bindings, report = profile_fixture(tmp_path)
    profile = read_latency_profiles(tmp_path, plan, rows, bindings)
    assert profile["models"]["qasr"]["summary"] == report["summary"]
    write_latency_csv(tmp_path / "summary.csv", profile)
    write_request_csv(tmp_path / "requests.csv", tmp_path, profile)
    with (tmp_path / "requests.csv").open() as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == 39
    assert {r["phase"] for r in exported} == {"first_request", "warmup", "measured"}
    with (tmp_path / "summary.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 11


@pytest.mark.parametrize(
    "mutation",
    ["checkpoint", "plan", "records", "hardware", "runtime", "summary", "identity", "path"],
)
def test_timing_cannot_be_attributed_to_other_checkpoint_protocol_or_records(tmp_path, mutation):
    plan, rows, bindings, report = profile_fixture(tmp_path)
    if mutation == "checkpoint":
        bindings["qasr"]["checkpoint_digest"] = "bad"
    if mutation == "plan":
        plan["manifest_digest"] = "different"
    if mutation == "records":
        (tmp_path / "qasr/requests-001.jsonl").write_text("{}")
    if mutation == "hardware":
        report["hardware"]["identity"]["uuid"] = "other"
    if mutation == "runtime":
        report["runtime"]["torch"] = "other"
    if mutation == "summary":
        report["summary"]["overall"]["request_seconds"]["mean"] = 0.0
    if mutation == "identity":
        report["profile_sha256"] = "bad"
    if mutation == "path":
        report["records_file"] = "../requests.jsonl"
    write_json(tmp_path / "qasr/report.json", report)
    with pytest.raises(ValueError):
        read_latency_profiles(tmp_path, plan, rows, bindings)


def test_unfinished_model_profile_is_explicitly_unavailable(tmp_path):
    plan, rows, bindings, _ = profile_fixture(tmp_path)
    (tmp_path / "qasr/report.json").unlink()
    p = read_latency_profiles(tmp_path, plan, rows, bindings)
    assert p["models"]["qasr"]["status"] == "unavailable"


def test_measure_request_synchronizes_all_phases_without_amortization_or_reference_prompt():
    events = []
    cuda = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        reset_peak_memory_stats=lambda: events.append("reset"),
        max_memory_allocated=lambda: 2**30,
        max_memory_reserved=lambda: 2**31,
    )

    def features(waves):
        assert len(waves) == 1 and np.array_equal(waves[0], np.zeros(16000))
        events.append("prepare")
        return "features"

    def infer(rows, feature, mode, budget):
        assert (
            "text" not in rows[0] and feature == "features" and mode == "arabic" and budget == 768
        )
        events.append("infer")
        return [{"pred_text": "نحن", "generated_tokens": 2, "generation_limit_hit": False}]

    obj = SimpleNamespace(
        torch=SimpleNamespace(cuda=cuda), features_from_audio=features, infer=infer
    )
    times = iter([10.0, 12.0, 17.0])
    result = runner.measure_request(
        obj, {"id": "clip"}, np.zeros(16000), 768, clock=lambda: next(times)
    )
    assert (
        result["request_seconds"] == 7
        and result["input_preparation_seconds"] == 2
        and result["inference_decode_seconds"] == 5
    )
    assert result["peak_allocated_gib"] == 1 and events == [
        "sync",
        "reset",
        "prepare",
        "sync",
        "infer",
        "sync",
    ]


def test_request_error_remains_visible_even_if_empty_cache_fails():
    def fail(*args):
        raise RuntimeError("CUDA problem")

    cuda = SimpleNamespace(
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None, empty_cache=fail
    )
    obj = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), features_from_audio=fail)
    times = iter([1.0, 4.0])
    result = runner.measure_request(obj, {}, None, 768, clock=lambda: next(times))
    assert result == {
        "status": "error",
        "request_seconds": 3.0,
        "error": "RuntimeError: CUDA problem",
    }


def test_isolation_never_kills_foreign_workers(monkeypatch):
    monkeypatch.setattr(runner, "smi", lambda *a, **k: [{"pid": "10", "gpu_uuid": "gpu"}])
    runner.idle_gpu("0", 10)
    with pytest.raises(ValueError, match="other compute processes"):
        runner.idle_gpu("0", 11)


def make_local_checkpoint(path):
    import torch
    from safetensors.torch import save_file

    (path / "tokenizer").mkdir(parents=True)
    for file in (
        "preprocessor_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    ):
        write_json(path / file, {"fixture": True})
    write_json(
        path / "config.json", {"model_type": "gqasr", "acoustic_config": {}, "decoder_config": {}}
    )
    write_json(path / "provenance.json", {"training": {"completed_optimizer_steps": 7500}})
    save_file({"weight": torch.zeros(1)}, path / "model.safetensors")
    return path


def test_fresh_checkpoint_is_direct_artifact_and_partial_save_is_rejected(tmp_path, monkeypatch):
    path = make_local_checkpoint(tmp_path / "checkpoint-00007500")
    monkeypatch.setenv("GQASR_REFRESH_CHECKPOINT", str(path))
    cfg = local_checkpoint.model_config(tmp_path, tmp_path / "run")
    assert cfg["checkpoint"] == str(path) and cfg["benchmark_model"]["training_step"] == 7500
    assert cfg["backend"] == "gqasr" and cfg["start_when_ready"]
    local_checkpoint.validate_registration(
        cfg["benchmark_model"], {"checkpoint": runner.checkpoint_identity(path)}
    )
    (path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="Incomplete GQASR checkpoint"):
        local_checkpoint.checkpoint_metadata(path)


def test_fresh_checkpoint_and_latency_import_preserve_historical_scores(public_plan, tmp_path):  # noqa: F811
    from data_processing.arabic_reference.leaderboard import build_leaderboard
    from data_processing.arabic_reference.public_benchmark import export_submissions
    from data_processing.arabic_reference.release import validate_bundle

    plan_root, source = public_plan
    plan, rows = load_plan(plan_root)
    root = tmp_path / "runs"
    public_folder = make_public_run(root, source)
    local_folder = root / local_checkpoint.NAME / "full"
    local_folder.mkdir(parents=True)
    local_run = json.loads((public_folder / "run.json").read_text())
    local_run.pop("public_model")
    local_run.update(
        backend="gqasr",
        benchmark_model={
            **local_checkpoint.ENTRY,
            "training_step": 7500,
            "checkpoint_digest": digest(local_run["checkpoint"]),
            "postprocessing": "native",
        },
    )
    write_json(local_folder / "run.json", local_run)
    for filename in ("progress.json", "predictions-arabic.jsonl"):
        (local_folder / filename).write_bytes((public_folder / filename).read_bytes())
    manifest = export_submissions(root, ["qwen3-asr-0.6b", local_checkpoint.NAME], plan, rows)
    latency_root = tmp_path / "latency"
    profile_fixture(latency_root, rows, plan, local_run, local_checkpoint.NAME)
    before = tmp_path / "without"
    after = tmp_path / "with"
    build_leaderboard(plan_root, before, submissions=manifest)
    build_leaderboard(plan_root, after, submissions=manifest, latency_root=latency_root)
    a = json.loads((before / "leaderboard.json").read_text())
    b = json.loads((after / "leaderboard.json").read_text())
    assert a["panels"] == b["panels"] and {"qasr", "gqasr", local_checkpoint.NAME} <= set(
        b["models"]
    )
    assert b["model_details"][local_checkpoint.NAME]["model_url"] is None
    assert b["latency"]["models"][local_checkpoint.NAME]["status"] == "complete"
    assert {"latency.csv", "latency.json", "latency-requests.csv", "latency.js"} <= set(
        validate_bundle(after)["files"]
    )


def test_orchestrator_resumes_complete_models_and_restarts_partial_model_with_new_attempt(
    tmp_path, monkeypatch
):
    root = tmp_path / "latency"
    source = {
        "checkpoint": {"weights": "a" * 64},
        "precision": "bf16",
        "max_new_tokens": 768,
        "backend": "qasr",
        "modes": ["arabic"],
    }
    rows = [{**r, "audio": "unused"} for r in input_rows() if r["actual_duration"] == 1]
    plan = {"manifest_digest": "plan", "sources": [{"name": "qasr", "run": source}]}
    env = {
        "ARABIC_REF_PLAN": "plan",
        "LATENCY_MODELS": "qasr",
        "PUBLIC_BENCH_ROOT": "runs",
        "LATENCY_QASR_CHECKPOINT": str(tmp_path),
        "PUBLIC_PREPARED_WORK": "work",
        "LATENCY_PER_DATASET": "1",
        "LATENCY_REPEATS": "1",
        "LATENCY_WARMUP": "2",
        "LATENCY_SEED": "20261020",
        "LATENCY_GPU": "0",
        "LATENCY_ROOT": str(root),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(runner, "load_plan", lambda p: (plan, rows))
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *a: source["checkpoint"])
    monkeypatch.setattr(runner, "bind_prepared", lambda *a: rows)
    monkeypatch.setattr(runner, "hardware", lambda g: {"identity": {"uuid": "gpu"}})
    monkeypatch.setattr(runner, "idle_gpu", lambda *a: None)
    monkeypatch.setattr(runner, "runtime", lambda: {"torch": "fixture"})
    monkeypatch.setattr(runner, "implementation", lambda s: {"profiler": "original"})
    jobs = []

    def child(args, **kwargs):
        job = json.loads(__import__("pathlib").Path(args[-1]).read_text())
        jobs.append(job)
        assert (
            kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["env"]["OMP_NUM_THREADS"] == "1"
        )
        manifest = json.loads((root / "profile.json").read_text())
        records = observations(manifest["protocol"])
        path = root / "qasr" / job["records_file"]
        write_jsonl(path, records)
        write_json(
            root / "qasr/report.json",
            {
                "status": "complete",
                "model": "qasr",
                "profile_sha256": manifest["profile_sha256"],
                "source": manifest["models"]["qasr"],
                "hardware": manifest["hardware"],
                "runtime": manifest["runtime"],
                "records_file": path.name,
                "records_sha256": sha256(path),
                "model_load_seconds": 1.0,
                "audio_preload_seconds": 1.0,
                "summary": summarize_profile(records, manifest["protocol"]),
            },
        )

    monkeypatch.setattr(runner.subprocess, "run", child)
    runner.main(["run"])
    runner.main(["run"])
    assert len(jobs) == 1
    (root / "qasr/report.json").unlink()
    runner.main(["run"])
    assert len(jobs) == 2 and jobs[1]["records_file"] == "requests-002.jsonl"
    assert (root / "qasr/requests-001.jsonl").exists() and (root / "latency-requests.csv").exists()
    monkeypatch.setattr(runner, "implementation", lambda s: {"profiler": "changed"})
    with pytest.raises(ValueError, match="choose a new LATENCY_ROOT"):
        runner.main(["run"])


@pytest.mark.parametrize("changed_audio", [False, True])
def test_worker_preloads_hashes_audio_and_writes_all_phases(tmp_path, monkeypatch, changed_audio):
    import torch
    from gqasr.arabic_benchmark import worker

    rows = []
    for ds in DATASETS:
        path = tmp_path / f"{ds}.npy"
        np.save(path, np.zeros(16000, dtype=np.float32))
        rows.append(
            {
                "id": f"{ds}:000000",
                "dataset": ds,
                "actual_duration": 1.0,
                "audio_sha256": sha256(path),
                "audio": str(path),
            }
        )
    protocol = make_protocol(rows, "fixture", per_dataset=1, repeats=1, warmup=2)
    source = {
        "checkpoint": {"weights": "a" * 64},
        "backend": "gqasr",
        "precision": "bf16",
        "max_new_tokens": 768,
    }
    manifest = {
        "protocol": protocol,
        "models": {"gqasr": source_binding(source)},
        "hardware": {"identity": {"uuid": "gpu"}},
        "runtime": {"fixture": True},
    }
    manifest["profile_sha256"] = digest(manifest)
    write_json(tmp_path / "profile.json", manifest)
    write_jsonl(tmp_path / "audio-paths.jsonl", rows)
    job = {
        "output": str(tmp_path),
        "manifest": str(tmp_path / "profile.json"),
        "name": "gqasr",
        "gpu": "0",
        "audio_manifest": str(tmp_path / "audio-paths.jsonl"),
        "checkpoint": str(tmp_path / "checkpoint"),
        "records_file": "requests-001.jsonl",
    }
    write_json(tmp_path / "job.json", job)
    monkeypatch.setattr(runner, "hardware", lambda gpu: manifest["hardware"])
    monkeypatch.setattr(runner, "runtime", lambda: manifest["runtime"])
    monkeypatch.setattr(runner, "idle_gpu", lambda *a: None)
    monkeypatch.setattr(runner, "telemetry", lambda gpu: {})
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *a: source["checkpoint"])
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        is_bf16_supported=lambda: True,
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 2**30,
        max_memory_reserved=lambda: 2**31,
    )
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr(
        torch,
        "backends",
        SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()),
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)

    class ResidentModel:
        def __init__(self, path, device, precision, fraction):
            assert (device, precision, fraction) == ("cuda:0", "bf16", 0.9)
            self.torch = torch

        def features_from_audio(self, waves):
            assert len(waves) == 1 and waves[0].shape == (16000,)
            return waves

        def infer(self, clips, features, mode, budget):
            assert all("text" not in r for r in clips) and mode == "arabic" and budget == 768
            return [{"pred_text": "مرحبا"}]

    monkeypatch.setattr(worker, "Recognizer", ResidentModel)
    if changed_audio:
        np.save(tmp_path / "sada.npy", np.ones(16000, dtype=np.float32))
        with pytest.raises(ValueError, match="audio changed"):
            runner.run_worker(tmp_path / "job.json")
        assert not (tmp_path / "report.json").exists()
    else:
        runner.run_worker(tmp_path / "job.json")
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["overall"]["attempted"] == 6
        assert report["summary"]["warmup"]["attempted"] == 2
        assert report["first_token_latency_seconds"] is None
        assert len((tmp_path / "requests-001.jsonl").read_text().splitlines()) == 9
