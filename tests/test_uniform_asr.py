"""Fresh-only publication and consistent decoding across accuracy/latency."""
import copy
import json

import pytest
from test_public_asr_benchmark import adapter, make_public_run, public_plan  # noqa: F401

from data_processing.arabic_reference import local_checkpoint
from data_processing.arabic_reference.core import digest, load_plan, write_json
from data_processing.arabic_reference.latency import source_binding
from data_processing.arabic_reference.leaderboard import build_leaderboard
from data_processing.arabic_reference.public_benchmark import REGISTRY, export_submissions, model_config
from data_processing.arabic_reference.uniform_benchmark import validate_uniform_runs


def test_fresh_only_ranks_no_historical_predictions(public_plan, tmp_path):
    plan_root, source = public_plan
    root = tmp_path / "runs"
    make_public_run(root, source)
    plan, rows = load_plan(plan_root)
    manifest = export_submissions(root, ["qwen3-asr-0.6b"], plan, rows)
    out = tmp_path / "board"
    build_leaderboard(plan_root, out, submissions=manifest, submissions_only=True)
    data = json.loads((out / "leaderboard.json").read_text())
    assert data["models"] == ["qwen3-asr-0.6b"]
    assert {p["name"] for p in data["prediction_sources"]} == {"qwen3-asr-0.6b"}
    assert {p["name"] for p in data["reference_curation_sources"]} == {"qasr", "gqasr"}
    assert all(set(e["predictions"]) == {"qwen3-asr-0.6b"} for e in data["examples"])
    assert all(s["model"] == "qwen3-asr-0.6b" for p in data["panels"] for s in p["scores"])
    assert load_plan(plan_root)[1] == rows
    assert data["ranking_source"] == "fresh_submissions_only"
    with pytest.raises(ValueError, match="provenance-bound"):
        build_leaderboard(plan_root, tmp_path / "bad", submissions_only=True)


def test_native_qasr_registration_and_checkpoint_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("QASR_REFRESH_CHECKPOINT", str(tmp_path))
    monkeypatch.setattr(local_checkpoint, "checkpoint_identity", lambda p, backend: {"weights": backend})
    cfg = local_checkpoint.model_config(tmp_path, tmp_path / "run", local_checkpoint.QASR_NAME)
    assert cfg["backend"] == "qasr"
    assert cfg["benchmark_model"]["checkpoint_digest"] == digest({"weights": "qasr"})
    local_checkpoint.validate_registration(
        cfg["benchmark_model"], {"checkpoint": {"weights": "qasr"}}, local_checkpoint.QASR_NAME
    )
    with pytest.raises(ValueError, match="identity"):
        local_checkpoint.validate_registration(
            cfg["benchmark_model"], {"checkpoint": {"weights": "changed"}}, local_checkpoint.QASR_NAME
        )


def test_jais_output_options_are_bound_to_accuracy_and_latency(tmp_path, monkeypatch):
    from data_processing.arabic_reference import public_benchmark
    snapshot = tmp_path / "jais-asr"
    source = {**REGISTRY["jais-asr"], "files": {"weights": "abc"}, "postprocessing": "old"}
    write_json(snapshot / "source.json", source)
    monkeypatch.setattr(public_benchmark, "checkpoint_identity", lambda *a: source["files"])
    monkeypatch.setenv("PUBLIC_JAIS_DIACRITICS", "1")
    cfg = model_config("jais-asr", tmp_path, tmp_path, tmp_path / "run")
    assert cfg["decoding_options"] == {"diacritics": True, "code_switch": False}
    assert "diacritics=true" in cfg["public_model"]["postprocessing"]
    run = {**cfg, "checkpoint": source["files"]}
    binding = source_binding(run)
    assert binding["decoding_options"] == cfg["decoding_options"]
    other = copy.deepcopy(run)
    other["decoding_options"]["diacritics"] = False
    assert source_binding(other) != binding
    assert json.loads((snapshot / "source.json").read_text()) == source
    monkeypatch.setenv("PUBLIC_JAIS_DIACRITICS", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        model_config("jais-asr", tmp_path, tmp_path, tmp_path / "run")


@pytest.mark.parametrize("kind,options", [
    ("jais", {"diacritics": "true"}),
    ("jais", {"unknown": True}),
    ("hf-qwen", {"diacritics": True}),
])
def test_invalid_format_options_fail_closed(kind, options):
    with pytest.raises(ValueError):
        adapter(kind).configure_decoding(options)


def uniform_fixture(root, phase="full"):
    names = [local_checkpoint.QASR_NAME, "jais-asr"]
    for name in names:
        run = {
            "selection": "same", "modes": ["arabic"], "precision": "bf16",
            "batch_size": 32, "task_size": 32, "runtime": {"torch": "2.8"},
            "backend": REGISTRY[name]["backend"], "max_new_tokens": 768,
            "gpu_isolation": "checked_before_during_after",
        }
        if name == "jais-asr":
            run["decoding_options"] = {"diacritics": True, "code_switch": False}
        folder = root / name / phase
        write_json(folder / "run.json", run)
        write_json(folder / "progress.json", {"event": "benchmark_complete", "completed": 6, "total": 6})
        write_json(folder / "scores.json", {"protocol": run, "performance": {
            "oom_batch_splits": 0,
            "attempts": [{"gpus": ["0", "1"], "workers_per_gpu": 4, "actual_workers": 8, "ready_workers": 8, "batch_size": 32, "status": "complete",
                          "start_policy": "all_ready" if phase == "smoke" else "as_ready"}],
        }})
    return names


@pytest.mark.parametrize("phase", ["smoke", "full"])
@pytest.mark.parametrize("change", ["none", "batch", "runtime", "workers", "oom", "format", "partial", "isolation", "interrupted", "not_ready", "start_policy"])
def test_uniform_validation_blocks_mixed_or_partial_runs(tmp_path, change, phase):
    names = uniform_fixture(tmp_path, phase)
    folder = tmp_path / "jais-asr" / phase
    run = json.loads((folder / "run.json").read_text())
    report = json.loads((folder / "scores.json").read_text())
    if change == "batch": run["batch_size"] = 16
    if change == "runtime": run["runtime"]["torch"] = "other"
    if change == "workers": report["performance"]["attempts"][0]["workers_per_gpu"] = 8
    if change == "oom": report["performance"]["oom_batch_splits"] = 1
    if change == "format": run["decoding_options"]["diacritics"] = False
    if change == "partial": write_json(folder / "progress.json", {"event": "benchmark_progress"})
    if change == "isolation": run.pop("gpu_isolation")
    if change == "interrupted": report["performance"]["attempts"][0]["status"] = "interrupted"
    if change == "not_ready": report["performance"]["attempts"][0]["ready_workers"] = 7
    if change == "start_policy": report["performance"]["attempts"][0]["start_policy"] = "as_ready" if phase == "smoke" else "all_ready"
    report["protocol"] = run
    write_json(folder / "run.json", run)
    write_json(folder / "scores.json", report)
    kw = dict(gpus=["0", "1"], workers=4, batch=32, diacritics=True, phase=phase)
    if change == "none":
        assert set(validate_uniform_runs(tmp_path, names, **kw)["run_identities"]) == set(names)
    else:
        with pytest.raises(ValueError):
            validate_uniform_runs(tmp_path, names, **kw)


def test_gpu_isolation_allows_only_this_runs_workers(monkeypatch):
    from gqasr.arabic_benchmark import runner

    def query(command, **kwargs):
        assert "--id=0,1" in command
        return "101\n102\n"

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    runner.assert_gpu_isolation(["0", "1"], [101, 102])
    with pytest.raises(ValueError, match="competing compute.*102"):
        runner.assert_gpu_isolation(["0", "1"], [101])
    with pytest.raises(ValueError, match="competing compute"):
        runner.assert_gpu_isolation(["0", "1"])


def _isolation_test_worker(slot, gpu, checkpoint, settings, jobs, events, begin, load_slots, log_path):
    import time

    events.put({"event": "ready", "slot": slot, "gpu": gpu})
    time.sleep(60)


@pytest.mark.parametrize("when", ["before", "during"])
def test_scheduler_rejects_competing_job_and_cleans_up_own_workers(
    public_plan, tmp_path, monkeypatch, when
):
    import multiprocessing
    from gqasr.arabic_benchmark import runner, worker

    _, source = public_plan
    monkeypatch.setattr(runner, "official_rows", lambda *a: source)
    monkeypatch.setattr(runner, "prepared_rows", lambda *a: source)
    monkeypatch.setattr(runner, "checkpoint_identity", lambda *a: {"weights": "fixture"})
    monkeypatch.setattr(worker, "worker_main", _isolation_test_worker)
    checks = []

    def isolation(gpus, allowed_pids=()):
        checks.append(list(allowed_pids))
        if when == "before" or allowed_pids:
            raise ValueError("competing compute fixture")

    monkeypatch.setattr(runner, "assert_gpu_isolation", isolation)
    out = tmp_path / "isolated"
    cfg = {"checkpoint": str(tmp_path), "work_dir": str(tmp_path), "output_dir": str(out),
           "require_idle_gpus": True, "task_size": 1}
    with pytest.raises(ValueError, match="competing compute"):
        runner.run(cfg, gpus="0", workers_per_gpu=1, modes=["arabic"])
    attempt = json.loads((out / "attempts.json").read_text())[-1]
    assert attempt["status"] == "interrupted"
    assert not (out / "scores.json").exists()
    assert checks[0] == []
    if when == "during":
        assert checks[-1] and not {p.pid for p in multiprocessing.active_children()} & set(checks[-1])
