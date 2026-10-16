import json
import sys
from types import SimpleNamespace

import pytest
from gqasr.tracking import Tracker, follow, scalars, training_config


class FakeRun:
    id = "fake123"
    url = "https://example.test/run"

    def __init__(self):
        self.logs = []
        self.definitions = []
        self.finished = None

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values):
        self.logs.append(values)

    def finish(self, exit_code=0):
        self.finished = exit_code


@pytest.fixture
def sdk(monkeypatch):
    runs, calls = [], []

    def init(**kwargs):
        calls.append(kwargs)
        run = FakeRun()
        runs.append(run)
        return run

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    return runs, calls


def cfg(tmp_path):
    return dict(output_dir=str(tmp_path), report_to="wandb", wandb_project="qasr",
                batch_size=40, accumulation_steps=1, expected_world_size=96)


def record(step=5, elapsed=10):
    return dict(step=step, elapsed_seconds=elapsed, rank0_last_microbatch_loss=2.3,
                rank0_last_microbatch_augmentation=[dict(applied=True), dict(applied=False)],
                learning_rates=dict(decoder=1e-5),
                validation={"strict/language:ar": {"wer": 0.2, "cer": 0.1},
                            "arabic_normalized/language:ar": {"wer": 0.15, "cer": None}})


def test_scalars_preserve_metric_meaning():
    result = scalars(record())
    assert result["train/rank0_last_microbatch_loss"] == 2.3
    assert result["eval/strict/language:ar/wer"] == 0.2
    assert result["eval/arabic_normalized/language:ar/wer"] == 0.15
    assert result["augmentation/rank0_applied_fraction"] == 0.5
    assert result["lr/decoder"] == 1e-5
    assert all(isinstance(v, (float, int)) for v in result.values())


def test_rank_zero_only_and_disabled(tmp_path, sdk):
    for settings, main in ((cfg(tmp_path), False), ({**cfg(tmp_path), "report_to": "none"}, True)):
        tracker = Tracker(settings, main=main)
        tracker.log(record())
        tracker.finish()
    assert not sdk[0]


def test_online_restart_skips_uploaded_steps(tmp_path, sdk):
    tracker = Tracker(cfg(tmp_path))
    tracker.log(record())
    tracker.log(record(10, 20))
    assert sdk[0][0].logs[-1]["performance/examples_per_second"] == 1920
    tracker.finish()
    resumed = Tracker(cfg(tmp_path))
    resumed.log(record())
    resumed.log(record(15, 30))
    resumed.finish()
    assert sdk[1][1]["id"] == "fake123"
    assert len(sdk[0][1].logs) == 1
    assert json.loads((tmp_path / "wandb_tracking.json").read_text())["last_step"] == 15


def test_tracking_failure_does_not_abort_training(tmp_path, sdk, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(sys.modules["wandb"], "init", fail)
    tracker = Tracker(cfg(tmp_path))
    with pytest.warns(UserWarning, match="tracking disabled"):
        tracker.log(record())
    tracker.log(record(10))
    tracker.finish()
    assert not tracker.enabled


def test_follow_ignores_partial_line_and_does_not_modify_metrics(tmp_path, sdk):
    (tmp_path / "config.json").write_text(json.dumps(cfg(tmp_path)))
    text = json.dumps(record()) + "\n" + '{"step":'
    source = tmp_path / "metrics.jsonl"
    source.write_text(text)
    follow(tmp_path)
    assert len(sdk[0][0].logs) == 1
    assert source.read_text() == text


def test_resume_only_excludes_logging_fields():
    old = {"batch_size": 20, "workers": 4, "output_dir": "a"}
    new = {**old, "output_dir": "b", "report_to": "wandb", "run_name": "new"}
    assert training_config(old) == training_config(new)
    assert training_config(old) != training_config({**new, "batch_size": 40})


def test_only_one_writer(tmp_path, sdk):
    first = Tracker(cfg(tmp_path))
    second = Tracker(cfg(tmp_path))
    first.log(record())
    with pytest.warns(UserWarning, match="tracking disabled"):
        second.log(record(10))
    first.finish()
    second.finish()
    assert len(sdk[0]) == 1
