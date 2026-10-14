from __future__ import annotations

import json

import pytest

from jev.events import EVENTS_NAME, RUN_NAME, RunLog
from jev_contracts import ValidationError, canonical_json


def identity(value="run-a"):
    return {"config": value}


def read_events(root):
    text = (root / EVENTS_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_rank_zero_writes_descriptor_and_durable_canonical_lines(tmp_path):
    root = tmp_path / "run"
    with RunLog(root, identity=identity(), rank=0) as log:
        log.event("start", {"update": 0})
        log.event("update", {"update": 1, "loss": 0.5})
        log.event("tick")
    descriptor = json.loads((root / RUN_NAME).read_text(encoding="utf-8"))
    assert descriptor == {"schema_version": 1, "identity": identity()}
    raw = (root / EVENTS_NAME).read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw.splitlines()[1] == canonical_json({"event": "update", "update": 1, "loss": 0.5})
    assert read_events(root) == [
        {"event": "start", "update": 0},
        {"event": "update", "update": 1, "loss": 0.5},
        {"event": "tick"},
    ]


def test_nonzero_ranks_construct_but_never_write(tmp_path):
    root = tmp_path / "run"
    log = RunLog(root, identity=identity(), rank=1)
    log.event("start", {"update": 0})
    log.close()
    assert not root.exists()


def test_resume_appends_only_for_the_identical_identity(tmp_path):
    root = tmp_path / "run"
    RunLog(root, identity=identity(), rank=0).event("start")
    RunLog(root, identity=identity(), rank=0).event("update")
    assert [entry["event"] for entry in read_events(root)] == ["start", "update"]
    with pytest.raises(ValueError, match="identity"):
        RunLog(root, identity=identity("other"), rank=0)


def test_guards_serialization_failures_and_bad_arguments(tmp_path):
    (tmp_path / "occupied").write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        RunLog(tmp_path / "occupied", identity=identity(), rank=0)
    with pytest.raises(ValueError, match="identity"):
        RunLog(tmp_path / "a", identity={}, rank=0)
    with pytest.raises(ValueError, match="rank"):
        RunLog(tmp_path / "b", identity=identity(), rank=-1)
    log = RunLog(tmp_path / "c", identity=identity(), rank=0)
    with pytest.raises(ValueError, match="Event name"):
        log.event("")
    with pytest.raises(ValueError, match="payload"):
        log.event("bad", ["not", "a", "mapping"])
    with pytest.raises(ValidationError):
        log.event("bad", {"value": float("nan")})
    log.close()
    assert read_events(tmp_path / "c") == []
