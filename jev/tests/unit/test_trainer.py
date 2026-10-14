from __future__ import annotations

import json
import math
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from jev import modeling
from jev.batching import Cursor
from jev.checkpoint import find_checkpoints, latest_checkpoint
from jev.config import TrainConfig
from jev.export import load_model_from_checkpoint
from jev.trainer import (
    build_optimizer,
    build_trainer,
    learning_rate,
    load_release_split,
    snapshot_identity,
)
from jev_contracts import atomic_write_json, write_manifest

from .test_modeling import ProcessorLoader, TinyBackbone, TinyConfig, snapshot_fixture
from .test_processing import make_row


@pytest.fixture
def world_one(monkeypatch):
    """A single-process gloo world, joined through the production env contract."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    for key, value in {
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class TrainerBackbone(TinyBackbone):
    """The tiny runtime carrying a genuine (empty) decoder-block sequence.

    decoder_blocks refuses a language_model without a layers container, so the
    sharded path requires a real ModuleList even when it holds no blocks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.language_model.layers = nn.ModuleList()


@pytest.fixture
def trainer_runtime(monkeypatch):
    monkeypatch.setattr(
        modeling, "_runtime", lambda: (ProcessorLoader, TinyConfig, TrainerBackbone),
    )


def release_fixture(root, *, splits=("train", "development"), approved=True,
                    kind="dataset_release", schema_version=2, split_names=None,
                    manifest_files=None, lines_hook=None):
    root.mkdir()
    split_names = split_names or {split: split + ".jsonl" for split in splits}
    files = []
    for split in splits:
        rows = []
        for index in range(3 if split == "train" else 2):
            row = make_row("choice", 3 if split == "train" else 2)
            row["id"] = f"{split}-{index:03d}"
            row["family_id"] = f"{split}-family-{index % 2}"
            row["split"] = split
            rows.append(row)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        if lines_hook is not None:
            lines = lines_hook(split, lines)
        name = split_names[split]
        (root / name).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        files.append(name)
    (root / "audit.json").write_text('{"approved": true}\n', encoding="utf-8")
    (root / "registry.json").write_text("{}\n", encoding="utf-8")
    files.extend(["audit.json", "registry.json"])
    write_manifest(root, manifest_files if manifest_files is not None else files, {
        "kind": kind, "approved": approved, "schema_version": schema_version,
        "splits": split_names,
    })
    return root


def train_config(tmp_path, *, snapshot, release, run_dir=None, **overrides):
    values = {
        "snapshot": str(snapshot), "release": str(release),
        "run_dir": str(run_dir if run_dir is not None else tmp_path / "run"),
        "seed": 7, "epochs": 1, "global_batch": 2, "micro_cap": 2,
        "token_budget": 64, "max_length": 32, "learning_rate": 1e-2,
        "warmup_ratio": 0.5, "dtype": "float32", "device": "cpu",
        "shard": 1, "replicate": 1, "gradient_checkpointing": False,
        "eval_every": 1, "checkpoint_every": 1, "keep_checkpoints": 10,
    }
    values.update(overrides)
    return TrainConfig(**values)


def read_events(run_dir):
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def checkpoint_state(checkpoint_dir, snapshot):
    """Reshard a sealed training checkpoint into one full CPU model."""
    model = load_model_from_checkpoint(checkpoint_dir, snapshot, initialization="autojev")
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_learning_rate_schedule_is_a_pure_function_of_update():
    def rate(update, total=10, warmup=3):
        return learning_rate(update=update, total_updates=total, peak=1e-2,
                             warmup_updates=warmup, min_ratio=0.1)

    assert rate(0) == pytest.approx(1e-2 / 3)
    assert rate(2) == pytest.approx(1e-2)
    assert rate(3) == pytest.approx(1e-2)
    assert 1e-3 <= rate(9) < rate(6) < 1e-2
    assert learning_rate(update=3, total_updates=3, peak=1e-2, warmup_updates=3,
                         min_ratio=0.0) == pytest.approx(1e-2)
    arguments = {"update": 1, "total_updates": 10, "peak": 1e-2,
                 "warmup_updates": 3, "min_ratio": 0.1}
    for key, value in (("update", -1), ("total_updates", 0), ("warmup_updates", 0)):
        with pytest.raises(ValueError):
            learning_rate(**{**arguments, key: value})


def test_load_release_split_reads_only_approved_training_releases(tmp_path):
    release = release_fixture(tmp_path / "release")
    rows, identity = load_release_split(release, "train")
    assert [row["id"] for row in rows] == ["train-000", "train-001", "train-002"]
    assert all(row["split"] == "train" for row in rows)
    assert len(identity["manifest"]) == 64
    assert load_release_split(release, "development")[1] == identity
    for split in ("calibration", "test", "other"):
        with pytest.raises(ValueError, match="cannot be loaded"):
            load_release_split(release, split)


def test_load_release_split_enforces_the_release_contract(tmp_path):
    with pytest.raises(ValueError, match="approved"):
        load_release_split(
            release_fixture(tmp_path / "unapproved", approved=False), "train")
    with pytest.raises(ValueError, match="approved"):
        load_release_split(
            release_fixture(tmp_path / "draft", kind="dataset_draft"), "train")
    with pytest.raises(ValueError, match="approved"):
        load_release_split(
            release_fixture(tmp_path / "legacy", schema_version=1), "train")
    with pytest.raises(ValueError, match="canonical"):
        load_release_split(
            release_fixture(tmp_path / "renamed", split_names={
                "train": "tr.jsonl", "development": "development.jsonl",
            }), "train")
    with pytest.raises(ValueError, match="does not contain"):
        load_release_split(
            release_fixture(tmp_path / "train-only", splits=("train",)), "development")
    with pytest.raises(ValueError, match="audit and registry"):
        load_release_split(
            release_fixture(tmp_path / "unbound", manifest_files=[
                "train.jsonl", "development.jsonl", "audit.json",
            ]), "train")


def test_load_release_split_enforces_row_level_invariants(tmp_path):
    def relabel(split, lines):
        if split == "train":
            row = json.loads(lines[0])
            row["split"] = "development"
            lines = [json.dumps(row, ensure_ascii=False), *lines[1:]]
        return lines

    def demote(split, lines):
        if split == "train":
            row = json.loads(lines[0])
            row["quality"]["tier"] = "bronze"
            lines = [json.dumps(row, ensure_ascii=False), *lines[1:]]
        return lines

    def blank(split, lines):
        return lines + [""] if split == "train" else lines

    def empty(split, lines):
        return [] if split == "development" else lines

    with pytest.raises(ValueError, match="filed under"):
        load_release_split(release_fixture(tmp_path / "relabeled", lines_hook=relabel),
                           "train")
    with pytest.raises(ValueError, match="tier"):
        load_release_split(release_fixture(tmp_path / "demoted", lines_hook=demote),
                           "train")
    with pytest.raises(ValueError, match="Blank"):
        load_release_split(release_fixture(tmp_path / "blank", lines_hook=blank), "train")
    with pytest.raises(ValueError, match="empty"):
        load_release_split(release_fixture(tmp_path / "empty-dev", lines_hook=empty),
                           "development")


def test_build_optimizer_splits_decay_groups_and_requires_trainables():
    model = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    config = TrainConfig(snapshot="/s", release="/r", run_dir="/run", weight_decay=0.1)
    optimizer = build_optimizer(model, config)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert {id(parameter) for parameter in decay["params"]} == {id(model[0].weight)}
    assert {id(parameter) for parameter in no_decay["params"]} == {
        id(model[0].bias), id(model[1].weight), id(model[1].bias),
    }
    frozen = nn.Linear(2, 2)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable"):
        build_optimizer(frozen, config)


def test_snapshot_identity_is_manifest_sealed_or_caller_trusted(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert snapshot_identity(plain) == {"kind": "caller-trusted-local", "root": str(plain)}
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    atomic_write_json(sealed / "manifest.json",
                      {"schema_version": 1, "files": {}, "metadata": {}})
    identity = snapshot_identity(sealed)
    assert identity["kind"] == "manifest" and len(identity["sha256"]) == 64
    assert identity == snapshot_identity(sealed)


def test_world_one_training_end_to_end_and_fresh_run_determinism(
    tmp_path, trainer_runtime, world_one,
):
    snapshot, _, _ = snapshot_fixture(tmp_path)
    release = release_fixture(tmp_path / "release")
    config_a = train_config(tmp_path, snapshot=snapshot, release=release,
                            run_dir=tmp_path / "run-a")
    trainer = build_trainer(config_a)
    assert (trainer.rank, trainer.world_size) == (0, 1)
    assert trainer.total_updates == 2 and trainer.warmup_updates == 1
    result = trainer.run()
    assert result == {"updates": 2, "examples": 3}
    events = read_events(tmp_path / "run-a")
    assert [entry["event"] for entry in events] == [
        "start", "update", "evaluation", "checkpoint",
        "update", "evaluation", "checkpoint", "done",
    ]
    updates = [entry for entry in events if entry["event"] == "update"]
    assert [entry["examples"] for entry in updates] == [2, 1]
    assert sum(entry["examples"] for entry in updates) == result["examples"]
    for entry in updates:
        assert math.isfinite(entry["loss"]) and entry["grad_norm"] > 0.0
        assert entry["learning_rate"] == pytest.approx(1e-2)
    evaluations = [entry for entry in events if entry["event"] == "evaluation"]
    assert [entry["count"] for entry in evaluations] == [2, 2]
    assert all(0.0 <= entry["accuracy"] <= 1.0 for entry in evaluations)
    checkpoints = find_checkpoints(tmp_path / "run-a" / "checkpoints")
    assert [update for update, _ in checkpoints] == [1, 2]
    descriptor = json.loads(
        (checkpoints[-1][1] / "descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["run"]["final"] is True
    assert descriptor["run"]["cursor"] == {
        "epoch": 1, "window_index": 0, "update": 2, "consumed": 3,
    }
    assert descriptor["identity"]["world_size"] == 1
    assert descriptor["identity"]["mesh"] == [1, 1]
    # A second fresh run over identical inputs must land on identical weights.
    config_b = train_config(tmp_path, snapshot=snapshot, release=release,
                            run_dir=tmp_path / "run-b")
    assert build_trainer(config_b).run() == result
    state_a = checkpoint_state(latest_checkpoint(tmp_path / "run-a" / "checkpoints"), snapshot)
    state_b = checkpoint_state(latest_checkpoint(tmp_path / "run-b" / "checkpoints"), snapshot)
    assert state_a.keys() == state_b.keys()
    for key, value in state_a.items():
        assert torch.equal(value, state_b[key]), key


def test_resume_from_a_periodic_checkpoint_replays_identically(
    tmp_path, trainer_runtime, world_one,
):
    snapshot, _, _ = snapshot_fixture(tmp_path)
    release = release_fixture(tmp_path / "release")
    build_trainer(train_config(tmp_path, snapshot=snapshot, release=release,
                               run_dir=tmp_path / "run-a")).run()
    periodic = tmp_path / "run-a" / "checkpoints" / "step_000000000001"
    config_c = train_config(tmp_path, snapshot=snapshot, release=release,
                            run_dir=tmp_path / "run-c", resume=str(periodic))
    trainer = build_trainer(config_c)
    assert trainer.resumed_from == str(periodic)
    assert trainer.cursor == Cursor(0, 1, 1, 2)
    assert trainer.run() == {"updates": 2, "examples": 3}
    start = read_events(tmp_path / "run-c")[0]
    assert start["event"] == "start" and start["resumed_from"] == str(periodic)
    final_a = checkpoint_state(latest_checkpoint(tmp_path / "run-a" / "checkpoints"), snapshot)
    final_c = checkpoint_state(latest_checkpoint(tmp_path / "run-c" / "checkpoints"), snapshot)
    for key, value in final_a.items():
        assert torch.equal(value, final_c[key]), key


def test_resume_latest_after_completion_is_an_idempotent_no_op(
    tmp_path, trainer_runtime, world_one,
):
    snapshot, _, _ = snapshot_fixture(tmp_path)
    release = release_fixture(tmp_path / "release")
    run_dir = tmp_path / "run-a"
    build_trainer(train_config(tmp_path, snapshot=snapshot, release=release,
                               run_dir=run_dir)).run()
    final = latest_checkpoint(run_dir / "checkpoints")
    resumed = build_trainer(train_config(tmp_path, snapshot=snapshot, release=release,
                                         run_dir=run_dir, resume="latest"))
    assert resumed.resumed_from == str(final)
    assert resumed.run() == {"updates": 2, "examples": 3}
    names = [entry["event"] for entry in read_events(run_dir)]
    assert names[-3:] == ["start", "checkpoint_exists", "done"]
    assert [update for update, _ in find_checkpoints(run_dir / "checkpoints")] == [1, 2]


def test_build_trainer_rejects_missing_resume_and_bad_meshes(
    tmp_path, trainer_runtime, world_one,
):
    snapshot, _, _ = snapshot_fixture(tmp_path)
    release = release_fixture(tmp_path / "release")
    config = train_config(tmp_path, snapshot=snapshot, release=release,
                          run_dir=tmp_path / "run", resume="latest")
    with pytest.raises(ValueError, match="No sealed checkpoint"):
        build_trainer(config)
    broken = train_config(tmp_path, snapshot=snapshot, release=release,
                          run_dir=tmp_path / "run2", shard=2, replicate=1)
    with pytest.raises(ValueError, match="world_size"):
        build_trainer(broken)
