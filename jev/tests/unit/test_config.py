from __future__ import annotations

from pathlib import Path

import pytest

from jev.config import TrainConfig


def config(**overrides):
    values = {"snapshot": "/snap", "release": "/rel", "run_dir": "/run"}
    values.update(overrides)
    return TrainConfig(**values)


def test_defaults_and_mesh_shape():
    base = config()
    assert base.initialization == "autojev"
    assert (base.epochs, base.global_batch, base.micro_cap) == (1, 256, 2)
    assert (base.dtype, base.device, base.shard, base.replicate) == ("bfloat16", "cuda", 8, 8)
    assert base.gradient_checkpointing is True and base.foreach_optimizer is True
    assert (base.eval_every, base.checkpoint_every, base.keep_checkpoints) == (50, 50, 2)
    assert base.mesh_shape == (8, 8)
    assert TrainConfig(**base.to_dict()) == base


def test_identity_excludes_locations_and_binds_training_fields():
    base = config()
    relocated = config(run_dir="/elsewhere", checkpoint_dir="/ckpts", resume="/ckpts/step_1")
    assert relocated.identity() == base.identity()
    assert relocated.identity_hash == base.identity_hash
    excluded = {"run_dir", "checkpoint_dir", "resume"}
    assert set(base.identity()) == set(base.to_dict()) - excluded
    for field, value in (
        ("seed", 1), ("epochs", 3), ("global_batch", 8), ("learning_rate", 1e-3),
        ("dtype", "float32"), ("shard", 4), ("replicate", 4), ("initialization", "fresh"),
    ):
        changed = config(**{field: value})
        assert changed.identity_hash != base.identity_hash, field
    assert config(seed=0).identity_hash == config(seed=0).identity_hash


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"seed": True}, "integer"), ({"seed": -1}, "seed"), ({"seed": 1.5}, "integer"),
        ({"epochs": 0}, "epochs"), ({"global_batch": True}, "integer"),
        ({"global_batch": 0}, "global_batch"), ({"micro_cap": 0}, "micro_cap"),
        ({"token_budget": 0}, "token_budget"), ({"max_length": 0}, "max_length"),
        ({"max_length": 8193}, "max_length"), ({"token_budget": 8191}, "token_budget"),
        ({"learning_rate": 0.0}, "learning_rate"), ({"learning_rate": -1.0}, "learning_rate"),
        ({"learning_rate": True}, "learning_rate"),
        ({"lr_min_ratio": -0.1}, "lr_min_ratio"), ({"lr_min_ratio": 1.1}, "lr_min_ratio"),
        ({"warmup_ratio": 0.0}, "warmup_ratio"), ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"weight_decay": -1.0}, "weight_decay"), ({"beta1": 0.0}, "beta1"),
        ({"beta1": 1.0}, "beta1"), ({"beta2": 0.0}, "beta2"), ({"epsilon": 0.0}, "epsilon"),
        ({"clip_norm": 0.0}, "clip_norm"), ({"timeout_minutes": 0.0}, "timeout"),
        ({"dtype": "float16"}, "dtype"), ({"device": "tpu"}, "device"),
        ({"gradient_checkpointing": 1}, "boolean"), ({"foreach_optimizer": 0}, "boolean"),
        ({"shard": 0}, "shard"), ({"replicate": True}, "replicate"),
        ({"eval_every": -1}, "eval_every"), ({"checkpoint_every": 0}, "checkpoint_every"),
        ({"keep_checkpoints": 0}, "keep_checkpoints"),
        ({"initialization": "other"}, "initialization"),
        ({"snapshot": ""}, "snapshot"), ({"run_dir": " "}, "run_dir"),
        ({"checkpoint_dir": "  "}, "checkpoint_dir"), ({"resume": " "}, "resume"),
    ],
)
def test_invalid_values_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


def test_weakly_inclusive_bounds():
    assert config(lr_min_ratio=0.0).lr_min_ratio == 0.0
    assert config(lr_min_ratio=1.0).lr_min_ratio == 1.0
    assert config(weight_decay=0.0).weight_decay == 0.0
    assert config(token_budget=8192).token_budget == 8192


def test_from_toml_roundtrip_and_strictness(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text(
        'snapshot = "/snap"\nrelease = "/rel"\nrun_dir = "/run"\nseed = 5\nepochs = 2\n',
        encoding="utf-8",
    )
    loaded = TrainConfig.from_toml(path)
    assert loaded == TrainConfig(snapshot="/snap", release="/rel", run_dir="/run", seed=5, epochs=2)
    path.write_text(
        'snapshot = "/snap"\nrelease = "/rel"\nrun_dir = "/run"\nwho = 1\n', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_toml(path)
    path.write_text('snapshot = "/snap"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        TrainConfig.from_toml(path)
    with pytest.raises(FileNotFoundError):
        TrainConfig.from_toml(tmp_path / "absent.toml")


def test_shipped_8x8_h100_config_loads_with_the_production_mesh():
    """The launcher and preflight depend on this file staying valid."""
    path = Path(__file__).resolve().parents[2] / "configs" / "train" / "h100_8x8.toml"
    config = TrainConfig.from_toml(path)
    assert config.mesh_shape == (8, 8) and config.shard * config.replicate == 64
    assert (config.device, config.dtype) == ("cuda", "bfloat16")
    assert (config.global_batch, config.micro_cap, config.token_budget) == (256, 8, 8192)
    assert config.token_budget >= config.max_length
    assert config.gradient_checkpointing is True and config.initialization == "autojev"
