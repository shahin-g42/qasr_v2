from __future__ import annotations

import random

import pytest
import torch
from torch import nn

from jev import checkpoint
from jev.checkpoint import (
    capture_rng,
    checkpoint_matches,
    find_checkpoints,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng,
    save_checkpoint,
)
from jev_contracts import ArtifactError, atomic_write_json, write_manifest


def stepped_model(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).pow(2).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def seal_descriptor(directory, descriptor, metadata):
    directory.mkdir(parents=True)
    atomic_write_json(directory / "descriptor.json", descriptor)
    write_manifest(directory, ["descriptor.json"], metadata)


def test_rng_tensor_encoding_roundtrip_and_validation():
    tensor = torch.arange(24, dtype=torch.uint8).reshape(4, 6)
    encoded = checkpoint._encode_rng_tensor(tensor)
    assert encoded["shape"] == [4, 6]
    assert isinstance(encoded["data"], str)
    assert torch.equal(checkpoint._decode_rng_tensor(encoded), tensor)
    empty = checkpoint._decode_rng_tensor({"shape": [0], "data": ""})
    assert empty.dtype == torch.uint8 and empty.numel() == 0
    scalar = checkpoint._decode_rng_tensor({"shape": [], "data": "AA=="})
    assert scalar.shape == () and scalar.dtype == torch.uint8
    for malformed in (
        {"shape": [4], "data": encoded["data"]},
        {"shape": [-1], "data": ""},
        {"shape": [True], "data": ""},
        {"shape": "bad", "data": ""},
        {"shape": [2], "data": "!!not base64!!"},
        ["not", "a", "mapping"],
    ):
        with pytest.raises(ValueError):
            checkpoint._decode_rng_tensor(malformed)
    with pytest.raises(ValueError, match="uint8"):
        checkpoint._encode_rng_tensor(torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="uint8"):
        checkpoint._encode_rng_tensor([0, 1])


def test_python_rng_state_roundtrip():
    encoded = checkpoint._encode_python_state(random.getstate())
    assert len(encoded) == 3 and isinstance(encoded[1], list)
    random.setstate(checkpoint._decode_python_state(encoded))
    assert checkpoint._encode_python_state(random.getstate()) == encoded


def test_capture_and_restore_rng_streams():
    torch.manual_seed(1234)
    state = capture_rng("cpu")
    assert set(state) == {"torch_cpu", "device_rng", "python"}
    expected = torch.rand(4)
    torch.rand(4)
    torch.manual_seed(7)
    torch.rand(9)
    restore_rng(state, "cpu")
    assert torch.equal(torch.rand(4), expected)
    with pytest.raises(ValueError, match="RNG state"):
        restore_rng(["not", "a", "mapping"], "cpu")


def test_save_and_load_roundtrip_restores_every_stream(tmp_path):
    model, optimizer = stepped_model(11)
    directory = tmp_path / "step_000000000001"
    run_state = {
        "cursor": {"epoch": 0, "window_index": 1, "update": 1, "consumed": 2},
        "update": 1, "final": False, "rng": capture_rng("cpu"),
    }
    identity = {"config": "abc", "release": "def"}
    descriptor = save_checkpoint(
        directory, model=model, optimizer=optimizer, run_state=run_state,
        identity=identity, mesh_shape=(1, 1), update=1,
    )
    assert descriptor["update"] == 1 and descriptor["identity"] == identity
    assert descriptor["mesh"] == [1, 1] and descriptor["world_size"] == 1
    assert isinstance(descriptor["torch"], str) and descriptor["torch"]
    assert (directory / "manifest.json").is_file()
    assert checkpoint_matches(directory, update=1, identity=identity)
    fresh, fresh_optimizer = stepped_model(99)
    assert any(not torch.equal(saved, other)
               for saved, other in zip(model.parameters(), fresh.parameters(), strict=True))
    restored = load_checkpoint(
        directory, model=fresh, optimizer=fresh_optimizer,
        identity=identity, mesh_shape=(1, 1),
    )
    assert restored == run_state
    for saved, other in zip(
        model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.equal(saved, other)
    for index, entry in optimizer.state_dict()["state"].items():
        loaded = fresh_optimizer.state_dict()["state"][index]
        assert entry["step"].equal(loaded["step"])
        assert torch.equal(entry["exp_avg"], loaded["exp_avg"])
        assert torch.equal(entry["exp_avg_sq"], loaded["exp_avg_sq"])


def test_save_rejects_existing_directories_and_bad_arguments(tmp_path):
    model, optimizer = stepped_model(3)
    arguments = {"model": model, "optimizer": optimizer, "identity": {"a": 1},
                 "run_state": {"a": 1}, "mesh_shape": (1, 1), "update": 0}
    save_checkpoint(tmp_path / "step_000000000000", **arguments)
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "step_000000000000", **arguments)
    for overrides in (
        {"run_state": {}}, {"identity": {}}, {"update": -1}, {"update": True},
        {"mesh_shape": (0, 1)}, {"mesh_shape": (1,)}, {"mesh_shape": (1, True)},
    ):
        with pytest.raises(ValueError):
            save_checkpoint(tmp_path / "unused", **{**arguments, **overrides})


def test_load_verifies_identity_and_mesh_before_restoring(tmp_path):
    model, optimizer = stepped_model(5)
    directory = tmp_path / "step_000000000002"
    identity = {"config": "a", "seed": 1}
    save_checkpoint(directory, model=model, optimizer=optimizer, run_state={"update": 2},
                    identity=identity, mesh_shape=(1, 1), update=2)
    fresh, fresh_optimizer = stepped_model(6)
    with pytest.raises(ValueError, match=r"identity mismatch in \['config'\]"):
        load_checkpoint(directory, model=fresh, optimizer=fresh_optimizer,
                        identity={"config": "b", "seed": 1}, mesh_shape=(1, 1))
    with pytest.raises(ValueError, match="cross-world"):
        load_checkpoint(directory, model=fresh, optimizer=fresh_optimizer,
                        identity=identity, mesh_shape=(2, 1))
    payload = next(directory.glob("*.distcp"))
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError, match="integrity"):
        load_checkpoint(directory, model=fresh, optimizer=fresh_optimizer,
                        identity=identity, mesh_shape=(1, 1))


def test_load_rejects_malformed_descriptors_and_foreign_manifests(tmp_path):
    model, optimizer = stepped_model(7)
    good = {"schema_version": 1, "kind": "jev-checkpoint", "update": 4,
            "identity": {"a": 1}, "mesh": [1, 1], "run": {"update": 4},
            "world_size": 1, "torch": "test"}
    malformed = {key: value for key, value in good.items() if key != "run"}
    seal_descriptor(tmp_path / "malformed", malformed,
                    {"kind": "jev-checkpoint", "descriptor": malformed})
    with pytest.raises(ValueError, match="Malformed"):
        load_checkpoint(tmp_path / "malformed", model=model, optimizer=optimizer,
                        identity={"a": 1}, mesh_shape=(1, 1))
    seal_descriptor(tmp_path / "foreign", good, {"kind": "jev-model-components",
                                                 "descriptor": good})
    with pytest.raises(ValueError, match="Not a sealed jev checkpoint"):
        load_checkpoint(tmp_path / "foreign", model=model, optimizer=optimizer,
                        identity={"a": 1}, mesh_shape=(1, 1))


def test_discovery_pruning_and_match_predicates(tmp_path):
    root = tmp_path / "checkpoints"
    model, optimizer = stepped_model(8)
    for update in (1, 2, 3):
        save_checkpoint(root / f"step_{update:012d}", model=model, optimizer=optimizer,
                        run_state={"update": update}, identity={"config": "x"},
                        mesh_shape=(1, 1), update=update)
    orphan = root / "step_000000000004"
    orphan.mkdir()
    atomic_write_json(orphan / "descriptor.json",
                      {"schema_version": 1, "kind": "jev-checkpoint"})
    found = find_checkpoints(root)
    assert [update for update, _ in found] == [1, 2, 3]
    assert latest_checkpoint(root) == found[-1][1]
    assert find_checkpoints(tmp_path / "absent") == []
    assert latest_checkpoint(tmp_path / "absent") is None
    sealed = found[0][1]
    assert checkpoint_matches(sealed, update=1, identity={"config": "x"})
    assert not checkpoint_matches(sealed, update=2, identity={"config": "x"})
    assert not checkpoint_matches(sealed, update=1, identity={"config": "y"})
    assert not checkpoint_matches(orphan, update=4, identity={})
    with pytest.raises(ValueError):
        checkpoint_matches(sealed, update=True, identity={})
    with pytest.raises(ValueError):
        checkpoint_matches(sealed, update=1, identity=None)
    prune_checkpoints(root, keep=2)
    assert [update for update, _ in find_checkpoints(root)] == [2, 3]
    assert orphan.is_dir()
    with pytest.raises(ValueError):
        prune_checkpoints(root, keep=0)
