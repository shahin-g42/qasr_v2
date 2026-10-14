from __future__ import annotations

import pytest
import torch
from torch import nn

from jev.distributed import (
    _Layout,
    backend_for,
    clip_gradients,
    decoder_blocks,
    hsdp_mesh,
    init_distributed,
    materialize_empty,
    shard_range,
    sync_replicated_gradients,
)
from jev.modeling import _Snapshot, build_meta_model, load_full_state

from .test_modeling import runtime as _runtime
from .test_modeling import snapshot_fixture
from .test_trainer import world_one as _world_one

# Cross-module fixtures re-exported under their fixture names so pytest resolves
# them here; the parameters below intentionally reuse those names.
runtime = _runtime
world_one = _world_one


def test_backend_for_device_types():
    assert backend_for("cuda") == "nccl"
    assert backend_for("cpu") == "gloo"
    for bad in ("mps", "", 3, None):
        with pytest.raises(ValueError):
            backend_for(bad)


def test_decoder_blocks_requires_a_real_block_sequence():
    class WithoutLanguageModel(nn.Module):
        def forward(self, x):
            return x

    class WithoutLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()

    class WithLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

    assert decoder_blocks(WithoutLanguageModel()) == []
    blocks = decoder_blocks(WithLayers())
    assert len(blocks) == 2 and all(isinstance(block, nn.Linear) for block in blocks)
    with pytest.raises(ValueError, match="Unsupported"):
        decoder_blocks(WithoutLayers())


def test_shard_range_follows_torch_chunk_ceil_semantics():
    assert [shard_range(10, 3, coordinate) for coordinate in range(3)] == [
        (0, 4), (4, 4), (8, 2),
    ]
    assert shard_range(2, 3, 0) == (0, 1)
    assert shard_range(1, 1, 0) == (0, 1)
    with pytest.raises(ValueError, match="no rows"):
        shard_range(2, 3, 2)
    for args in ((0, 1, 0), (10, 0, 0), (10, 3, 3), (10, 3, -1), (True, 1, 0)):
        with pytest.raises(ValueError):
            shard_range(*args)


def test_materialize_empty_only_targets_real_devices():
    with pytest.raises(ValueError, match="cuda/cpu"):
        materialize_empty(nn.Linear(2, 2), "meta")


def test_collectives_require_an_initialized_group():
    model = nn.Linear(2, 2)
    model(torch.ones(2)).sum().backward()
    with pytest.raises(ValueError, match="initialized"):
        sync_replicated_gradients(model, world_size=1)
    with pytest.raises(ValueError, match="initialized"):
        clip_gradients(model, 1.0, world_size=1)


def test_hsdp_mesh_rejects_mismatched_degrees_without_a_world():
    with pytest.raises(ValueError, match="world_size"):
        hsdp_mesh(device_type="cpu", shard=2, replicate=1, local_world_size=1)
    with pytest.raises(ValueError, match="world_size"):
        hsdp_mesh(device_type="cpu", shard=1, replicate=2, local_world_size=1)
    for shard, replicate in ((0, 1), (True, 1), (1, 0)):
        with pytest.raises(ValueError):
            hsdp_mesh(device_type="cpu", shard=shard, replicate=replicate,
                      local_world_size=1)


def test_init_distributed_argument_validation():
    for bad in (0, -1.0, 24 * 60 + 1, True):
        with pytest.raises(ValueError, match="timeout"):
            init_distributed(device_type="cpu", timeout_minutes=bad)


@pytest.mark.parametrize("initialization", ["autojev", "fresh"])
@pytest.mark.parametrize("sharded", [False, True])
def test_layout_rows_agree_byte_for_byte_with_load_full_state(
    tmp_path, runtime, initialization, sharded,
):
    path, _, _ = snapshot_fixture(tmp_path, initialization, sharded=sharded)
    meta = build_meta_model(path, initialization)
    state = load_full_state(path, initialization)
    with _Layout(_Snapshot(path), meta, initialization) as layout:
        for name, parameter in meta.named_parameters():
            rows = layout.rows(name, 0, int(parameter.shape[0]))
            assert torch.equal(rows, state[name]), name
        name = "backbone.language_model.embed_tokens.weight"
        assert torch.equal(layout.rows(name, 5, 3), state[name][5:8])
    with pytest.raises(ValueError, match="offset"):
        _Layout(_Snapshot(path), meta, initialization).rows(name, -1, 1)


def test_init_distributed_and_hsdp_mesh_world_one(world_one):
    assert init_distributed(device_type="cpu") == (0, 1, 0)
    mesh = hsdp_mesh(device_type="cpu", shard=1, replicate=1, local_world_size=1)
    assert tuple(mesh.mesh.shape) == (1, 1)
    assert mesh.mesh_dim_names == ("dp_replicate", "dp_shard")


def test_init_distributed_rejects_bad_local_layouts(world_one, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="divide"):
        init_distributed(device_type="cpu")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "5")
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        init_distributed(device_type="cpu")


def test_hsdp_mesh_requires_node_local_sharding(world_one):
    init_distributed(device_type="cpu")
    with pytest.raises(ValueError, match="node-local"):
        hsdp_mesh(device_type="cpu", shard=1, replicate=1, local_world_size=2)


def test_gradient_sync_and_clipping_on_dense_parameters(world_one):
    init_distributed(device_type="cpu")
    model = nn.Linear(3, 2)
    model(torch.ones(3)).sum().backward()
    assert sync_replicated_gradients(model, world_size=1) == ["weight", "bias"]
    manual = (
        model.weight.grad.double().pow(2).sum() + model.bias.grad.double().pow(2).sum()
    ).sqrt()
    before = model.weight.grad.clone()
    assert clip_gradients(model, 1e9, world_size=1) == pytest.approx(manual.item(), rel=1e-6)
    assert torch.equal(model.weight.grad, before)
    assert clip_gradients(model, 1e-3, world_size=1) == pytest.approx(manual.item(), rel=1e-6)
    assert model.weight.grad.abs().max() < 1e-3
    assert model.weight.grad.abs().max() < before.abs().max()


def test_clipping_rejects_trainable_parameters_without_gradients(world_one):
    init_distributed(device_type="cpu")

    class Partial(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(2, 1)
            self.idle = nn.Linear(2, 1)

        def forward(self, x):
            return self.used(x)

    model = Partial()
    model(torch.ones(2)).sum().backward()
    with pytest.raises(ValueError, match="no gradient"):
        clip_gradients(model, 1.0, world_size=1)
    frozen = nn.Linear(2, 2)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="No gradients"):
        clip_gradients(frozen, 1.0, world_size=1)
    with pytest.raises(ValueError, match="max_norm"):
        clip_gradients(model, 0.0, world_size=1)
