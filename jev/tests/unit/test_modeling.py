from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from jev import modeling
from jev.codes import (
    AUTOJEV_MODEL,
    AUTOJEV_REVISION,
    AUTOJEV_SOURCE_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    discover_codes,
)
from jev.modeling import DecisionModel, build_meta_model, load_full_state, load_model

from .test_processing import FakeProcessor, make_row


class TinyConfig:
    def __init__(self, raw):
        self.raw = raw
        self.text_config = SimpleNamespace(**raw["text_config"])

    @classmethod
    def from_dict(cls, raw):
        return cls(raw)


class TinyBackboneOutput(NamedTuple):
    # NamedTuple, not SimpleNamespace: FSDP2 finds the differentiable backbone
    # outputs it must hook by walking dict/list/tuple/NamedTuple/dataclass
    # containers only, exactly like HF's OrderedDict-based ModelOutput. An
    # opaque namespace leaves managed parameters gradientless after backward.
    last_hidden_state: torch.Tensor


class TinyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(
            config.text_config.vocab_size, config.text_config.hidden_size,
        )
        self.visual = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        self.checkpointing = None

    def forward(self, input_ids, attention_mask, *, use_cache):
        assert use_cache is False
        return TinyBackboneOutput(
            last_hidden_state=self.language_model.embed_tokens(input_ids)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.checkpointing = gradient_checkpointing_kwargs

    def save_pretrained(self, directory, *, safe_serialization, max_shard_size):
        assert safe_serialization and max_shard_size == "5GB"
        directory = Path(directory)
        raw = dict(self.config.raw, architectures=["Qwen3_5Model"])
        (directory / "config.json").write_text(json.dumps(raw))
        save_file(self.state_dict(), str(directory / "model.safetensors"))


class ProcessorLoader:
    @staticmethod
    def from_pretrained(path, **kwargs):
        assert Path(path).is_dir()
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return FakeProcessor()


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(modeling, "_runtime", lambda: (ProcessorLoader, TinyConfig, TinyBackbone))


def snapshot_fixture(root, initialization="autojev", *, sharded=False):
    path = root / (AUTOJEV_REVISION if initialization == "autojev" else BASE_REVISION)
    path.mkdir()
    raw = {
        "model_type": "qwen3_5", "vision_config": {"model_type": "qwen3_5"},
        "architectures": ["Qwen3_5Model" if initialization == "autojev"
                          else "Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 4, "vocab_size": 1024, "intermediate_size": 8,
                        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
                        "mtp_num_hidden_layers": 1},
    }
    (path / "config.json").write_text(json.dumps(raw))
    backbone = TinyBackbone(TinyConfig(raw))
    head = torch.arange(255 * 4, dtype=torch.float32).reshape(255, 4) / 1000
    weights = dict(backbone.state_dict())
    if initialization == "fresh":
        weights = {"model." + key: value for key, value in weights.items()}
        weights["lm_head.weight"] = torch.arange(1024 * 4, dtype=torch.float32).reshape(1024, 4)
        head = weights["lm_head.weight"][discover_codes(FakeProcessor().tokenizer)[1]].clone()
    else:
        save_file({"weight": head}, str(path / "readout.safetensors"))
        codes, ids = discover_codes(FakeProcessor().tokenizer)
        (path / "decision_config.json").write_text(json.dumps({
            "format_version": 1, "base_model": BASE_MODEL, "revision": BASE_REVISION,
            "codes": codes, "token_ids": ids, "temperature": 2.207568021892729,
        }))
    if sharded:
        groups = [dict(list(weights.items())[::2]), dict(list(weights.items())[1::2])]
        index = {}
        for i, group in enumerate(groups):
            name = f"model-{i}.safetensors"
            save_file(group, str(path / name))
            index.update({key: name for key in group})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(weights, str(path / "model.safetensors"))
    return path, weights, head


@pytest.mark.parametrize("initialization", ["autojev", "fresh"])
@pytest.mark.parametrize("sharded", [False, True])
def test_full_state_meta_and_loading(tmp_path, runtime, initialization, sharded):
    path, weights, head = snapshot_fixture(tmp_path, initialization, sharded=sharded)
    meta = build_meta_model(path, initialization)
    assert all(parameter.is_meta for parameter in meta.parameters())
    state = load_full_state(path, initialization)
    assert all(value.device.type == "cpu" for value in state.values())
    assert state.keys() == meta.state_dict().keys()
    torch.testing.assert_close(state["readout.weight"], head)
    model = load_model(path, initialization=initialization, gradient_checkpointing=True)
    torch.testing.assert_close(model.readout.weight, head)
    for key, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, weights[("model." if initialization == "fresh" else "")
                                                 + key])
    assert model.backbone.checkpointing == {"use_reentrant": False}
    assert not any(p.requires_grad for p in model.backbone.visual.parameters())
    assert model.readout.weight.requires_grad
    assert model.temperature == (2.207568021892729 if initialization == "autojev" else 1.0)


def test_forward_finite_fp32_padding_and_prediction_modes(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    model = load_model(path, dtype=torch.bfloat16)
    batch = model.prepare([make_row(count=1), make_row(count=3)])
    logits = model(batch)
    assert logits.dtype == torch.float32 and logits.shape == (2, 255)
    assert torch.isfinite(logits).all()
    assert (logits[0, 1:] == -1e9).all()
    expected = model.readout(model.backbone.language_model.embed_tokens(
        batch.inputs["input_ids"][:, -1],
    )).float()
    torch.testing.assert_close(logits[1, :3], expected[1, :3])
    model.train()
    model.backbone.visual.eval()
    probabilities = model.predict([make_row(count=1), make_row(count=3)], temperature=1e100)
    assert probabilities[0] == [1.0]
    assert sum(probabilities[1]) == pytest.approx(1.0)
    assert model.training and not model.backbone.visual.training
    tiny_temperature = model.predict([make_row()], temperature=1e-320)[0]
    assert sum(tiny_temperature) == pytest.approx(1.0)
    for bad in (True, 0, float("inf")):
        with pytest.raises(ValueError):
            model.predict([], temperature=bad)
    with torch.no_grad():
        model.readout.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(batch)


def test_new_snapshot_export_reload_and_integrity(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    source_bytes = (path / "decision_config.json").read_bytes()
    model = load_model(path)
    model.metadata["custom"] = "retained"
    expected = model(model.prepare([make_row()]))
    destination = tmp_path / "export"
    model.save_pretrained(destination)
    assert (destination / "manifest.json").exists()
    assert (path / "decision_config.json").read_bytes() == source_bytes
    loaded = load_model(destination)
    assert loaded.metadata["custom"] == "retained"
    torch.testing.assert_close(loaded(loaded.prepare([make_row()])), expected)
    with pytest.raises(FileExistsError):
        model.save_pretrained(destination)
    with (destination / "readout.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_model(destination)


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "integer", "prefix"])
def test_strict_tensor_validation(tmp_path, runtime, damage):
    path, weights, _ = snapshot_fixture(tmp_path, "fresh")
    key = "model.language_model.embed_tokens.weight"
    if damage == "missing":
        weights.pop(key)
    elif damage == "extra":
        weights["not_a_backbone.weight"] = torch.ones(2)
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "integer":
        weights[key] = weights[key].long()
    else:
        weights[key.removeprefix("model.")] = weights.pop(key)
    save_file(weights, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="tensor"):
        load_full_state(path, "fresh")


def test_exact_auxiliary_mtp_layout_only(tmp_path, runtime):
    path, weights, head = snapshot_fixture(tmp_path, "fresh")
    config = TinyConfig(json.loads((path / "config.json").read_text())).text_config
    mtp = modeling._mtp_shapes(config)
    weights.update({key: torch.zeros(shape) for key, shape in mtp.items()})
    save_file(weights, str(path / "model.safetensors"))
    state = load_full_state(path, "fresh")
    assert not any("mtp" in key for key in state)
    torch.testing.assert_close(state["readout.weight"], head)
    weights["mtp.surprise.weight"] = torch.zeros(1)
    save_file(weights, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="extra"):
        load_full_state(path, "fresh")


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_index_path_traversal_rejected(tmp_path, runtime, filename):
    path, _, _ = snapshot_fixture(tmp_path)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"language_model.embed_tokens.weight": filename},
    }))
    with pytest.raises(ValueError):
        load_model(path)


def test_index_missing_duplicate_json_and_symlink(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path, sharded=True)
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["missing.weight"] = "model-0.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing"):
        load_model(path)
    index_path.write_text('{"weight_map": {}, "weight_map": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_model(path)
    linked = tmp_path / "linked"
    linked.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError):
        load_model(linked)


def test_mapping_and_incomplete_artifacts(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path)
    config_path = path / "decision_config.json"
    config = json.loads(config_path.read_text())
    config["token_ids"][0] = 999
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="vocabulary"):
        build_meta_model(path)
    (path / "jev_components.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete"):
        build_meta_model(path)


def test_meta_does_not_open_weights(tmp_path, runtime, monkeypatch):
    path, _, _ = snapshot_fixture(tmp_path)
    def no_weights(*args, **kwargs):
        raise AssertionError("Meta construction opened weights")
    monkeypatch.setattr(modeling, "safe_open", no_weights)
    build_meta_model(path)


def test_actual_tiny_qwen_meta_buffers_forward_backward(tmp_path, monkeypatch):
    from transformers import Qwen3_5Config, Qwen3_5Model

    config = Qwen3_5Config(
        text_config={
            "hidden_size": 32, "vocab_size": 1024, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
            "attn_output_gate": True, "output_gate_type": "swish", "mamba_ssm_dtype": "float32",
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16},
    )
    config._attn_implementation = "sdpa"
    backbone = Qwen3_5Model(config)
    processor = FakeProcessor()
    codes, ids = discover_codes(processor.tokenizer)
    original = DecisionModel(backbone, nn.Linear(32, 255, bias=False), processor, codes, ids)
    original.freeze_vision()
    original.eval()
    destination = tmp_path / "real-tiny"
    original.save_pretrained(destination)
    monkeypatch.setattr(
        modeling, "_runtime", lambda: (ProcessorLoader, Qwen3_5Config, Qwen3_5Model),
    )
    meta = build_meta_model(destination)
    assert all(parameter.is_meta for parameter in meta.parameters())
    assert all(not buffer.is_meta for buffer in meta.buffers())
    loaded = load_model(destination, gradient_checkpointing=True)
    rows = [make_row(count=2), make_row(count=3)]
    with torch.no_grad():
        expected = original(original.prepare(rows))
        actual = loaded(loaded.prepare(rows))
    torch.testing.assert_close(actual, expected)
    loaded.train()
    logits = loaded(loaded.prepare(rows))
    logits[:, :2].sum().backward()
    assert torch.isfinite(loaded.readout.weight.grad).all()
    meta.to_empty(device="cpu")
    meta.materialize_runtime_buffers("cpu")
    meta.load_state_dict(load_full_state(destination), strict=True)
    meta.eval()
    with torch.no_grad():
        torch.testing.assert_close(meta(meta.prepare(rows)), expected)


def test_release_manifest_checks_consumed_assets_not_optional_source_tree(tmp_path, runtime):
    import hashlib

    path, _, _ = snapshot_fixture(tmp_path)
    files = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in path.iterdir()
    }
    files["source/not-installed/LICENSE"] = {"sha256": "0" * 64, "bytes": 1}
    manifest = {
        "schema_version": 1, "model_repository": AUTOJEV_MODEL,
        "source_commit": AUTOJEV_SOURCE_REVISION, "base_revision": BASE_REVISION, "files": files,
    }
    (path / "release-manifest.json").write_text(json.dumps(manifest))
    assert load_model(path).metadata["snapshot_integrity"] == "manifest"
    with (path / "readout.safetensors").open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        load_full_state(path)


@pytest.mark.parametrize("damage", ["shape", "extra", "missing", "nan"])
def test_readout_validation(tmp_path, runtime, damage):
    path, _, head = snapshot_fixture(tmp_path)
    weights = {"weight": head}
    if damage == "shape":
        weights["weight"] = head[:-1]
    elif damage == "extra":
        weights["bias"] = torch.zeros(255)
    elif damage == "missing":
        weights = {"not_weight": head}
    else:
        weights["weight"][0, 0] = float("nan")
    save_file(weights, str(path / "readout.safetensors"))
    with pytest.raises(ValueError, match="Readout"):
        load_model(path)


def test_source_dtype_preserved_and_vision_guard(tmp_path, runtime):
    path, weights, _ = snapshot_fixture(tmp_path, "fresh")
    save_file({key: value.bfloat16() for key, value in weights.items()},
              str(path / "model.safetensors"))
    state = load_full_state(path, "fresh")
    assert {value.dtype for value in state.values()} == {torch.bfloat16}
    model = load_model(path, initialization="fresh")
    assert {value.dtype for value in model.parameters()} == {torch.float32}
    del model.backbone.visual
    with pytest.raises(ValueError, match="backbone.visual"):
        model.freeze_vision()


def test_source_pin_and_initialization_gate(tmp_path, runtime):
    path, _, _ = snapshot_fixture(tmp_path, "fresh")
    raw = json.loads((path / "config.json").read_text())
    raw["_commit_hash"] = "0" * 40
    (path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pinned"):
        build_meta_model(path, "fresh")
    with pytest.raises(ValueError, match="requires"):
        build_meta_model(path, "autojev")
