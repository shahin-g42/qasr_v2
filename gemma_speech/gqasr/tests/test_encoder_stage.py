"""Continuation scope, real frontend backward, optimizer and exact resume checks."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_model

from gqasr.augmentation import augmentation_rng
from gqasr._qasr.augmentation import SpecAugment
from gqasr.backward_audit import audit_batches
from gqasr.config import load_config
from gqasr.data import StratifiedUpdateSampler, data_contract
from gqasr.gradients import clip_gradients
from gqasr.modeling import GQASRForSpeechRecognition
from gqasr.optimization import build_optimizer, training_scope
from gqasr.processing import GQASRFeatureExtractor
from gqasr.train import train
from test_training import fixture, tiny_model


CONFIGS = Path(__file__).parents[1] / "configs"


def deep_model():
    model = tiny_model()
    model.config.acoustic_config["num_hidden_layers"] = 12
    return type(model)(model.config)


def test_continuation_recipe_preserves_index_and_augmentation():
    previous = load_config(CONFIGS / "02_full_b40_12node.yaml")
    cfg = load_config(CONFIGS / "03_encoder12_b40_12node.yaml")
    assert cfg["base_model"] == str(Path(previous["output_dir"]) / "final")
    assert cfg["output_dir"] != previous["output_dir"]
    assert data_contract(cfg) == data_contract(previous)
    assert cfg["index_path"] == previous["index_path"]
    assert cfg["augmentation"] == previous["augmentation"]
    assert cfg["code_switch_index"] == previous["code_switch_index"]
    assert not cfg.get("streaming_datasets")
    assert cfg["language_batch_counts"] == previous["language_batch_counts"]
    assert cfg["max_steps"] == 31446
    assert cfg["expected_world_size"] * cfg["batch_size"] * cfg["accumulation_steps"] == 3840
    assert cfg["acoustic_layers"] == 12 and not cfg["train_acoustic_frontend"]
    assert cfg["data_seed"] == previous["data_seed"]  # unchanged validation subset
    assert cfg["training_data_seed"] != cfg["data_seed"]
    assert cfg["eval_at_start"] and cfg["eval_steps"] == cfg["save_steps"] == 500
    assert cfg["warmup_steps"] + cfg["decay_steps"] == cfg["max_steps"]
    assert cfg["wandb_project"] == "gqasr"
    for key in ("learning_rate", "acoustic_learning_rate", "decoder_learning_rate", "embedding_learning_rate"):
        assert 0 < cfg[key] < previous[key]


def test_batch80_recipe_and_real_sampler_keep_language_proportions():
    from collections import Counter

    small = load_config(CONFIGS / "03_encoder12_b40_12node.yaml")
    large = load_config(CONFIGS / "03_encoder12_b80_12node.yaml")
    changed = {key for key in small if small[key] != large[key]}
    assert changed == {"batch_size", "language_batch_counts", "output_dir", "run_name",
                       "online_code_switching", "code_switch_probability"}
    assert not large["online_code_switching"] and large["code_switch_probability"] == 0.0
    assert small["online_code_switching"] and small["code_switch_probability"] == 0.4
    assert large["batch_size"] == 80 and large["accumulation_steps"] == 1
    assert large["batch_size"] * large["expected_world_size"] == 7680
    assert large["max_steps"] == 31446
    assert large["max_steps"] * 7680 == 241505280
    assert data_contract(small) == data_contract(large)
    quotas = {language: 2 * count for language, count in small["language_batch_counts"].items()}
    assert large["language_batch_counts"] == quotas
    counts = dict.fromkeys(quotas, 10000)
    # Check every rank, two updates, including the seeds used by the GPU audit.
    for rank in range(96):
        batches = list(StratifiedUpdateSampler(counts, quotas, 80, 2, rank=rank, world=96,
                       seed=large["training_data_seed"], augmentation_seeds=True))
        assert len(batches) == 2
        for batch in batches:
            assert len(batch) == 80
            assert Counter(language for language, _, _ in batch) == quotas
        replay = [keys for _, _, keys in audit_batches(counts, large, rank, 1, 0, 2)]
        assert replay == batches


@pytest.mark.parametrize("layers", [4, 12])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_encoder_blocks_train_with_frozen_frontend_and_specaugment(layers, bf16, checkpointing):
    model = deep_model()
    model.configure_training(layers, 2, True, checkpointing, .1, train_acoustic_frontend=False)
    audio = model.gqasr_acoustic.core
    assert not audio.subsample_conv_projection.training
    assert all(layer.training == (i >= 12 - layers) for i, layer in enumerate(audio.layers))
    assert all(layer.gradient_checkpointing == (checkpointing and i >= 12 - layers)
               for i, layer in enumerate(audio.layers))
    cfg = load_config(CONFIGS / "03_encoder12_b40_12node.yaml")
    optimizer = build_optimizer(model, cfg)
    scope = training_scope(model, optimizer)
    assert scope["trainable_acoustic_block_indices"] == list(range(12 - layers, 12))
    assert not scope["acoustic_frontend_trainable"]
    assert scope["acoustic_output_projection_trainable"]
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in params}) == len(params)  # tied embeddings once
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = audio.layers[-1].state_dict()
    before = {name: p.clone() for name, p in before.items()}
    waves = [np.sin(np.arange(4800, dtype=np.float32) / 10) * .1,
             np.random.default_rng(42).normal(0, .01, 9600).astype(np.float32), np.zeros(3200, np.float32)]
    batch = GQASRFeatureExtractor()(waves, sampling_rate=16000, padding=True,
                                  truncation=False, max_length=None, return_tensors="pt")
    x, mask = batch["input_features"], batch["input_features_mask"].bool()
    with augmentation_rng([{"augmentation_seed": 3}]):
        augmented = SpecAugment(p=1)(x, mask)
    assert not torch.equal(x, augmented)
    torch.testing.assert_close(augmented[~mask], x[~mask], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        loss = model(augmented, mask, torch.tensor([[3, 1]] * 3), torch.ones(3, 2, dtype=torch.bool))["loss"]
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None

    class Runtime:
        rank = 0

        @staticmethod
        def all_finite(value):
            return bool(torch.isfinite(value).all())

    assert torch.isfinite(clip_gradients(model, cfg["grad_clip"], Runtime(), 0))
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    assert any(not torch.equal(p, before[name]) for name, p in audio.layers[-1].state_dict().items())
    model.eval()
    assert not any(layer.training for layer in audio.layers)
    model.train()
    assert not audio.subsample_conv_projection.training
    assert all(layer.training for layer in audio.layers[-layers:])


def test_frontend_legacy_behavior_and_invalid_selection():
    model = deep_model()
    model.configure_training(12)
    assert all(p.requires_grad for p in model.gqasr_acoustic.parameters())
    with pytest.raises(ValueError, match="requires all acoustic layers"):
        model.configure_training(4, train_acoustic_frontend=True)


def test_new_training_draws_and_audit_match_without_changing_eval_seed():
    cfg = load_config(CONFIGS / "03_encoder12_b40_12node.yaml")
    counts = dict.fromkeys(cfg["language_batch_counts"], 10000)
    _, _, replay = next(audit_batches(counts, cfg, 5, 1, 0, 1))
    def keys(seed):
        return next(iter(StratifiedUpdateSampler(counts, cfg["language_batch_counts"], 40, 1,
                         rank=5, world=96, seed=seed, augmentation_seeds=True)))
    assert replay == keys(cfg["training_data_seed"])
    assert replay != keys(cfg["data_seed"])


def test_encoder_continuation_initial_eval_and_exact_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    path, cfg = fixture(tmp_path)
    # Keep the fixture's tokenizer/frontend/index; replace its tiny two-block
    # artifact with a tiny twelve-block artifact to exercise the real trainer.
    model = deep_model()
    base = Path(cfg["base_model"])
    (base / "config.json").write_text(json.dumps(asdict(model.config)))
    save_model(model, str(base / "model.safetensors"))
    cfg.update(acoustic_layers=12, decoder_layers=2, train_embeddings=True,
               train_acoustic_frontend=False, eval_at_start=True, training_data_seed=43)
    path.write_text(yaml.safe_dump(cfg))
    initial_frontend = {name: p.clone() for name, p in model.gqasr_acoustic.core.subsample_conv_projection.state_dict().items()}
    train(path)
    records = [json.loads(line) for line in (tmp_path / "out/metrics.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "initial_validation" and records[0]["step"] == 0
    assert "rank0_last_microbatch_loss" not in records[0]
    assert (tmp_path / "out/predictions-0.jsonl").exists()
    cfg["output_dir"] = str(tmp_path / "resumed")
    path.write_text(yaml.safe_dump(cfg))
    train(path, tmp_path / "out/checkpoint-00000001")
    original = GQASRForSpeechRecognition.from_pretrained(tmp_path / "out/final")
    resumed = GQASRForSpeechRecognition.from_pretrained(tmp_path / "resumed/final")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    for key, value in original.gqasr_acoustic.core.subsample_conv_projection.state_dict().items():
        torch.testing.assert_close(value, initial_frontend[key], rtol=0, atol=0)
    resumed_records = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert all(row["step"] > 0 for row in resumed_records)
