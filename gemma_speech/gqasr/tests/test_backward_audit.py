"""Depth/precision/zero-feature probes; reproduce the acoustic failure, not hide it."""

import pytest
import torch
from test_training import tiny_model


@pytest.mark.parametrize("mode", ["bridge", "decoder", "acoustic", "full"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_deep_variable_length_backward(mode, checkpointing, bf16):
    model = tiny_model()
    model.config.acoustic_config["num_hidden_layers"] = 12
    model = type(model)(model.config)
    model.configure_training(
        12 if mode in ("acoustic", "full") else 0,
        2 if mode in ("decoder", "full") else 0,
        mode in ("decoder", "full"), checkpointing, 0.1,
    )
    features = torch.randn(2, 141, 128)
    mask = torch.arange(141)[None] < torch.tensor([9, 141])[:, None]
    features[~mask] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        loss = model(features, mask, torch.tensor([[3, 1], [3, 1]]), torch.ones(2, 2, dtype=torch.bool))["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_reproduces_zero_feature_acoustic_instability():
    # An explicit reproducer of an unresolved failure mode in the inherited encoder.
    # This is not a claim that real silence produces zero frontend features.
    model = tiny_model()
    model.config.acoustic_config["num_hidden_layers"] = 12
    model = type(model)(model.config)
    batch = (
        torch.zeros(2, 141, 128), torch.ones(2, 141, dtype=torch.bool),
        torch.tensor([[3, 1], [3, 1]]), torch.ones(2, 2, dtype=torch.bool),
    )
    model.configure_training()
    model(*batch)["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.configure_training(12, 2, True, False, 0.1)
    loss = model(*batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.gqasr_acoustic.parameters()
    )


def test_checkpointing_preserves_deep_gradients():
    baseline = tiny_model()
    baseline.config.acoustic_config["num_hidden_layers"] = 12
    baseline = type(baseline)(baseline.config)
    checkpointed = type(baseline)(baseline.config)
    checkpointed.load_state_dict(baseline.state_dict())
    features = torch.randn(2, 141, 128)
    mask = torch.arange(141)[None] < torch.tensor([9, 141])[:, None]
    features[~mask] = 0
    losses = []
    for model, enabled in ((baseline, False), (checkpointed, True)):
        model.configure_training(12, 2, True, enabled, 0.1)
        loss = model(features, mask, torch.tensor([[3, 1], [3, 1]]), torch.ones(2, 2, dtype=torch.bool))["loss"]
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for (name, p), (other_name, q) in zip(baseline.named_parameters(), checkpointed.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_zero_failure_exists_without_hybrid():
    from transformers import Gemma4AudioConfig, Gemma4AudioModel

    config = dict(tiny_model().config.acoustic_config)
    config["num_hidden_layers"] = 12
    encoder = Gemma4AudioModel(Gemma4AudioConfig(**config))
    output = encoder(
        input_features=torch.zeros(2, 141, 128),
        attention_mask=torch.ones(2, 141, dtype=torch.bool),
    ).last_hidden_state
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in encoder.parameters()
    )


def test_multi_rank_replay_matches_training_sampler():
    from gqasr.backward_audit import audit_batches
    from gqasr.data import StratifiedUpdateSampler

    cfg = {"language_batch_counts": {"ar": 2}, "batch_size": 2, "expected_world_size": 96, "data_seed": 42}
    replay = list(audit_batches({"ar": 1000}, cfg, 2, 3, 0, 1))
    assert [r for r, _, _ in replay] == [2, 3, 4]
    for rank, batch, keys in replay:
        expected = next(iter(StratifiedUpdateSampler(
            {"ar": 1000}, {"ar": 2}, 2, 1, rank=rank, world=96, seed=42, augmentation_seeds=True,
        )))
        assert batch == 0
        assert keys == expected
