"""Check actual backward recomputation, not just the outer checkpoint flag."""

import copy
from collections import Counter

import numpy as np
import pytest
import torch
from test_audio_types import artifact

from audm import AUDForConditionalGeneration, AUDProcessor


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("stage", ["adapter", "decoder", "full"])
def test_gemma_layer_recomputation_and_gradients(tmp_path, tokenizer, reentrant, stage):
    source = artifact(tmp_path, tokenizer, "audflash")
    model = AUDForConditionalGeneration.from_pretrained(source, attn_implementation="sdpa")
    model.configure_training(stage)
    model.train()
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    processor = AUDProcessor.from_pretrained(source)
    batch = processor.prepare_training_batch(
        [np.random.default_rng(i).normal(0, 0.02, n) for i, n in enumerate((3200, 8000))],
        ["hello", "a longer transcript"],
        ["en", "ml"],
    )
    calls = Counter()
    handles = []
    layers = {
        **{f"encoder.{i}": layer for i, layer in enumerate(model.aud_encoder.layers)},
        **{f"decoder.{i}": layer for i, layer in enumerate(model.aud_decoder.model.layers)},
    }
    for name, layer in layers.items():
        assert layer.gradient_checkpointing

        def count(module, inputs, name=name):
            calls[name] += 1

        handles.append(layer.register_forward_pre_hook(count))
    actual = model(**batch, loss_only=True, loss_smoothing=0.1).loss
    assert all(calls[name] == 1 for name in layers)
    actual.backward()
    for name in layers:
        assert calls[name] == (1 if name.startswith("encoder") and stage != "full" else 2)
    for handle in handles:
        handle.remove()
    expected = reference(**batch, loss_only=True, loss_smoothing=0.1).loss
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for (name, parameter), (_, original) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (parameter.grad is None) == (original.grad is None), name
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, original.grad, atol=2e-6, rtol=3e-4, msg=name
            )
    # Disable is reversible, and inference never recomputes frozen layers.
    model.gradient_checkpointing_disable()
    assert not any(layer.gradient_checkpointing for layer in layers.values())


def test_gemma_encoder_retains_fewer_tensors_for_backward(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audflash")
    model = AUDForConditionalGeneration.from_pretrained(source)
    model.configure_training("full")
    model.train()
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    processor = AUDProcessor.from_pretrained(source)
    batch = processor.prepare_training_batch([np.zeros(16000)], ["hello"], ["en"])

    def measure(candidate):
        sizes = []

        def pack(tensor):
            sizes.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            hidden, _ = candidate.encode_audio(
                batch["input_features"], batch["input_features_mask"]
            )
        hidden.square().mean().backward()
        return sum(sizes)

    assert measure(model) < measure(reference)


def test_selective_checkpoint_policy_recomputes_only_selected_layers(tmp_path, tokenizer):
    from audm.efficiency import apply_checkpoint_policy, resolve_efficiency

    source = artifact(tmp_path, tokenizer, "audflash")
    model = AUDForConditionalGeneration.from_pretrained(source)
    config = copy.deepcopy(model.config)
    config.audio_config.num_hidden_layers = 2
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "full_attention"]
    model = AUDForConditionalGeneration(config)
    model.configure_training("full")
    model.train()
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    policy = resolve_efficiency(
        dict(encoder_checkpoint_every=2, decoder_checkpoint_every=2), tmp_path
    )
    apply_checkpoint_policy(model, policy)
    processor = AUDProcessor.from_pretrained(source)
    batch = processor.prepare_training_batch(
        [np.random.default_rng(1).normal(0, 0.01, 6400)], ["hello"], ["en"]
    )
    counts = Counter()
    handles = []
    for kind, layers in (
        ("encoder", model.aud_encoder.layers),
        ("decoder", model.aud_decoder.model.layers),
    ):
        for index, layer in enumerate(layers):

            def count(module, inputs, key=(kind, index)):
                counts[key] += 1

            handles.append(layer.register_forward_pre_hook(count))
    model(**batch, loss_only=True).loss.backward()
    reference(**batch, loss_only=True).loss.backward()
    assert counts == {("encoder", 0): 2, ("encoder", 1): 1, ("decoder", 0): 2, ("decoder", 1): 1}
    for handle in handles:
        handle.remove()
    for (name, actual), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=3e-4, msg=name)
