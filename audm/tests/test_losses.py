import copy

import numpy as np
import pytest
import torch
from qasr.losses import make_smoothed_causal_lm_loss
from test_models import encoder_source, gqasr_source

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.export import export_model
from audm.losses import chunked_causal_loss


def test_trainer_identifies_only_real_labels():
    from transformers.utils.generic import find_labels

    assert find_labels(AUDForConditionalGeneration) == ["labels"]


@pytest.mark.parametrize("epsilon", [0.0, 0.1])
@pytest.mark.parametrize("denominator", [None, 27])
@pytest.mark.parametrize("frozen_head", [False, True])
def test_loss_and_gradients_match_dense_with_padding(epsilon, denominator, frozen_head):
    torch.manual_seed(42)
    hidden = torch.randn(3, 11, 8, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(8, 47)
    head.requires_grad_(not frozen_head)
    reference_head = copy.deepcopy(head)
    labels = torch.randint(0, 47, (3, 11))
    labels[:, :4] = -100
    labels[0, -3:] = -100
    labels[2] = -100  # One empty row must not change reduction over real targets.
    dense = make_smoothed_causal_lm_loss(epsilon)(
        logits=reference_head(other),
        labels=labels,
        vocab_size=47,
        num_items_in_batch=denominator,
    )
    bounded = chunked_causal_loss(
        hidden, head, labels, chunk_size=3, epsilon=epsilon, num_items_in_batch=denominator
    )
    torch.testing.assert_close(bounded, dense)
    dense.backward()
    bounded.backward()
    torch.testing.assert_close(hidden.grad, other.grad)
    if not frozen_head:
        for actual, expected in zip(head.parameters(), reference_head.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)


def test_accumulation_matches_single_update_token_weighting():
    torch.manual_seed(2)
    head = torch.nn.Linear(5, 31)
    reference = copy.deepcopy(head)
    hidden = torch.randn(4, 8, 5, requires_grad=True)
    labels = torch.randint(0, 31, (4, 8))
    labels[:, :3] = -100
    labels[0, -3:] = -100
    count = labels[:, 1:].ne(-100).sum()
    expected = make_smoothed_causal_lm_loss(0.1)(
        logits=reference(hidden),
        labels=labels,
        vocab_size=31,
    )
    expected.backward()
    for x, y in zip(hidden.detach().split(2), labels.split(2), strict=True):
        chunked_causal_loss(
            x, head, y, chunk_size=3, epsilon=0.1, num_items_in_batch=count
        ).backward()
    for actual, wanted in zip(head.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)


def test_does_not_retain_vocabulary_logits_for_backward():
    head = torch.nn.Linear(8, 97)
    hidden = torch.randn(2, 41, 8, requires_grad=True)
    labels = torch.randint(0, 97, (2, 41))
    labels[:, :20] = -100
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = chunked_causal_loss(hidden, head, labels, chunk_size=4)
    assert not any(shape and shape[-1] == 97 for shape in saved)
    assert not any(len(shape) == 3 and shape[:2] == (2, 41) for shape in saved)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


def test_all_masked_and_no_grad_evaluation():
    head = torch.nn.Linear(3, 11)
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    labels = torch.full((2, 4), -100)
    loss = chunked_causal_loss(hidden, head, labels)
    loss.backward()
    assert loss == 0
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
    assert torch.equal(head.weight.grad, torch.zeros_like(head.weight))
    with torch.no_grad():
        labels[:, -1] = 1
        value = chunked_causal_loss(hidden, head, labels, chunk_size=1)
        expected = make_smoothed_causal_lm_loss(0)(
            logits=head(hidden), labels=labels, vocab_size=11
        )
    torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bf16_autocast_checkpoint_keeps_loss_and_gradients(dtype):
    torch.manual_seed(19)
    head = torch.nn.Linear(16, 97).to(dtype)
    reference = copy.deepcopy(head)
    hidden = torch.randn(2, 9, 16, dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    labels = torch.randint(0, 97, (2, 9))
    labels[:, :4] = -100
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = chunked_causal_loss(hidden, head, labels, chunk_size=3, epsilon=0.1)
        expected = make_smoothed_causal_lm_loss(0.1)(
            logits=reference(other), labels=labels, vocab_size=97
        )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hidden.grad, other.grad, atol=1e-3, rtol=0.02)
    torch.testing.assert_close(head.weight.grad, reference.weight.grad, atol=1e-3, rtol=0.02)


@pytest.mark.parametrize("stage", ["adapter", "decoder", "full"])
@pytest.mark.parametrize("tied", [True, False])
def test_gemma_loss_path_with_checkpointing_matches_native(tmp_path, tokenizer, stage, tied):
    gqasr_source(tmp_path / "decoder", tokenizer, tied=tied)
    encoder_source(tmp_path / "encoder", "audflash_gemma")
    export_model(
        "audflash",
        tmp_path / "aud",
        gqasr_checkpoint=tmp_path / "decoder",
        encoder_source=tmp_path / "encoder",
        dtype="float32",
        flash_audio_stride=1,
    )
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "aud")
    processor = AUDProcessor.from_pretrained(tmp_path / "aud")
    model.configure_training(stage)
    # Match cluster's decoder/encoder checkpoint implementation (ZeRO-2).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()
    reference = copy.deepcopy(model)
    reference.aud_decoder.loss_function = make_smoothed_causal_lm_loss(0.1)
    batch = processor.prepare_training_batch(
        [
            np.random.default_rng(i).normal(0, 0.03, n).astype("float32")
            for i, n in enumerate([3200, 5600])
        ],
        text=["hello", "a longer transcript"],
        language=["en", "ml"],
    )
    torch.manual_seed(91)
    expected = reference(**batch).loss
    expected.backward()
    torch.manual_seed(91)
    result = model(**batch, loss_only=True, loss_chunk_size=3, loss_smoothing=0.1)
    assert result.logits is None
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    for (name, actual), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (actual.grad is None) == (wanted.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-6, rtol=3e-4, msg=name)
    # Public inference remains native and returns full logits.
    model.eval()
    with torch.no_grad():
        assert (
            model(**{k: v for k, v in batch.items() if k != "labels"}).logits.shape[:2]
            == batch["input_ids"].shape
        )
