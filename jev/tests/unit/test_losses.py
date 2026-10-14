from __future__ import annotations

import math

import pytest
import torch

from jev.losses import decision_loss, distillation_loss, permutation_consistency, scaled_loss


def tensors():
    logits = torch.zeros((3, 255), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1
    targets[1, :2] = torch.tensor([0.25, 0.75])
    targets[2, :3] = torch.tensor([0.2, 0.3, 0.5])
    return logits, targets, [1, 2, 3]


def test_ce_valid_slices_and_gradients():
    logits, targets, counts = tensors()
    loss = decision_loss(logits, targets, counts)
    torch.testing.assert_close(loss, torch.tensor([0, math.log(2), math.log(3)]))
    torch.testing.assert_close(decision_loss(logits, targets, counts, "mean"), loss.mean())
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    for i, count in enumerate(counts):
        assert torch.count_nonzero(logits.grad[i, count:]) == 0
        torch.testing.assert_close(logits.grad[i, :count],
                                   torch.full((count,), 1 / count) - targets[i, :count])


@pytest.mark.parametrize("damage", ["nan", "inf", "negative", "mass", "padding", "bool", "int"])
def test_invalid_targets(damage):
    logits, targets, counts = tensors()
    if damage == "nan":
        targets[0, 0] = float("nan")
    elif damage == "inf":
        targets[0, 0] = float("inf")
    elif damage == "negative":
        targets[1, :2] = torch.tensor([-0.1, 1.1])
    elif damage == "mass":
        targets[0, 0] = 0.9
    elif damage == "padding":
        targets[0, 2] = 0.1
    elif damage == "bool":
        targets = targets.bool()
    else:
        targets = targets.long()
    with pytest.raises(ValueError):
        decision_loss(logits, targets, counts)


@pytest.mark.parametrize("counts", [[True, 2, 3], [0, 2, 3], [1, 2, 256], [1, 2], [1.0, 2, 3]])
def test_invalid_counts(counts):
    logits, targets, _ = tensors()
    with pytest.raises(ValueError):
        decision_loss(logits, targets, counts)


def test_nonfinite_padded_logits_rejected():
    logits, targets, counts = tensors()
    with torch.no_grad():
        logits[0, 200] = -torch.inf
    with pytest.raises(ValueError, match="finite"):
        decision_loss(logits, targets, counts)


def test_unequal_microbatches_match_serial_average_gradient():
    base = torch.tensor(1.5, requires_grad=True)
    serial = ((base - torch.arange(5.0)) ** 2).mean()
    serial.backward()
    gradients = []
    for groups in ([torch.arange(2.0), torch.arange(2.0, 3.0)], [torch.arange(3.0, 5.0)]):
        parameter = torch.tensor(1.5, requires_grad=True)
        for group in groups:
            scaled_loss((parameter - group) ** 2, global_examples=5, world_size=2).backward()
        gradients.append(parameter.grad)
    torch.testing.assert_close(torch.stack(gradients).mean(), base.grad)
    with pytest.raises(ValueError):
        scaled_loss(torch.ones(1), global_examples=True, world_size=2)


def test_distillation_uses_full_teacher_distribution_and_temperature():
    logits, targets, counts = tensors()
    teacher = targets.clone().requires_grad_()
    ids = [["a"], ["false", "true"], ["0", "1", "2"]]
    kd = distillation_loss(
        logits, targets, teacher, counts, temperature=2.0, alpha=0.4,
        candidate_ids=ids, teacher_candidate_ids=ids,
    )
    expected = []
    for row, count in zip(targets, counts, strict=True):
        q = row[:count]
        kl = (torch.special.xlogy(q, q) + q * math.log(count)).sum()
        expected.append(0.6 * math.log(count) + 0.4 * 4 * kl)
    torch.testing.assert_close(kd, torch.stack(expected))
    kd.sum().backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="ordered candidates"):
        distillation_loss(
            logits, targets, teacher, counts, temperature=2.0,
            candidate_ids=ids, teacher_candidate_ids=[["a"], ["true", "false"], ids[2]],
        )
    with pytest.raises(ValueError):
        distillation_loss(
            logits, targets, teacher, counts, temperature=True,
            candidate_ids=ids, teacher_candidate_ids=ids,
        )


def test_inverse_permutation_js_and_gradients():
    logits = torch.zeros((1, 255))
    logits[0, :3] = torch.tensor([1.0, -2.0, 3.0])
    logits.requires_grad_()
    other = torch.zeros_like(logits)
    other[0, :3] = logits.detach()[0, [2, 0, 1]]
    other.requires_grad_()
    loss = permutation_consistency(logits, other, [3], [[1, 2, 0]])
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    changed = permutation_consistency(logits, other, [3], [[0, 1, 2]])
    assert 0 < changed.item() <= math.log(2)
    changed.sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(other.grad).all()
    assert logits.grad[0, 3:].count_nonzero() == 0
    for indices in ([[0, 0, 1]], [[True, 1, 2]], [[0, 1]], [[0, 1, 3]]):
        with pytest.raises(ValueError):
            permutation_consistency(logits, other, [3], indices)


def test_tensor_counts_bf16_logits_and_extreme_probabilities():
    logits = torch.zeros((1, 255), dtype=torch.bfloat16, requires_grad=True)
    targets = torch.zeros((1, 255))
    targets[0, 0] = 1
    loss = decision_loss(logits, targets, torch.tensor([255]))
    assert loss.dtype == torch.float32
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    a = torch.zeros((1, 255))
    b = a.clone()
    a[0, :2] = torch.tensor([1000, -1000])
    b[0, :2] = torch.tensor([-1000, 1000])
    js = permutation_consistency(a, b, [2], [[0, 1]])
    assert js.item() == pytest.approx(math.log(2))
