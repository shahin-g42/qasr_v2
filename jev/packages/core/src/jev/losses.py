"""Valid-candidate decision objectives; no verbal-confidence distillation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from jev_contracts import PROBABILITY_SUM_TOLERANCE

from .codes import MAX_OPTIONS
from .processing import validate_counts


def _logits(logits: torch.Tensor, counts: Sequence[int]) -> tuple[int, ...]:
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 2 or logits.shape[1] != MAX_OPTIONS or logits.shape[0] == 0):
        raise ValueError("logits must be a nonempty floating-point [B,255] tensor")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("All logits, including masked positions, must be finite")
    if isinstance(counts, torch.Tensor):
        if counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("Tensor counts must be one-dimensional integer counts, not bool")
        counts = counts.tolist()
    return validate_counts(counts, logits.shape[0])


def _targets(targets: torch.Tensor, logits: torch.Tensor, counts: tuple[int, ...]) -> None:
    if (not isinstance(targets, torch.Tensor) or not targets.is_floating_point()
            or targets.shape != logits.shape or targets.device != logits.device):
        raise ValueError("targets must be floating-point [B,255] on the logits device, not bool")
    if not bool(torch.isfinite(targets).all()) or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("targets must be finite probabilities in [0,1]")
    for row, count in zip(targets, counts, strict=True):
        if bool((row[count:] != 0).any()):
            raise ValueError("Padded target probabilities must be exactly zero")
        if abs(float(row[:count].double().sum().item()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError("Valid target probabilities must sum to one; no renormalization")


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite loss; check logit magnitude and temperature")
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def decision_loss(
    logits: torch.Tensor, targets: torch.Tensor, counts: Sequence[int], reduction: str = "none",
) -> torch.Tensor:
    """Per-example CE against validated hard one-hot or complete soft targets."""
    counts = _logits(logits, counts)
    _targets(targets, logits, counts)
    values = torch.stack([
        -(target[:count].float() * row[:count].float().log_softmax(-1)).sum()
        for row, target, count in zip(logits, targets, counts, strict=True)
    ])
    return _reduce(values, reduction)


def scaled_loss(
    losses: torch.Tensor, *, global_examples: int, world_size: int,
) -> torch.Tensor:
    """D/N times the local loss SUM for a reducer averaging across D ranks.

    Caller zero-weights dummy examples before this function. No additional
    accumulation-step or token-count division belongs in this normalization.
    """
    if (type(global_examples) is not int or global_examples < 1
            or type(world_size) is not int or world_size < 1):
        raise ValueError("global_examples and world_size must be positive integers")
    if (not isinstance(losses, torch.Tensor) or not losses.is_floating_point()
            or losses.ndim != 1 or not bool(torch.isfinite(losses).all())):
        raise ValueError("losses must be a finite one-dimensional floating-point tensor")
    return losses.sum() * (world_size / global_examples)


def _candidate_ids(ids: Sequence[Sequence[str]], counts: tuple[int, ...]) -> list[list[str]]:
    if not isinstance(ids, (list, tuple)) or len(ids) != len(counts):
        raise ValueError("Candidate IDs must contain one ordered ID array per example")
    result = []
    for row, count in zip(ids, counts, strict=True):
        if (not isinstance(row, (list, tuple)) or len(row) != count
                or any(type(key) is not str or not key for key in row)
                or len(set(row)) != count):
            raise ValueError("Candidate IDs must be unique nonempty strings in candidate order")
        result.append(list(row))
    return result


def distillation_loss(
    logits: torch.Tensor, targets: torch.Tensor, teacher_probabilities: torch.Tensor,
    counts: Sequence[int], *, temperature: float, candidate_ids: Sequence[Sequence[str]],
    teacher_candidate_ids: Sequence[Sequence[str]], alpha: float = 0.5, reduction: str = "none",
) -> torch.Tensor:
    """(1-alpha)*CE + alpha*T²*KL(teacher(T) || student(T)).

    Teacher probabilities MUST already be evaluated at the explicit temperature
    over the complete identical ordered candidate set. They are detached, never
    treated as verbal confidence or silently rescaled from a different temperature.
    """
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("Distillation temperature must be explicit, positive, finite, not bool")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite in [0,1], not bool")
    counts = _logits(logits, counts)
    _targets(teacher_probabilities, logits, counts)
    student_ids = _candidate_ids(candidate_ids, counts)
    if student_ids != _candidate_ids(teacher_candidate_ids, counts):
        raise ValueError("Teacher and student must use exactly the same ordered candidates")
    gold = decision_loss(logits, targets, counts)
    terms = []
    for row, teacher, count in zip(logits, teacher_probabilities.detach(), counts, strict=True):
        q = teacher[:count].float()
        log_p = (row[:count].float() / temperature).log_softmax(-1)
        kl = (torch.special.xlogy(q, q) - q * log_p).sum()
        terms.append(kl * temperature ** 2)
    values = (1 - alpha) * gold + alpha * torch.stack(terms)
    return _reduce(values, reduction)


def permutation_consistency(
    logits: torch.Tensor, permuted_logits: torch.Tensor, counts: Sequence[int],
    aligned_indices: Sequence[Sequence[int]] | torch.Tensor, *, reduction: str = "none",
) -> torch.Tensor:
    """JS(original, inverse-aligned permutation), with gradients through both views.

    aligned_indices[b][i] is the position in the permuted row of ORIGINAL option i.
    Supply K indices per row, or a padded integer [B,255] tensor (tail ignored).
    Every valid row must be a bijection of 0..K-1. Candidate sets must not change.
    """
    counts = _logits(logits, counts)
    _logits(permuted_logits, counts)
    if logits.shape != permuted_logits.shape or logits.device != permuted_logits.device:
        raise ValueError("Permutation views must have matching shapes and devices")
    if isinstance(aligned_indices, torch.Tensor):
        if (aligned_indices.shape != logits.shape
                or aligned_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Padded alignment must be an integer [B,255] tensor, not bool")
        aligned_indices = [row[:count] for row, count in zip(
            aligned_indices.tolist(), counts, strict=True,
        )]
    if not isinstance(aligned_indices, (list, tuple)) or len(aligned_indices) != len(counts):
        raise ValueError("An inverse permutation is required for every example")
    terms = []
    for row, other, count, indices in zip(
        logits, permuted_logits, counts, aligned_indices, strict=True,
    ):
        if (not isinstance(indices, (list, tuple)) or len(indices) != count
                or any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(count))):
            raise ValueError("Alignment must be a complete bijection of valid candidate indices")
        index = torch.tensor(indices, device=other.device)
        log_p = row[:count].float().log_softmax(-1)
        log_q = other[:count].float().log_softmax(-1)[index]
        log_m = torch.logaddexp(log_p, log_q) - math.log(2)
        terms.append((0.5 * ((log_p.exp() * (log_p - log_m)).sum()
                           + (log_q.exp() * (log_q - log_m)).sum())).clamp_min(0))
    return _reduce(torch.stack(terms), reduction)
