"""Label-smoothed causal LM loss for QASR training.

Transformers 5.14 computes the causal LM loss through ``model.loss_function``
(defaulting to ``ForCausalLMLoss``). Its ``label_smoothing_factor`` Trainer
option does NOT reach this loss path, so label smoothing must be provided by
swapping the loss function itself. ``make_smoothed_causal_lm_loss`` returns a
drop-in replacement that mirrors ``ForCausalLMLoss``/``fixed_cross_entropy``
bit-for-bit (same shift/pad logic, same reduction semantics) and only adds
``label_smoothing`` to the cross-entropy call. Keeping the reduction
semantics identical is what keeps gradient-accumulation normalization
(``num_items_in_batch``) correct under DeepSpeed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def make_smoothed_causal_lm_loss(epsilon: float) -> Callable[..., torch.Tensor]:
    """Build a label-smoothed ``ForCausalLMLoss`` replacement.

    Args:
        epsilon: Label-smoothing factor in [0, 0.2). 0 reproduces the stock
            loss exactly.

    Returns:
        A function with the exact ``ForCausalLMLoss`` signature, suitable for
        ``model.loss_function = ...``.
    """
    if not 0 <= epsilon < 0.2:
        raise ValueError(f"label smoothing epsilon must be in [0, 0.2), got {epsilon}")

    def smoothed_causal_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | None = None,
        ignore_index: int = -100,
        shift_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        # Upcast to float to avoid precision issues, exactly like the stock loss.
        logits = logits.float()

        if shift_labels is None:
            # Shift so that tokens < n predict n.
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens.
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)

        if not (shift_labels != ignore_index).any():
            # Fully masked batch (e.g. a micro-batch of silence samples):
            # cross_entropy would return NaN from a 0/0 mean. There is no
            # supervision signal here, so the correct value is zero, kept
            # on-graph so gradient bookkeeping stays consistent.
            return logits.sum() * 0.0

        # Mirror fixed_cross_entropy: sum reduction divided by the global
        # token count when the Trainer provides it, else mean.
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(
            logits,
            shift_labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=epsilon,
        )
        if reduction == "sum":
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss

    return smoothed_causal_lm_loss


__all__ = ["make_smoothed_causal_lm_loss"]
