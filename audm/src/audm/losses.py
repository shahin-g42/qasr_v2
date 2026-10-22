"""Exact causal cross-entropy without retaining batch × sequence × vocab logits."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_causal_loss(
    hidden, head, labels, *, chunk_size=128, epsilon=0.0, num_items_in_batch=None
):
    """Project supervised predecessor states only, recomputing each head chunk.

    Decoder attention still sees the complete audio/text context. This changes
    neither next-token targets nor uniform label smoothing. Checkpointing the
    head AND CE is essential: simply summing chunk losses retains every softmax
    activation until backward and does not bound memory.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("loss chunk_size must be a positive integer")
    if not 0 <= epsilon < 0.2:
        raise ValueError("label smoothing must be in [0, 0.2)")
    if hidden.shape[:2] != labels.shape:
        raise ValueError("Hidden states and causal labels must have matching batch/sequence shapes")
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    selected = hidden[:, :-1][valid]
    targets = targets[valid].to(hidden.device)
    if not len(targets):
        # Keep both backbone and head on-graph for a completely masked batch.
        return head(hidden[:, :1]).float().sum() * 0

    def loss_chunk(states, target):
        return F.cross_entropy(
            head(states).float(), target, reduction="sum", label_smoothing=epsilon
        )

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        states, target = selected[start : start + chunk_size], targets[start : start + chunk_size]
        if torch.is_grad_enabled():
            value = checkpoint(loss_chunk, states, target, use_reentrant=False)
        else:
            value = loss_chunk(states, target)
        total = total + value
    denominator = len(targets) if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(total.device)
    return total / denominator
