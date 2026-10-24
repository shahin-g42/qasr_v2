"""Exact causal cross-entropy without retaining batch × sequence × vocab logits."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def validate_liger():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError as exc:
        raise RuntimeError("loss_backend=liger requires pip install 'audm[fused]' on CUDA") from exc
    import inspect

    required = {"label_smoothing", "accum_dtype"}
    if not required <= inspect.signature(LigerFusedLinearCrossEntropyLoss).parameters.keys():
        raise RuntimeError("Installed Liger lacks required label-smoothing/FP32 accumulation API")
    return LigerFusedLinearCrossEntropyLoss


def chunked_causal_loss(
    hidden,
    head,
    labels,
    *,
    chunk_size=128,
    epsilon=0.0,
    num_items_in_batch=None,
    backend="chunked",
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
    if backend not in ("chunked", "liger"):
        raise ValueError("Unknown loss backend")
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    selected = hidden[:, :-1][valid]
    targets = targets[valid].to(hidden.device)
    if not len(targets):
        # Keep backbone/head on graph without allocating any vocabulary logits.
        zero = hidden.sum() * 0 + head.weight.reshape(-1)[:1].sum() * 0
        return zero if head.bias is None else zero + head.bias[:1].sum() * 0

    def loss_chunk(states, target):
        return F.cross_entropy(
            head(states).float(), target, reduction="sum", label_smoothing=epsilon
        )

    total = hidden.new_zeros((), dtype=torch.float32)
    if backend == "liger" and torch.is_grad_enabled():
        if hidden.device.type != "cuda":
            raise ValueError("Liger training requires CUDA; use chunked for CPU verification")
        criterion = validate_liger()(
            reduction="sum",
            label_smoothing=epsilon,
            accum_dtype=torch.float32,
        )
        total = criterion(head.weight, selected.contiguous(), targets.contiguous(), head.bias)
    else:
        for start in range(0, len(targets), chunk_size):
            states, target = (
                selected[start : start + chunk_size],
                targets[start : start + chunk_size],
            )
            if torch.is_grad_enabled():
                value = checkpoint(loss_chunk, states, target, use_reentrant=False)
            else:
                value = loss_chunk(states, target)
            total = total + value
    denominator = len(targets) if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(total.device)
    return total / denominator


@torch.no_grad()
def audio_category_metrics(hidden, head, labels, token_ids):
    """Small detached audit: [count, NLL sum, correct, target-token count] per class.

    Computes only category-position logits, in small chunks. No weighting changes.
    NLL is unsmoothed so it remains comparable between adaptation and full training.
    """
    targets = labels[:, 1:]
    if targets.numel() == 0:
        return hidden.new_zeros((4, 4), dtype=torch.float32)
    valid = targets.ne(-100)
    first = valid.long().argmax(-1)
    rows = torch.arange(len(labels), device=labels.device)
    classes = targets[rows, first]
    states = hidden[:, :-1][rows, first]
    result = hidden.new_zeros((4, 4), dtype=torch.float32)
    for i, kind in enumerate(("speech", "music", "noise", "silence")):
        mask = classes.eq(token_ids[kind]) & valid.any(-1)
        subset = states[mask]
        count = mask.sum()
        result[i, 0] = count
        result[i, 3] = valid[mask].sum()
        for start in range(0, len(subset), 128):
            logits = head(subset[start : start + 128]).float()
            expected = torch.full(
                (len(logits),), token_ids[kind], device=logits.device, dtype=torch.long
            )
            result[i, 1] += F.cross_entropy(logits, expected, reduction="sum")
            result[i, 2] += logits.argmax(-1).eq(expected).sum()
    return result
