"""Explicit runtime tradeoffs; defaults retain existing training behavior."""

import copy
import os
from pathlib import Path

DEFAULTS = dict(
    loss_backend="chunked",
    loss_chunk_size=128,
    duration_bucketing=False,
    index_cache=None,
    dataloader_timeout=300,
    prefetch_factor=2,
    encoder_checkpoint_every=1,
    decoder_checkpoint_every=1,
    category_metrics=True,
    distributed_wer=False,
)


def resolve_efficiency(value, base):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError(f"efficiency accepts only {sorted(DEFAULTS)}")
    result = dict(DEFAULTS, **copy.deepcopy(value))
    for key in ("loss_chunk_size", "prefetch_factor", "dataloader_timeout"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"efficiency.{key} must be a positive integer")
    for key in ("encoder_checkpoint_every", "decoder_checkpoint_every"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"efficiency.{key} must be an integer >= 0 (0 disables)")
    for key in ("duration_bucketing", "category_metrics", "distributed_wer"):
        if type(result[key]) is not bool:
            raise ValueError(f"efficiency.{key} must be boolean")
    if result["loss_backend"] not in ("chunked", "liger"):
        raise ValueError("loss_backend must be chunked or liger")
    if result["index_cache"] is not None:
        path = os.path.expandvars(str(result["index_cache"]))
        if "$" in path:
            raise ValueError("Unset variable in index_cache")
        path = Path(path).expanduser()
        result["index_cache"] = str((Path(base) / path).resolve())
    return result


def apply_checkpoint_policy(model, policy):
    """Called after Trainer enables checkpointing, before the first forward."""
    for name in ("encoder", "decoder"):
        module = model.aud_encoder if name == "encoder" else model.aud_decoder.model
        every = policy[f"{name}_checkpoint_every"]
        for index, layer in enumerate(getattr(module, "layers", [])):
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = bool(every and index % every == 0)
    model.aud_checkpoint_policy = policy
