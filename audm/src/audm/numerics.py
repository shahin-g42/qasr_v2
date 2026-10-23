"""Persist tensor-only failing batches for numerical debugging; never model weights."""

import json
from pathlib import Path

import torch


def save_failure(directory, inputs, *, rank, step, error):
    directory = Path(directory) / "numerics"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"rank-{rank}-step-{step}"
    tensors = {
        name: value.detach().cpu().clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    torch.save(tensors, prefix.with_suffix(".pt"))
    report = {
        "rank": rank,
        "global_step": step,
        "error": str(error),
        "batch": str(prefix.with_suffix(".pt")),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all()),
            }
            for name, value in tensors.items()
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
