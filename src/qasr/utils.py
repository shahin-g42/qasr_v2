from __future__ import annotations

import math

import torch


def get_subsampling_output_lengths(
    input_lengths: torch.Tensor,
    *,
    subsampling_factor: int = 8,
    kernel_size: int = 3,
    stride: int = 2,
) -> torch.Tensor:
    """Return valid Cohere encoder frame counts after its strided 2-D convolutions."""
    if subsampling_factor < 1 or subsampling_factor & (subsampling_factor - 1):
        raise ValueError("subsampling_factor must be a positive power of two")
    if kernel_size < 1 or stride < 1:
        raise ValueError("kernel_size and stride must be positive")

    lengths = input_lengths.to(dtype=torch.long)
    padding = (kernel_size - 1) // 2
    for _ in range(int(math.log2(subsampling_factor))):
        lengths = torch.div(
            lengths + 2 * padding - kernel_size,
            stride,
            rounding_mode="floor",
        ) + 1
    return lengths


def get_subsampled_attention_mask(
    attention_mask: torch.Tensor,
    *,
    target_length: int | None = None,
    subsampling_factor: int = 8,
    kernel_size: int = 3,
    stride: int = 2,
) -> torch.Tensor:
    """Convert a mel-frame mask into a boolean encoder-output mask."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (batch, frames)")
    lengths = get_subsampling_output_lengths(
        attention_mask.sum(-1),
        subsampling_factor=subsampling_factor,
        kernel_size=kernel_size,
        stride=stride,
    )
    width = int(lengths.max().item()) if target_length is None else int(target_length)
    return torch.arange(width, device=attention_mask.device).unsqueeze(0) < lengths.unsqueeze(1)


__all__ = ["get_subsampled_attention_mask", "get_subsampling_output_lengths"]
