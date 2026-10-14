import math

import torch
from torch import nn


def length_mask(lengths, width=None):
    width = int(lengths.max()) if width is None else width
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def sinusoidal(values, width):
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=values.device).float() * (-math.log(10000.0) / width)
    )
    angles = values.float()[..., None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ContextBlock(nn.Module):
    """Small bidirectional attention block plus masked depthwise local context."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2, groups=width),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = self.attention(x, src_key_padding_mask=~mask)
        x = x.masked_fill(~mask[..., None], 0)
        local = self.norm(x).masked_fill(~mask[..., None], 0)
        x = x + self.conv(local.transpose(1, 2)).transpose(1, 2)
        return x.masked_fill(~mask[..., None], 0)


def masked_mean(x, mask):
    return (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def average_upsample(x, source_lengths, target_lengths, width):
    """Uniform monotonic initial alignment, separately for each unpadded example."""
    positions = torch.arange(width, device=x.device)[None]
    indices = positions * source_lengths[:, None] // target_lengths[:, None].clamp_min(1)
    indices = torch.minimum(indices, source_lengths[:, None] - 1).clamp_min(0)
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))
