"""Language-balanced distributed sampling for multilingual QASR training.

``CombinedSpeechDataset`` naive concatenation lets dominant corpora (ar/en)
drown small ones (ml/hi): a uniform shuffle sees each record with probability
proportional to its corpus size. ``LanguageBalancedSampler`` instead draws a
language first (probability proportional to ``hours ** temperature`` — the
Canary alpha/beta recipe; ``temperature=0.5`` gives square-root balancing),
then draws uniformly *with replacement* inside that language's index slice.

Because draws are with replacement, each distributed rank can generate its
own i.i.d. stream seeded by ``(seed, epoch, rank)``: no cross-rank index
traffic, no giant materialized index lists at 100M+ sample scale, and exact
determinism for a given (seed, epoch, rank). ``set_epoch`` advances the mix
deterministically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch.utils.data import Sampler

# Records in manifests without duration metadata contribute no hours; assume
# ~10 s per record when estimating their corpus size so they still get a
# principled weight instead of being dropped.
_ASSUMED_SECONDS_PER_UNKNOWN_RECORD = 10.0


def language_mix_weights(parts: Sequence[Any], temperature: float) -> np.ndarray:
    """Return per-part sampling probabilities (hours^temperature, normalized).

    Uses ``stats.total_kept_hours`` when known; falls back to
    ``kept_records * assumed duration`` for manifests lacking durations.
    """
    if not 0 < temperature <= 1:
        raise ValueError(f"temperature must be in (0, 1], got {temperature}")
    if not parts:
        raise ValueError("At least one dataset part is required")

    bases: list[float] = []
    for part in parts:
        stats = part.stats
        hours = float(getattr(stats, "total_kept_hours", 0.0) or 0.0)
        if hours <= 0:
            kept = int(getattr(stats, "kept_records", 0) or 0)
            hours = kept * _ASSUMED_SECONDS_PER_UNKNOWN_RECORD / 3600.0
        if hours <= 0:
            raise ValueError("Dataset part has neither kept hours nor kept records")
        bases.append(hours)

    weights = np.asarray(bases, dtype=np.float64) ** float(temperature)
    total = weights.sum()
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Language mix weights must be finite and positive")
    return weights / total


class LanguageBalancedSampler(Sampler[int]):
    """Map-style sampler over a ``CombinedSpeechDataset`` index space.

    Args:
        parts: The per-manifest ``JsonlSpeechDataset`` objects, in the same
            order they were combined (defines the index slices).
        temperature: Balancing exponent applied to per-language hours.
        seed: Base RNG seed (use ``data_seed``).
        epoch_size: Number of global samples one epoch should yield.
        num_replicas: World size; each rank emits ``ceil(epoch_size /
            num_replicas)`` indices.
        rank: This rank's index.
    """

    def __init__(
        self,
        parts: Sequence[Any],
        *,
        temperature: float,
        seed: int,
        epoch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
        if epoch_size < 1:
            raise ValueError("epoch_size must be positive")

        self.parts = tuple(parts)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.epoch_size = int(epoch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        self.probs = language_mix_weights(self.parts, self.temperature)
        self.slice_starts: list[int] = []
        self.slice_sizes: list[int] = []
        start = 0
        for part in self.parts:
            size = len(part)
            if size < 1:
                raise ValueError("LanguageBalancedSampler cannot sample an empty part")
            self.slice_starts.append(start)
            self.slice_sizes.append(size)
            start += size
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        # Seed mixes base seed, epoch, and rank; streams are i.i.d. across
        # ranks, which is valid because sampling is with replacement.
        rng = np.random.default_rng([self.seed & 0x7FFFFFFF, self.epoch, self.rank])
        counts = rng.multinomial(self.num_samples, self.probs)
        chunks: list[np.ndarray] = []
        for language_index, count in enumerate(counts):
            if count == 0:
                continue
            start = self.slice_starts[language_index]
            size = self.slice_sizes[language_index]
            chunks.append(start + rng.integers(0, size, size=int(count)))
        indices = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        rng.shuffle(indices)
        return iter(indices.tolist())


__all__ = ["LanguageBalancedSampler", "language_mix_weights"]
