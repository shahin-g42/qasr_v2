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
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch.utils.data import Sampler

# Records in manifests without duration metadata contribute no hours; assume
# ~10 s per record when estimating their corpus size so they still get a
# principled weight instead of being dropped.
_ASSUMED_SECONDS_PER_UNKNOWN_RECORD = 10.0

# The stratified affine map computes offsets * a in int64; with a < n the
# product stays below n**2, so n must stay below sqrt(2**63 - 1) or the map
# silently wraps and stops being a bijection (wrong records, broken coverage).
_MAX_AFFINE_LANGUAGE_SIZE = math.isqrt(2**63 - 1)


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
        # Records kept WITHOUT a duration contribute no hours, so a corpus
        # that is 92% undated would report ~8% of its true size and be
        # sampled far below its share. Credit each undated record with the
        # assumed duration instead of only falling back when hours is
        # exactly zero.
        missing = int(getattr(stats, "missing_duration", 0) or 0)
        hours += missing * _ASSUMED_SECONDS_PER_UNKNOWN_RECORD / 3600.0
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


def _coprime_multiplier(n: int, rng: np.random.Generator) -> int:
    """Return ``a`` in [1, n) with gcd(a, n) == 1, drawn from ``rng``."""
    if n <= 2:
        return 1
    for _ in range(64):
        a = int(rng.integers(1, n))
        if math.gcd(a, n) == 1:
            return a
    return 1


class StratifiedLanguageSampler(Sampler[int]):
    """Equal languages in every mini-batch, with full coverage of every corpus.

    ``LanguageBalancedSampler`` draws *with replacement*, so one epoch of steps
    leaves a large fraction of the dominant language unseen (at temperature 0.5
    roughly half of Arabic). This sampler answers the other requirement: every
    record of every language is visited at least once per epoch, and each
    per-device mini-batch holds exactly ``batch_size // num_languages`` samples
    from each language.

    Mechanics. Let ``n_l`` be the record count of language ``l`` and
    ``N = max_l n_l``. Every language contributes ``N`` draws per epoch, so the
    largest language is traversed once and the smaller ones cycle -- language
    ``l`` repeats roughly ``N / n_l`` times. That repetition is the explicit
    trade for equal per-batch representation; call ``repeat_factors`` to see it
    before launching. Per-rank draw counts are rounded UP to whole batch slots,
    so across ranks a language receives at most ``num_replicas * per_language``
    draws beyond ``N`` -- coverage is always >= 1 visit per record, with a
    rounding tail of a few duplicate visits (a few hundred records at 100M
    scale), never a shortfall.

    Coverage across ranks is exact rather than probabilistic. Positions
    ``[rank * M, (rank + 1) * M)`` of each language's virtual stream are mapped
    through a per-(epoch, cycle) affine bijection ``q -> (a * q + b) mod n_l``
    with ``gcd(a, n_l) == 1``. Because the map is a bijection, disjoint position
    ranges yield disjoint records, so the union over ranks covers ``[0, n_l)``.
    The affine map is a weak permutation on its own -- it is used only to
    partition records across ranks, which is harmless under data parallelism
    since every rank's gradient lands in the same update. The *order* a rank
    trains on is a real RNG shuffle applied afterwards.

    Memory is O(M) per language, never O(n_l): a rank materializes only its own
    slice (a few int64 arrays of length ``M``), tens of MB per rank at
    100M-record scale rather than the ~1 GB a full materialized permutation
    would cost.

    Args:
        parts: Per-manifest datasets in combined order; each needs ``.language``.
        batch_size: Per-device batch size. Must be a positive multiple of the
            number of distinct languages.
        seed: Base RNG seed (use ``data_seed``).
        num_replicas: World size.
        rank: This rank's index.
    """

    def __init__(
        self,
        parts: Sequence[Any],
        *,
        batch_size: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")
        if not parts:
            raise ValueError("At least one dataset part is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        # Group each part's contiguous index slice under its language tag,
        # preserving the order the parts were concatenated in.
        ranges: dict[str, list[tuple[int, int]]] = {}
        start = 0
        for part in parts:
            size = len(part)
            if size < 1:
                raise ValueError("StratifiedLanguageSampler cannot sample an empty part")
            language = getattr(part, "language", None) or "und"
            ranges.setdefault(language, []).append((start, size))
            start += size

        self.languages = tuple(sorted(ranges))
        num_languages = len(self.languages)
        if batch_size % num_languages:
            raise ValueError(
                f"per_device_train_batch_size ({batch_size}) must be a multiple of the "
                f"number of languages ({num_languages}: {', '.join(self.languages)}) so "
                "every mini-batch can hold an equal share of each"
            )

        self._starts: dict[str, np.ndarray] = {}
        self._bounds: dict[str, np.ndarray] = {}
        self.language_sizes: dict[str, int] = {}
        for language in self.languages:
            spans = ranges[language]
            sizes = np.asarray([size for _, size in spans], dtype=np.int64)
            self._starts[language] = np.asarray([s for s, _ in spans], dtype=np.int64)
            # bounds[j] is the first virtual position belonging to span j.
            self._bounds[language] = np.concatenate(
                ([0], np.cumsum(sizes)[:-1])
            ).astype(np.int64)
            language_size = int(sizes.sum())
            if language_size > _MAX_AFFINE_LANGUAGE_SIZE:
                raise ValueError(
                    f"language {language!r} has {language_size:,} records, above the "
                    f"{_MAX_AFFINE_LANGUAGE_SIZE:,} limit where the int64 affine map "
                    "overflows and coverage silently breaks"
                )
            self.language_sizes[language] = language_size

        self.batch_size = int(batch_size)
        self.per_language = self.batch_size // num_languages
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        # Every language yields `epoch_draws` samples per epoch, set by the
        # largest so that it is fully covered (see the rounding-tail note in
        # the class docstring).
        self.epoch_draws = max(self.language_sizes.values())
        per_rank = math.ceil(self.epoch_draws / self.num_replicas)
        # Round up to a whole number of per-language batch slots so no rank
        # emits a ragged tail and coverage is not truncated.
        self.draws_per_rank = math.ceil(per_rank / self.per_language) * self.per_language
        self.num_batches = self.draws_per_rank // self.per_language
        self.num_samples = self.num_batches * self.batch_size

    @property
    def repeat_factors(self) -> dict[str, float]:
        """How many times each language is traversed per epoch."""
        return {
            language: self.epoch_draws / size
            for language, size in self.language_sizes.items()
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _language_indices(self, language: str, rng: np.random.Generator) -> np.ndarray:
        """Global dataset indices this rank draws for one language, shuffled."""
        size = self.language_sizes[language]
        lo = self.rank * self.draws_per_rank
        positions = np.arange(lo, lo + self.draws_per_rank, dtype=np.int64)
        cycles = positions // size
        offsets = positions % size

        # A fresh affine bijection per cycle, so a record that repeats does not
        # land at the same point of every pass.
        mapped = np.empty_like(offsets)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            # zlib.crc32, not builtin hash(): string hashing is randomized per
            # process by PYTHONHASHSEED, which would give each rank a different
            # permutation and silently destroy the disjoint-coverage guarantee.
            cycle_rng = np.random.default_rng(
                [
                    self.seed & 0x7FFFFFFF,
                    self.epoch,
                    zlib.crc32(language.encode("utf-8")),
                    int(cycle),
                ]
            )
            a = _coprime_multiplier(size, cycle_rng)
            b = int(cycle_rng.integers(0, size)) if size > 1 else 0
            mapped[mask] = (offsets[mask] * a + b) % size

        # Virtual position -> global index across this language's spans.
        bounds = self._bounds[language]
        span = np.searchsorted(bounds, mapped, side="right") - 1
        indices = self._starts[language][span] + (mapped - bounds[span])
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        rng = np.random.default_rng([self.seed & 0x7FFFFFFF, self.epoch, self.rank])
        per_language = [self._language_indices(lang, rng) for lang in self.languages]
        # Interleave so each consecutive `batch_size` slice holds an equal share
        # of every language. The DataLoader chunks the stream in order, so this
        # layout *is* the batch composition.
        stacked = np.stack(
            [block[: self.num_batches * self.per_language] for block in per_language]
        ).reshape(len(self.languages), self.num_batches, self.per_language)
        interleaved = stacked.transpose(1, 0, 2).reshape(-1)
        # Lazy conversion: a materialized Python list of ~9M ints costs ~250 MB
        # per rank at Phase-2 scale; map() keeps only the numpy array (~70 MB).
        return map(int, interleaved)


__all__ = [
    "LanguageBalancedSampler",
    "StratifiedLanguageSampler",
    "language_mix_weights",
]
