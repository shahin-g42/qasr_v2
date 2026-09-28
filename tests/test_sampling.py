"""Tests for the language-balanced distributed sampler."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from qasr.sampling import (
    LanguageBalancedSampler,
    StratifiedLanguageSampler,
    language_mix_weights,
)


def make_part(
    size: int, hours: float, language: str | None = None, missing_duration: int = 0
):
    """Minimal stand-in for JsonlSpeechDataset (stats + language + __len__)."""

    class FakePart:
        def __init__(self) -> None:
            self.stats = SimpleNamespace(
                total_kept_hours=hours,
                kept_records=size,
                missing_duration=missing_duration,
            )
            self.language = language

        def __len__(self) -> int:
            return size

    return FakePart()


class LanguageMixWeightsTest(unittest.TestCase):
    def test_weights_proportional_to_hours_sqrt(self) -> None:
        parts = [make_part(1000, 100.0), make_part(1000, 400.0)]

        weights = language_mix_weights(parts, temperature=0.5)

        # sqrt(100) : sqrt(400) = 10 : 20
        np.testing.assert_allclose(weights, [1 / 3, 2 / 3], atol=1e-12)

    def test_temperature_one_is_size_proportional(self) -> None:
        parts = [make_part(1000, 100.0), make_part(1000, 300.0)]

        weights = language_mix_weights(parts, temperature=1.0)

        np.testing.assert_allclose(weights, [0.25, 0.75], atol=1e-12)

    def test_fallback_to_kept_records_when_hours_missing(self) -> None:
        parts = [
            make_part(100, 0.0),
            make_part(100, 1.0),  # 1 hour
        ]
        # Part 0 fallback: 100 records * 10 s = 1000 s = 0.2778 h.
        weights = language_mix_weights(parts, temperature=1.0)
        self.assertAlmostEqual(weights[0], (1000 / 3600) / (1000 / 3600 + 1.0), places=10)

    def test_partially_missing_durations_credited_not_ignored(self) -> None:
        """Regression: a corpus with 92% undated records used to report ~8%
        of its true hours (the record-count fallback fired only at exactly
        zero), silently under-sampling it. Undated records now each credit
        the assumed 10 s."""
        # 1.0 known hour + 360 undated records * 10 s = 2.0 effective hours,
        # against a fully-dated 2.0-hour part -> equal weights.
        parts = [
            make_part(1000, 1.0, missing_duration=360),
            make_part(1000, 2.0),
        ]
        weights = language_mix_weights(parts, temperature=1.0)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)

    def test_invalid_temperature_rejected(self) -> None:
        parts = [make_part(10, 1.0)]
        with self.assertRaises(ValueError):
            language_mix_weights(parts, temperature=0.0)
        with self.assertRaises(ValueError):
            language_mix_weights(parts, temperature=1.5)


class LanguageBalancedSamplerTest(unittest.TestCase):
    def _parts(self):
        # Dominant language (10k samples) vs starved small language (100).
        return [make_part(10_000, 1000.0), make_part(100, 100.0)]

    def test_epoch_size_and_slice_bounds(self) -> None:
        sampler = LanguageBalancedSampler(
            self._parts(), temperature=0.5, seed=42, epoch_size=1000
        )

        indices = list(sampler)

        self.assertEqual(len(indices), 1000)
        self.assertTrue(all(0 <= i < 10_100 for i in indices))

    def test_deterministic_across_set_epoch_and_rebuild(self) -> None:
        first = list(
            LanguageBalancedSampler(
                self._parts(), temperature=0.5, seed=7, epoch_size=500
            )
        )
        sampler = LanguageBalancedSampler(
            self._parts(), temperature=0.5, seed=7, epoch_size=500
        )
        sampler.set_epoch(0)
        second = list(sampler)
        sampler.set_epoch(1)
        third = list(sampler)
        sampler.set_epoch(0)
        fourth = list(sampler)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)  # epoch advances the mix
        self.assertEqual(second, fourth)  # same epoch => same stream

    def test_small_language_upsampled_to_target_ratio(self) -> None:
        # hours^0.5 mix: sqrt(1000) : sqrt(100) ≈ 0.76 : 0.24.
        sampler = LanguageBalancedSampler(
            self._parts(), temperature=0.5, seed=42, epoch_size=100_000
        )

        indices = list(sampler)
        small_count = sum(1 for index in indices if index >= 10_000)
        ratio = small_count / len(indices)

        self.assertAlmostEqual(ratio, 0.2402, delta=0.05)

    def test_ranks_get_equal_length_distinct_streams(self) -> None:
        kwargs = {"temperature": 0.5, "seed": 42, "epoch_size": 1001}
        rank0 = list(
            LanguageBalancedSampler(
                self._parts(), num_replicas=2, rank=0, **kwargs
            )
        )
        rank1 = list(
            LanguageBalancedSampler(
                self._parts(), num_replicas=2, rank=1, **kwargs
            )
        )

        self.assertEqual(len(rank0), math.ceil(1001 / 2))
        self.assertEqual(len(rank1), math.ceil(1001 / 2))
        self.assertNotEqual(rank0, rank1)

    def test_invalid_rank_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LanguageBalancedSampler(
                self._parts(),
                temperature=0.5,
                seed=1,
                epoch_size=10,
                num_replicas=2,
                rank=2,
            )


class StratifiedLanguageSamplerTest(unittest.TestCase):
    """Full-coverage guarantees behind the v7.6 `sampling_strategy: stratified`.

    The operator requirement this sampler answers: every record of every
    language is seen at least once per epoch (Arabic — the largest — exactly
    once), small languages repeat to fill their equal per-batch share, and
    every per-device mini-batch holds the same number of samples per language.
    """

    def _parts(self):
        # Imbalanced mix, with Arabic split over two manifests so the
        # multi-span index mapping is exercised (spans are non-contiguous
        # in the combined index space: ar=[0,600)+[1000,1400)).
        return [
            make_part(600, 60.0, "ar"),
            make_part(400, 40.0, "en"),
            make_part(400, 40.0, "ar"),
            make_part(130, 13.0, "hi"),
            make_part(37, 3.7, "ml"),
            make_part(11, 1.1, "zh"),
        ]

    def _language_of(self, index: int) -> str:
        for start, stop, language in [
            (0, 600, "ar"),
            (600, 1000, "en"),
            (1000, 1400, "ar"),
            (1400, 1530, "hi"),
            (1530, 1567, "ml"),
            (1567, 1578, "zh"),
        ]:
            if start <= index < stop:
                return language
        raise AssertionError(f"index {index} out of range")

    def _streams(self, num_replicas: int, batch_size: int = 10):
        streams = []
        for rank in range(num_replicas):
            sampler = StratifiedLanguageSampler(
                self._parts(),
                batch_size=batch_size,
                seed=42,
                num_replicas=num_replicas,
                rank=rank,
            )
            streams.append((sampler, list(sampler)))
        return streams

    def test_every_record_of_every_language_seen(self) -> None:
        """The core operator guarantee: nothing unseen, especially Arabic."""
        streams = self._streams(num_replicas=4)
        seen: dict[int, int] = {}
        for _, indices in streams:
            for index in indices:
                seen[index] = seen.get(index, 0) + 1

        missing = [i for i in range(1578) if i not in seen]
        self.assertEqual(missing, [], "records never sampled in one epoch")

        # Arabic (largest, 1000 records) is covered exactly once per epoch.
        ar_counts = [seen[i] for i in range(1578) if self._language_of(i) == "ar"]
        self.assertEqual(min(ar_counts), 1)
        self.assertEqual(max(ar_counts), 1)

        # Small languages repeat near-uniformly (max - min <= 1 visits).
        for language in ("en", "hi", "ml", "zh"):
            counts = [
                seen[i] for i in range(1578) if self._language_of(i) == language
            ]
            self.assertGreaterEqual(min(counts), 1)
            self.assertLessEqual(max(counts) - min(counts), 1, language)

    def test_every_batch_has_equal_language_share(self) -> None:
        for sampler, indices in self._streams(num_replicas=2):
            per = sampler.per_language
            for offset in range(0, len(indices), sampler.batch_size):
                batch = indices[offset : offset + sampler.batch_size]
                composition = {}
                for index in batch:
                    language = self._language_of(index)
                    composition[language] = composition.get(language, 0) + 1
                self.assertEqual(
                    composition,
                    dict.fromkeys(sampler.languages, per),
                )

    def test_ranks_partition_largest_language_exactly(self) -> None:
        """Coverage is a bijective partition, not an i.i.d. approximation."""
        streams = self._streams(num_replicas=4)
        ar_sets = [
            {i for i in indices if self._language_of(i) == "ar"}
            for _, indices in streams
        ]
        for a in range(4):
            for b in range(a + 1, 4):
                self.assertEqual(ar_sets[a] & ar_sets[b], set())
        union = set().union(*ar_sets)
        self.assertEqual(len(union), 1000)

    def test_multi_span_language_indices_stay_in_bounds(self) -> None:
        (_, indices), = self._streams(num_replicas=1)
        ar_indices = {i for i in indices if self._language_of(i) == "ar"}
        self.assertTrue(
            all(i < 600 or 1000 <= i < 1400 for i in ar_indices)
        )
        self.assertEqual(len(ar_indices), 1000)  # both spans fully covered

    def test_deterministic_and_epoch_advances(self) -> None:
        sampler = StratifiedLanguageSampler(
            self._parts(), batch_size=10, seed=7, num_replicas=2, rank=1
        )
        first = list(sampler)
        second = list(sampler)
        sampler.set_epoch(1)
        third = list(sampler)
        sampler.set_epoch(0)
        fourth = list(sampler)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(first, fourth)

    def test_batch_size_must_divide_by_language_count(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            StratifiedLanguageSampler(self._parts(), batch_size=8, seed=1)
        self.assertIn("multiple of the number of languages", str(ctx.exception))

    def test_repeat_factors_reflect_language_sizes(self) -> None:
        sampler = StratifiedLanguageSampler(self._parts(), batch_size=10, seed=1)
        factors = sampler.repeat_factors
        self.assertEqual(factors["ar"], 1.0)
        self.assertAlmostEqual(factors["zh"], 1000 / 11, places=6)

    def test_language_above_affine_overflow_limit_rejected(self) -> None:
        """Beyond ~3.03e9 records the int64 affine map silently wraps and
        stops being a bijection (wrong records, broken coverage), so the
        constructor must refuse rather than corrupt sampling silently."""
        huge = make_part(3_100_000_000, 1.0, "ar")  # len() only; no allocation
        small = make_part(10, 1.0, "en")
        with self.assertRaises(ValueError) as ctx:
            StratifiedLanguageSampler([huge, small], batch_size=2, seed=1)
        self.assertIn("overflows", str(ctx.exception))

    def test_coverage_shortfall_never_occurs_despite_rounding_tail(self) -> None:
        """Per-rank round-up may DUPLICATE a few largest-language records
        (bounded by num_replicas * per_language) but must never skip one."""
        parts = [make_part(10, 1.0, "a"), make_part(7, 0.7, "b")]
        for num_replicas in (1, 2, 3, 5):
            seen: set[int] = set()
            for rank in range(num_replicas):
                sampler = StratifiedLanguageSampler(
                    parts, batch_size=4, seed=3,
                    num_replicas=num_replicas, rank=rank,
                )
                seen.update(sampler)
            self.assertEqual(seen, set(range(17)), f"R={num_replicas}")


if __name__ == "__main__":
    unittest.main()
