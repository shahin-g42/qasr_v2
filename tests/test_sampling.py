"""Tests for the language-balanced distributed sampler."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from qasr.sampling import LanguageBalancedSampler, language_mix_weights


def make_part(size: int, hours: float):
    """Minimal stand-in for JsonlSpeechDataset (stats + __len__)."""

    class FakePart:
        def __init__(self) -> None:
            self.stats = SimpleNamespace(total_kept_hours=hours, kept_records=size)

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


if __name__ == "__main__":
    unittest.main()
