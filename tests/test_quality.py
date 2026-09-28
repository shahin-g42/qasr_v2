"""Tests for the hard gates: the richness metric, its floor, and text-only mode.

These cover the additions the staged build relies on -- a transcript can clear
every correctness gate and still be worthless for training ("yes", one repeated
word), so richness is scored from text alone and can gate it. Text-only mode
(``require_duration=False``) lets Stage 1 gate and rank external samples whose
duration is unknown until the audio is fetched, without dropping every row.
"""

from __future__ import annotations

import unittest

from data_processing.quality import GateResult, QualityConfig, Verdict, gate

_AR_RICH = "سبحان الله كل شي تطور دفعة واحد في كل المجالات الحديثة هذه الأيام"
_EN_RICH = "the quick brown fox jumps over the lazy dog while the cat watches quietly"
_ZH_RICH = "今天天气非常好我们一起去公园散步聊天顺便买点东西"


class TestRichnessMetric(unittest.TestCase):
    def test_rich_text_outscores_trivial_text(self):
        cfg = QualityConfig()
        rich = gate(_EN_RICH, "en", None, cfg, require_duration=False).richness
        trivial = gate("yes", "en", None, cfg, require_duration=False).richness
        repeated = gate("yes yes yes yes yes", "en", None, cfg, require_duration=False).richness
        self.assertGreater(rich, 0.6)
        self.assertLess(trivial, 0.2)
        self.assertLess(repeated, 0.25)
        self.assertGreater(rich, repeated)

    def test_single_token_is_not_rewarded_for_a_perfect_type_token_ratio(self):
        """TTR is trivially 1.0 for one token; the distinct>=2 guard must catch it."""
        cfg = QualityConfig()
        one = gate("hello", "en", None, cfg, require_duration=False).richness
        self.assertLess(one, 0.3)

    def test_repetition_scores_below_variety_in_every_language(self):
        cfg = QualityConfig()
        for lang, rich, spam in [
            ("ar", _AR_RICH, "نعم نعم نعم نعم نعم نعم"),
            ("zh", _ZH_RICH, "好好好好好好好好好好"),
        ]:
            r = gate(rich, lang, None, cfg, require_duration=False).richness
            s = gate(spam, lang, None, cfg, require_duration=False).richness
            self.assertGreater(r, s, f"{lang}: rich {r} !> spam {s}")

    def test_richness_is_recorded_in_metrics(self):
        cfg = QualityConfig()
        res = gate(_EN_RICH, "en", None, cfg, require_duration=False)
        self.assertIn("richness", res.metrics)
        self.assertAlmostEqual(res.metrics["richness"], res.richness, places=4)


class TestMinRichnessGate(unittest.TestCase):
    def test_default_zero_never_rejects_on_richness(self):
        cfg = QualityConfig()
        self.assertEqual(cfg.min_richness, 0.0)
        res = gate("yes", "en", 1.0, cfg)
        self.assertTrue(res.ok)
        self.assertNotIn("low_richness", "|".join(res.reasons))

    def test_floor_drops_the_trivial_tail_but_keeps_real_content(self):
        cfg = QualityConfig(min_richness=0.30)
        self.assertFalse(gate("yes", "en", None, cfg, require_duration=False).ok)
        self.assertFalse(gate("yes yes yes yes yes", "en", None, cfg, require_duration=False).ok)
        self.assertTrue(gate(_EN_RICH, "en", None, cfg, require_duration=False).ok)

    def test_rejection_names_the_reason_and_zeroes_quality(self):
        cfg = QualityConfig(min_richness=0.99)
        res = gate(_EN_RICH, "en", 4.0, cfg)
        self.assertIs(res.verdict, Verdict.REJECT)
        self.assertTrue(any(r.startswith("low_richness") for r in res.reasons))
        self.assertEqual(res.quality, 0.0)


class TestCompositeRank(unittest.TestCase):
    def test_composite_is_quality_times_richness(self):
        cfg = QualityConfig()
        res = gate(_EN_RICH, "en", 4.0, cfg)
        self.assertAlmostEqual(res.composite, round(res.quality * res.richness, 4), places=4)

    def test_composite_separates_rich_from_trivial_when_both_pass(self):
        cfg = QualityConfig()
        rich = gate(_EN_RICH, "en", 4.0, cfg).composite
        trivial = gate("hello there friend", "en", 4.0, cfg).composite
        self.assertGreater(rich, trivial)


class TestTextOnlyMode(unittest.TestCase):
    def test_missing_duration_is_not_rejected_by_default(self):
        # gate_duration defaults to False: a missing duration never rejects, and
        # require_duration=True is moot while the duration gates are disarmed.
        res = gate(_EN_RICH, "en", None, QualityConfig())
        self.assertTrue(res.ok)
        self.assertNotIn("units_per_sec", res.metrics)

    def test_text_only_mode_passes_without_a_duration(self):
        res = gate(_EN_RICH, "en", None, QualityConfig(), require_duration=False)
        self.assertTrue(res.ok)
        self.assertNotIn("units_per_sec", res.metrics)
        self.assertGreater(res.quality, 0.0)  # rate term dropped, not zeroed

    def test_text_only_mode_still_applies_text_gates(self):
        cfg = QualityConfig()
        self.assertFalse(gate("", "en", None, cfg, require_duration=False).ok)
        self.assertFalse(gate("visit https://spam.example now", "en", None, cfg,
                              require_duration=False).ok)
        # script purity still binds in text-only mode
        self.assertFalse(gate("这是中文", "en", None, cfg, require_duration=False).ok)

    def test_text_only_score_is_not_destroyed_by_the_absent_rate_term(self):
        """Regression: defaulting units_per_sec to 0.0 read as 'infinitely slow'
        and multiplied the score to zero."""
        cfg = QualityConfig()
        text_only = gate(_EN_RICH, "en", None, cfg, require_duration=False).quality
        self.assertGreater(text_only, 0.5)

    def test_duration_band_binds_when_gating_armed(self):
        cfg = QualityConfig(gate_duration=True)
        in_band = gate(_EN_RICH, "en", 4.0, cfg)
        out_of_band = gate(_EN_RICH, "en", 999.0, cfg)
        self.assertTrue(in_band.ok)
        self.assertIn("units_per_sec", in_band.metrics)
        self.assertIs(out_of_band.verdict, Verdict.REJECT)
        self.assertTrue(out_of_band.reasons[0].startswith("duration_out_of_band"))


class TestDurationGateSwitch(unittest.TestCase):
    """``gate_duration`` is the master switch for every duration-derived gate."""

    def test_off_by_default_never_rejects_on_duration(self):
        cfg = QualityConfig()
        self.assertFalse(cfg.gate_duration)
        # missing, far too short, far too long -- none reject while disarmed
        self.assertTrue(gate(_EN_RICH, "en", None, cfg).ok)
        self.assertTrue(gate(_EN_RICH, "en", 0.05, cfg).ok)
        self.assertTrue(gate(_EN_RICH, "en", 999.0, cfg).ok)

    def test_out_of_band_duration_is_measured_but_not_gating(self):
        cfg = QualityConfig()
        res = gate(_EN_RICH, "en", 999.0, cfg)
        self.assertTrue(res.ok)
        self.assertIn("units_per_sec", res.metrics)  # measured for ranking, not gating

    def test_armed_rejects_missing_out_of_band_and_bad_rate(self):
        cfg = QualityConfig(gate_duration=True)
        self.assertEqual(gate(_EN_RICH, "en", None, cfg).reasons,
                         ("duration_missing_or_nonpositive",))
        self.assertTrue(
            gate(_EN_RICH, "en", 999.0, cfg).reasons[0].startswith("duration_out_of_band"))
        # 14 words in 1.0s is far above the en ceiling (7 words/s)
        self.assertTrue(
            gate(_EN_RICH, "en", 1.0, cfg).reasons[0].startswith("speech_rate_high"))


class TestGateResultShape(unittest.TestCase):
    def test_richness_defaults_to_zero(self):
        self.assertEqual(GateResult(Verdict.PASS).richness, 0.0)
        self.assertEqual(GateResult(Verdict.PASS).composite, 0.0)


if __name__ == "__main__":
    unittest.main()
