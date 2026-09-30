"""Tests for the corrector prompt families (the Stage-2 quality upgrade).

The rich prompts are what make the corrector transcript quality the prime
focus: language-specialized cleaner system prompts (Arabic dialect/diacritics
aware, per-language conventions for en/zh/hi/ml) plus batch user templates
whose numbering contract matches what ``BatchCorrector``'s parser reads back.
These tests pin the guardrails that make the rich prompts safe to trust --
rule-0 verbatim fidelity, the ITN guardrails, the 5-15% Arabic diacritics
target, the strict-JSON contracts -- and that ``--llm-prompt compact``
reproduces the old prompt byte-for-byte, so the fallback is a known quantity.
"""

from __future__ import annotations

import json
import unittest

from data_processing.accent import preservation_block
from data_processing.build_corpus import BatchCorrector, BuildConfig
from data_processing.generic_prompts import (
    GENERIC_JUDGE_BATCH_SYSTEM_PROMPT,
    build_generic_batch_cleaner_messages,
    build_generic_cleaner_system_prompt,
)
from data_processing.prompts import (
    _RULE_ITN,
    _RULE_VERBATIM,
    JUDGE_BATCH_SYSTEM_PROMPT,
    build_batch_cleaner_messages,
    build_cleaner_system_prompt,
)


class TestArabicRichSystemPrompt(unittest.TestCase):
    def test_rule0_verbatim_block_is_present_and_first(self):
        system = build_cleaner_system_prompt()
        # The full rule block, verbatim -- not a paraphrase of it.
        self.assertIn(_RULE_VERBATIM, system)
        # ...and it leads the rules, because it overrides the ones below.
        self.assertLess(system.index("0. VERBATIM FIDELITY"),
                        system.index("1. CLEANING"))

    def test_itn_guardrails(self):
        system = build_cleaner_system_prompt()
        self.assertIn(_RULE_ITN, system)
        self.assertIn("Convert EXACTLY the number spoken", system)
        self.assertIn("NEVER convert between Hijri and Gregorian", system)
        self.assertIn("number-noun agreement", system)

    def test_diacritics_target_band(self):
        system = build_cleaner_system_prompt()
        self.assertIn("Target: 5-15% of letters carry marks", system)
        self.assertIn("Do NOT fully vocalize", system)

    def test_no_diacritics_mode(self):
        system = build_cleaner_system_prompt(restore_diacritics=False)
        self.assertIn("Do NOT add any diacritical marks", system)
        self.assertNotIn("Target: 5-15%", system)

    def test_dialect_rule_adapts(self):
        preserve = build_cleaner_system_prompt(preserve_dialects=True)
        msa = build_cleaner_system_prompt(preserve_dialects=False)
        self.assertIn("Do NOT normalize colloquial forms to MSA", preserve)
        self.assertIn("Normalize dialectal forms to Modern Standard Arabic", msa)

    def test_fewshot_anchor_only_in_full_mode(self):
        full = build_cleaner_system_prompt()
        self.assertIn("Example:", full)
        partial = build_cleaner_system_prompt(restore_diacritics=False)
        self.assertNotIn("Example:", partial)


class TestArabicBatchTemplate(unittest.TestCase):
    def test_user_message_numbers_transcripts_from_one(self):
        msgs = build_batch_cleaner_messages(["أول نص", "ثاني نص"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        user = msgs[1]["content"]
        self.assertIn("1. <<<أول نص>>>", user)
        self.assertIn("2. <<<ثاني نص>>>", user)

    def test_user_message_declares_the_strict_json_contract(self):
        user = build_batch_cleaner_messages(["نص"])[1]["content"]
        self.assertIn('"index": 1', user)
        self.assertIn("STRICT JSON array", user)
        self.assertIn("no markdown fences", user)

    def test_accent_label_appends_the_preservation_block(self):
        plain = build_batch_cleaner_messages(["نص"])
        tagged = build_batch_cleaner_messages(["نص"], accent_label="egyptian")
        self.assertNotIn("This sample is tagged", plain[0]["content"])
        block = preservation_block("ar", "egyptian")
        self.assertIn("This sample is tagged ar:egyptian.", tagged[0]["content"])
        self.assertIn(block, tagged[0]["content"])
        self.assertTrue(tagged[0]["content"].endswith(block))


class TestGenericRichPrompt(unittest.TestCase):
    def test_per_language_conventions_present(self):
        cases = {
            "en": ("don't", "$50", "10%"),
            "zh": ("。", "erhua", "No spaces between Chinese characters"),
            "hi": ("danda", "Hinglish"),
            "ml": ("Manglish",),
        }
        for lang, needles in cases.items():
            with self.subTest(lang=lang):
                system = build_generic_cleaner_system_prompt(lang)
                for needle in needles:
                    self.assertIn(needle, system)
                # The two invariants every language shares.
                self.assertIn("0. VERBATIM FIDELITY", system)
                self.assertIn("Never infer an absent year", system)

    def test_unknown_language_falls_back_to_default_conventions(self):
        system = build_generic_cleaner_system_prompt("zz")
        self.assertIn("Use the standard punctuation of the language.", system)
        self.assertIn("0. VERBATIM FIDELITY", system)

    def test_batch_user_numbers_from_zero_with_i_contract(self):
        msgs = build_generic_batch_cleaner_messages(["one", "two"], "en")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        user = msgs[1]["content"]
        self.assertIn("0. <<<one>>>", user)
        self.assertIn("1. <<<two>>>", user)
        self.assertIn('"i": 0', user)
        self.assertIn("STRICT JSON array", user)

    def test_accent_label_appends_the_preservation_block(self):
        msgs = build_generic_batch_cleaner_messages(["一行"], "zh", accent_label="hant")
        block = preservation_block("zh", "hant")
        self.assertIn("This sample is tagged zh:hant.", msgs[0]["content"])
        self.assertTrue(msgs[0]["content"].endswith(block))


_MODE_ITEMS = [("first", "egyptian"), ("second", "egyptian")]


class TestBatchCorrectorPromptModes(unittest.TestCase):

    @staticmethod
    def _corrector(mode: str) -> BatchCorrector:
        return BatchCorrector(BuildConfig(
            llm_url="http://localhost:8010/v1", llm_prompt=mode))

    def test_compact_reproduces_the_old_prompt_byte_for_byte(self):
        msgs = self._corrector("compact")._messages("ar", "egyptian", _MODE_ITEMS)
        expected_system = (
            "You correct ASR transcripts for speech-to-text training data.\n"
            "Fix spelling errors and add punctuation where it is missing.\n"
            "Apply inverse text normalization: spoken numbers, dates, times and "
            "currencies become their written form.\n"
            "NEVER add diacritics or vocalization that the source does not have.\n"
            "NEVER convert colloquial or dialectal speech into the standard register.\n"
            "NEVER paraphrase, reorder, or drop words.\n"
            + preservation_block("ar", "egyptian")
            + "\nReturn JSON: {\"items\": [{\"i\": <int>, \"text\": <str>, "
            "\"dialect\": <str>, \"confidence\": <float 0-1>}]}"
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], expected_system)
        self.assertEqual(
            msgs[1]["content"],
            json.dumps([{"i": 0, "text": "first"}, {"i": 1, "text": "second"}],
                      ensure_ascii=False),
        )

    def test_rich_arabic_selects_the_arabic_batch_builder(self):
        msgs = self._corrector("rich")._messages("ar", "egyptian", _MODE_ITEMS)
        self.assertEqual(msgs, build_batch_cleaner_messages(
            ["first", "second"], accent_label="egyptian"))

    def test_rich_generic_selects_the_generic_batch_builder(self):
        for lang in ("en", "zh", "hi", "ml"):
            with self.subTest(lang=lang):
                msgs = self._corrector("rich")._messages(lang, None, [("x", None)])
                self.assertEqual(
                    msgs, build_generic_batch_cleaner_messages(["x"], lang))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            BatchCorrector(BuildConfig(llm_url="http://x/v1", llm_prompt="bogus"))


class TestJudgePromptPolicy(unittest.TestCase):
    """Stage 3 is an aggressive filter: doubt defaults to rejection, and
    every rejection carries a machine-readable tag so a later repair pass
    can triage. Surface variety (dialect, code-switching) stays protected."""

    _SYSTEMS = (JUDGE_BATCH_SYSTEM_PROMPT, GENERIC_JUDGE_BATCH_SYSTEM_PROMPT)
    _TAGS = ("repetition", "cutoff", "nonword", "incoherent",
             "nonsense", "hallucination", "other")

    def test_doubt_defaults_to_rejection(self):
        for system in self._SYSTEMS:
            with self.subTest(prompt="arabic" if system is
                              JUDGE_BATCH_SYSTEM_PROMPT else "generic"):
                self.assertIn("When in doubt, REJECT", system)
                self.assertNotIn("When in doubt, keep", system)

    def test_reject_classes_cover_the_observed_failures(self):
        for system in self._SYSTEMS:
            with self.subTest(prompt="arabic" if system is
                              JUDGE_BATCH_SYSTEM_PROMPT else "generic"):
                for needle in ("repetition", "cutoff", "nonword",
                               "incoherent", "hallucination",
                               "a single clearly corrupted word"):
                    self.assertIn(needle, system)

    def test_surface_variety_stays_protected(self):
        for system in self._SYSTEMS:
            with self.subTest(prompt="arabic" if system is
                              JUDGE_BATCH_SYSTEM_PROMPT else "generic"):
                self.assertIn("code-switching", system)
                self.assertIn("short but complete", system)

    def test_rejections_carry_issue_tags_for_the_repair_pass(self):
        for system in self._SYSTEMS:
            with self.subTest(prompt="arabic" if system is
                              JUDGE_BATCH_SYSTEM_PROMPT else "generic"):
                self.assertIn('"issues"', system)
                for tag in self._TAGS:
                    self.assertIn(tag, system)

    def test_no_template_markers_in_judge_prompts(self):
        # <<< >>> belongs to the user templates only; a stray marker in a
        # system prompt would leak into every batch and corrupt parsing.
        for system in self._SYSTEMS:
            with self.subTest(prompt="arabic" if system is
                              JUDGE_BATCH_SYSTEM_PROMPT else "generic"):
                self.assertNotIn("<<<", system)
                self.assertNotIn(">>>", system)


if __name__ == "__main__":
    unittest.main()
