"""Regression tests for transcript-faithful WER/CER scoring."""

import unittest

from qasr.evaluate import error_counts, normalize_text


class TextMetricsTest(unittest.TestCase):
    def test_combining_marks_do_not_split_words(self) -> None:
        for text, char_count in [("عَرَبِيّ", 8), ("बताया", 5), ("മലയാളം", 6)]:
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text), text)
                self.assertEqual(
                    error_counts(text, text),
                    {"word_errors": 0, "word_count": 1, "char_errors": 0, "char_count": char_count},
                )

    def test_missing_or_extra_marks_are_errors(self) -> None:
        cases = [
            ("علّم", "علم", 4),       # Arabic shadda
            ("كتابًا", "كتابا", 6),   # Arabic tanween
            ("बताया", "बताय", 5),    # Hindi vowel sign
            ("മലയാളം", "മലയാള", 6),  # Malayalam anusvara
        ]
        for marked, bare, char_count in cases:
            with self.subTest(reference=marked):
                self.assertEqual(
                    error_counts(marked, bare),
                    {"word_errors": 1, "word_count": 1, "char_errors": 1, "char_count": char_count},
                )
            with self.subTest(reference=bare):
                self.assertEqual(
                    error_counts(bare, marked),
                    {"word_errors": 1, "word_count": 1, "char_errors": 1, "char_count": char_count - 1},
                )

    def test_wrong_vowel_is_a_substitution(self) -> None:
        self.assertEqual(
            error_counts("علَم", "علِم"),
            {"word_errors": 1, "word_count": 1, "char_errors": 1, "char_count": 4},
        )

    def test_case_punctuation_and_number_spellings_are_significant(self) -> None:
        cases = [
            ("Hello.", "hello.", 1, 6),
            ("Hello.", "Hello", 1, 6),
            ("你好。", "你好", 1, 3),
            ("12", "١٢", 2, 2),
            ("Ａ", "A", 1, 1),  # NFC must not fold compatibility characters.
        ]
        for reference, hypothesis, errors, char_count in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(
                    error_counts(reference, hypothesis),
                    {"word_errors": 1, "word_count": 1, "char_errors": errors, "char_count": char_count},
                )

    def test_canonically_equivalent_unicode_has_zero_errors(self) -> None:
        cases = [
            ("café", "cafe\u0301", 4),
            ("أ", "ا\u0654", 1),
            ("क़", "क\u093c", 2),
            ("കൊ", "ക\u0d46\u0d3e", 2),
            ("ع\u0651\u064e", "ع\u064e\u0651", 3),
        ]
        for reference, hypothesis, char_count in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(
                    error_counts(reference, hypothesis),
                    {"word_errors": 0, "word_count": 1, "char_errors": 0, "char_count": char_count},
                )

    def test_whitespace_is_collapsed_for_words_and_excluded_from_cer(self) -> None:
        text = " \tA\nعلّم\u00a0 B  "
        self.assertEqual(normalize_text(text), "A علّم B")
        self.assertEqual(
            error_counts(text, "A علّم B"),
            {"word_errors": 0, "word_count": 3, "char_errors": 0, "char_count": 6},
        )
        self.assertEqual(
            error_counts("a b", "ab"),
            {"word_errors": 2, "word_count": 2, "char_errors": 0, "char_count": 2},
        )

    def test_empty_text_counts(self) -> None:
        self.assertEqual(
            error_counts(" \n", ""),
            {"word_errors": 0, "word_count": 0, "char_errors": 0, "char_count": 0},
        )
        self.assertEqual(
            error_counts("", "علّم"),
            {"word_errors": 1, "word_count": 0, "char_errors": 4, "char_count": 0},
        )
        self.assertEqual(
            error_counts("علّم", ""),
            {"word_errors": 1, "word_count": 1, "char_errors": 4, "char_count": 4},
        )


if __name__ == "__main__":
    unittest.main()
