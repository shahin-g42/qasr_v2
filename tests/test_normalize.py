"""Regression tests for normalization fidelity fixes.

Two bugs these guard against:

1. U+0671 (Alef Wasla) is a LETTER, not a diacritic -- it used to sit in
   ARABIC_OTHER_MARKS and was silently deleted, corrupting words.
2. collapse_repeated_chars collapsed 5+ DIGIT runs, damaging spoken digit
   sequences ("111111" is six spoken digits, not filler).
"""

from __future__ import annotations

from data_processing.arabic_utils import collapse_repeated_chars
from data_processing.normalize import DiacriticPolicy, dedup_key, normalize


def _ar(text: str, policy: DiacriticPolicy = DiacriticPolicy.CRITICAL_ONLY):
    return normalize(text, "ar", diacritic_policy=policy)


def test_alef_wasla_is_folded_not_deleted() -> None:
    # The word for "Allah" starts with U+0671; the alef must survive.
    text = "\u0671\u0644\u0644\u0651\u064e\u0670\u0647"
    out, applied = _ar(text)
    assert out.count("\u0627") >= 1, out
    assert "\u0671" not in out
    assert "fold_alef_wasla" in applied
    # Letters intact: the marks-only removal must not have eaten the alef.
    assert out.startswith("\u0627")


def test_alef_wasla_equals_plain_alef_in_dedup_key() -> None:
    wasla = "\u0671\u0644\u0643\u062a\u0627\u0628"
    plain = "\u0627\u0644\u0643\u062a\u0627\u0628"
    assert dedup_key(wasla, "ar") == dedup_key(plain, "ar")


def test_alef_wasla_fold_is_idempotent() -> None:
    text = "\u0671\u0644\u0644\u0651\u064e\u0670\u0647 \u0641\u064a \u0671\u0644\u0628\u064a\u062a"
    once, _applied = _ar(text)
    twice, applied2 = _ar(once)
    assert once == twice
    assert "fold_alef_wasla" not in applied2


def test_strip_all_policy_still_keeps_alef_wasla_letter() -> None:
    text = "\u0671\u0644\u0643\u062a\u0627\u0628"
    out, _ = _ar(text, policy=DiacriticPolicy.STRIP_ALL)
    assert "\u0627\u0644\u0643\u062a\u0627\u0628" in out


def test_repeated_digits_are_never_collapsed() -> None:
    # Digit runs are spoken content: six ones stay six ones.
    assert collapse_repeated_chars("111111") == "111111"
    assert collapse_repeated_chars("\u0661\u0661\u0661\u0661\u0661") == "\u0661\u0661\u0661\u0661\u0661"


def test_repeated_letters_are_still_collapsed() -> None:
    assert collapse_repeated_chars("soooooo good") == "sooo good"
    assert collapse_repeated_chars("wwwwwwow") == "wwwow"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
