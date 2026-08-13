"""Tests for scripts/assemble_training_manifests.py.

Covers the file-classification engine (especially the _PART_RE fix for
distributed rejected part files) and the <<<>>> fence-stripping sanitisation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "assemble_training_manifests.py"

spec = importlib.util.spec_from_file_location("assemble_training_manifests", SCRIPT_PATH)
asm = importlib.util.module_from_spec(spec)
sys.modules["assemble_training_manifests"] = asm
spec.loader.exec_module(asm)


# ─── classify_file ──────────────────────────────────────────────────────────


class ClassifyPartFileExclusionTest(unittest.TestCase):
    """Regression: distributed *_rejected_pNNNN part files must be excluded.

    Before the fix these fell through to "base_merged" and became their own
    training corpus — 63 files / 1.45M rejected records in v7.0.
    """

    def test_still_rejected_part_is_excluded(self) -> None:
        corpus, role = asm.classify_file(
            Path("train_ar_inworld_full_still_rejected_p0019.jsonl")
        )
        self.assertEqual(role, "exclude")
        self.assertEqual(corpus, "train_ar_inworld_full")

    def test_plain_rejected_part_is_excluded(self) -> None:
        corpus, role = asm.classify_file(
            Path("train_ar_inworld_full_rejected_p0003.jsonl")
        )
        self.assertEqual(role, "exclude")
        self.assertEqual(corpus, "train_ar_inworld_full")

    def test_selected_part_is_excluded(self) -> None:
        corpus, role = asm.classify_file(
            Path("train_ar_inworld_full_selected_p0005.jsonl")
        )
        self.assertEqual(role, "exclude")
        self.assertEqual(corpus, "train_ar_inworld_full")

    def test_suspect_recovered_part_is_still_an_overlay(self) -> None:
        _corpus, role = asm.classify_file(
            Path("train_ar_x_suspect_recovered_p0003.jsonl")
        )
        self.assertEqual(role, "overlay")

    def test_recovered_part_is_still_an_overlay(self) -> None:
        _corpus, role = asm.classify_file(
            Path("train_ar_inworld_full_still_recovered_p0019.jsonl")
        )
        self.assertEqual(role, "overlay")

    def test_cleaned_and_bare_still_work(self) -> None:
        self.assertEqual(
            asm.classify_file(Path("train_ar_x_cleaned.jsonl")),
            ("train_ar_x", "base_merged"),
        )
        self.assertEqual(
            asm.classify_file(Path("train_ar_x.jsonl")),
            ("train_ar_x", "base_merged"),
        )


# ─── fence stripping ───────────────────────────────────────────────────────


class FenceStrippingTest(unittest.TestCase):
    """<<<>>> fences leaked into ~1% of v7.0 records. Verify they are removed."""

    def _assemble_texts(self, records: list[dict]) -> list[str]:
        """Write records, run assemble_corpus, return resulting texts."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus_cleaned.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            merged, _stats = asm.assemble_corpus([path], [])
        return [r["text"] for r in merged]

    def test_fences_are_stripped_from_text(self) -> None:
        records = [
            {"audio_filepath": "/a.wav", "text": "<<<مرحبا>>>", "original_text": "x"},
            {"audio_filepath": "/b.wav", "text": "نص نظيف", "original_text": "y"},
        ]
        texts = self._assemble_texts(records)
        self.assertIn("مرحبا", texts)
        self.assertNotIn("<<<", " ".join(texts))
        self.assertNotIn(">>>", " ".join(texts))

    def test_record_with_only_fences_is_dropped(self) -> None:
        """If stripping fences leaves empty text, the record is invalid."""
        records = [
            {"audio_filepath": "/a.wav", "text": "<<<>>>", "original_text": "x"},
            {"audio_filepath": "/b.wav", "text": "ok", "original_text": "y"},
        ]
        texts = self._assemble_texts(records)
        self.assertEqual(texts, ["ok"])

    def test_strip_record_also_cleans_fences(self) -> None:
        """The training-field stripper (used at write time) also removes fences."""
        out = asm.strip_record(
            {"audio_filepath": "/a.wav", "text": "<<<hello>>>", "duration": 2.0}
        )
        self.assertEqual(out["text"], "hello")

    def test_partial_fences_are_stripped(self) -> None:
        """A text that opens a fence but doesn't close it is still cleaned."""
        out = asm.strip_record(
            {"audio_filepath": "/a.wav", "text": "<<<start of text", "duration": 2.0}
        )
        self.assertEqual(out["text"], "start of text")


# ─── end-to-end main ───────────────────────────────────────────────────────


class EndToEndTest(unittest.TestCase):
    def test_rejected_parts_are_excluded_and_fences_stripped(self) -> None:
        """Integration test: a rejected part file must not appear in output."""
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "cleaned" / "ar"
            in_dir.mkdir(parents=True)
            # Good corpus
            good = [
                {"audio_filepath": "/a.wav", "text": "<<<نص محمي>>>",
                 "original_text": "x", "duration": 2.0},
                {"audio_filepath": "/b.wav", "text": "نص عادي",
                 "original_text": "y", "duration": 3.0},
            ]
            (in_dir / "train_ar_x_cleaned.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in good) + "\n",
                encoding="utf-8",
            )
            # Rejected part file (previously leaked)
            rej = [
                {"audio_filepath": "/c.wav", "text": "should be rejected",
                 "original_text": "z", "duration": 1.0},
            ]
            (in_dir / "train_ar_x_still_rejected_p0000.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rej) + "\n",
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "out"
            # Run the CLI directly
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH),
                 "--input-dir", str(in_dir.parent),
                 "--output-dir", str(out_dir),
                 "--version", "v99"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = (out_dir / "v99" / "ar" / "train_ar_x.jsonl")
            self.assertTrue(output.exists(), f"expected {output}")
            records = [json.loads(line) for line in output.read_text().strip().split("\n")]
            texts = [r["text"] for r in records]
            # Fences stripped, rejected not included
            self.assertIn("نص محمي", texts)
            self.assertNotIn("<<<", " ".join(texts))
            self.assertNotIn("should be rejected", " ".join(texts))
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
