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


class ClassifyResidualLeakVariantsTest(unittest.TestCase):
    """Regression: residual variants of the v7.0 *_rejected_pNNNN leak.

    Same failure class as the original incident (63 files / 1.45M rejected
    records shipped as training corpora): non-corpus artifacts falling
    through to "base_merged". Covers odd-width part indices, bare _suspect
    parts, audit.py *_safe splits, and _cleaned overlay keying.
    """

    def test_part_suffix_widths_for_every_role(self) -> None:
        """3/4/5-digit _pN part files classify by role, never base_merged.

        {slice_idx:04d} formats slice indices >= 10000 as 5 digits, and the
        old _PART_RE demanded exactly 4 — so *_rejected_p00123 (and any
        3-digit variant) leaked as its own corpus. Bare _suspect was missing
        from the alternation entirely, so *_suspect_p0007 leaked too.
        """
        cases = [
            ("suspect_recovered", "overlay"),
            ("suspect", "exclude"),
            ("recovered", "overlay"),
            ("still_rejected", "exclude"),
            ("rejected", "exclude"),
            ("selected", "exclude"),
        ]
        for role_name, want_role in cases:
            for digits in ("019", "0019", "00019"):
                name = f"train_ar_x_{role_name}_p{digits}.jsonl"
                with self.subTest(name=name):
                    corpus, role = asm.classify_file(Path(name))
                    self.assertEqual(role, want_role)
                    self.assertEqual(corpus, "train_ar_x")

    def test_audit_safe_split_is_excluded(self) -> None:
        """audit.py writes <stem>_safe.jsonl beside every audited manifest.

        Its records already ship via the base file; before the fix
        train_ar_x_cleaned_safe registered as its own base corpus and the
        records shipped twice.
        """
        for name in ("train_ar_x_cleaned_safe.jsonl", "train_ar_x_safe.jsonl"):
            with self.subTest(name=name):
                corpus, role = asm.classify_file(Path(name))
                self.assertEqual(role, "exclude")
                self.assertEqual(corpus, "train_ar_x")

    def test_cleaned_recovery_chain_keys_to_parent_corpus(self) -> None:
        """Reprocessed audit-suspect files must overlay the parent corpus.

        Reprocessing x_cleaned_suspect.jsonl yields
        x_cleaned_suspect_recovered.jsonl; before the fix its overlay keyed
        to phantom corpus "x_cleaned" while the base keyed to "x", so the
        recovery shipped as a SEPARATE training file instead of overlaying.
        """
        self.assertEqual(
            asm.classify_file(Path("train_ar_x_cleaned.jsonl")),
            ("train_ar_x", "base_merged"),
        )
        self.assertEqual(
            asm.classify_file(Path("train_ar_x_cleaned_suspect_recovered.jsonl")),
            ("train_ar_x", "overlay"),
        )
        self.assertEqual(
            asm.classify_file(Path("train_ar_x_cleaned_suspect_recovered_p0002.jsonl")),
            ("train_ar_x", "overlay"),
        )
        self.assertEqual(
            asm.classify_file(Path("train_ar_x_cleaned_suspect.jsonl")),
            ("train_ar_x", "exclude"),
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

    def test_cleaned_suspect_recovery_folds_into_parent_corpus(self) -> None:
        """Integration for the v7.0-leak follow-up: audit/reprocess chains.

        x_cleaned_suspect_recovered.jsonl used to key to phantom corpus
        "x_cleaned" and ship as its own training file; x_cleaned_safe.jsonl
        (audit split) and 5-digit rejected parts leaked as base corpora.
        """
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "cleaned" / "ar"
            in_dir.mkdir(parents=True)
            base = [
                {"audio_filepath": "/a.wav", "text": "old text",
                 "original_text": "x", "duration": 2.0},
                {"audio_filepath": "/b.wav", "text": "keep",
                 "original_text": "y", "duration": 3.0},
            ]
            (in_dir / "train_ar_x_cleaned.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in base) + "\n",
                encoding="utf-8",
            )
            # audit.py safe split — duplicates the base, must be excluded
            (in_dir / "train_ar_x_cleaned_safe.jsonl").write_text(
                json.dumps(base[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            # suspect re-clean recovery — must overlay corpus train_ar_x
            recovered = {"audio_filepath": "/a.wav", "text": "recovered text",
                         "original_text": "x", "duration": 2.0}
            (in_dir / "train_ar_x_cleaned_suspect_recovered.jsonl").write_text(
                json.dumps(recovered, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            # 5-digit rejected part (slice index >= 10000) — must be excluded
            rej = {"audio_filepath": "/c.wav", "text": "bad",
                   "original_text": "z", "duration": 1.0}
            (in_dir / "train_ar_x_rejected_p00123.jsonl").write_text(
                json.dumps(rej, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "out"
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH),
                 "--input-dir", str(in_dir.parent),
                 "--output-dir", str(out_dir),
                 "--version", "v99"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lang_dir = out_dir / "v99" / "ar"
            written = sorted(p.name for p in lang_dir.iterdir())
            # ONE file per corpus: no phantom train_ar_x_cleaned.jsonl,
            # no leaked _safe or rejected-part corpora
            self.assertEqual(written, ["train_ar_x.jsonl"])
            records = [
                json.loads(line) for line in
                (lang_dir / "train_ar_x.jsonl").read_text().strip().split("\n")
            ]
            texts = {r["text"] for r in records}
            self.assertEqual(len(records), 2)
            self.assertIn("recovered text", texts)  # overlay replaced the base
            self.assertNotIn("old text", texts)
            self.assertNotIn("bad", texts)


if __name__ == "__main__":
    unittest.main()
