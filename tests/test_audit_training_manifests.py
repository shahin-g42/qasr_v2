"""Tests for scripts/audit_training_manifests.py.

The auditor is the instrument used to judge a downloaded snapshot, so its own
judgements have to be trustworthy: it must tolerate a half-written final line
(downloads in flight), separate corrupted text from a merely-missing duration,
and catch rejected piles that were registered as training corpora.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_training_manifests.py"

# The auditor inserts scripts/ onto sys.path and imports prepare_q3asr_filter;
# do the same here so the import resolves the same way it does in production.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("audit_training_manifests", SCRIPT_PATH)
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_training_manifests"] = audit
spec.loader.exec_module(audit)


def write_jsonl(path: Path, records: list[dict], *, truncated_tail: bool = False) -> None:
    """Write records one per line; optionally leave the last line half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    text = "\n".join(lines)
    if truncated_tail:
        # Simulate a download caught mid-write: a valid prefix then a partial
        # object with no closing brace and no trailing newline.
        text += '\n{"audio_filepath": "/x/y.wav", "te'
    else:
        text += "\n"
    path.write_text(text, encoding="utf-8")


class NonTrainingDetectionTest(unittest.TestCase):
    def test_rejected_part_file_is_flagged_non_training(self) -> None:
        """The exact leak found on disk: a `_still_rejected_pNNNN` part file."""
        report = audit.FileReport(
            Path("train_ar_inworld_full_still_rejected_p0019.jsonl"), "ar"
        )
        self.assertTrue(report.is_non_training)

    def test_ordinary_corpus_is_training(self) -> None:
        report = audit.FileReport(Path("train_ar_q3asr_r10909274_13091129.jsonl"), "ar")
        self.assertFalse(report.is_non_training)

    def test_suspect_and_selected_are_also_non_training(self) -> None:
        for name in ("train_x_suspect.jsonl", "train_x_selected.jsonl"):
            self.assertTrue(audit.FileReport(Path(name), "ar").is_non_training)


class TailTruncationTest(unittest.TestCase):
    def test_partial_final_line_is_tolerated_not_a_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            write_jsonl(
                path,
                [{"audio_filepath": "/a.wav", "text": "مرحبا", "duration": 2.0}] * 5,
                truncated_tail=True,
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertTrue(report.tail_truncated)
            self.assertEqual(report.parse_errors, [])  # the tail is not counted as a defect
            self.assertEqual(report.records, 5)

    def test_a_broken_line_in_the_middle_is_a_real_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            path.write_text(
                '{"audio_filepath": "/a.wav", "text": "ok", "duration": 2.0}\n'
                "{ this is not json }\n"
                '{"audio_filepath": "/b.wav", "text": "ok2", "duration": 2.0}\n',
                encoding="utf-8",
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertFalse(report.tail_truncated)
            self.assertEqual(len(report.parse_errors), 1)


class DefectClassificationTest(unittest.TestCase):
    def test_missing_duration_is_not_a_content_defect(self) -> None:
        """A missing duration is recoverable by a probe, so it must not be

        lumped in with corrupted text or the headline rate becomes a lie."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            write_jsonl(
                path,
                [{"audio_filepath": "/a.wav", "text": "نص سليم"}] * 4,  # no duration
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertEqual(report.duration_missing, 4)
            self.assertEqual(report.content_defects, 0)

    def test_llm_fence_is_counted_as_a_content_defect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            write_jsonl(
                path,
                [
                    {"audio_filepath": "/a.wav", "text": "<<<نص>>>", "duration": 2.0},
                    {"audio_filepath": "/b.wav", "text": "نص نظيف", "duration": 2.0},
                ],
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertEqual(report.artifacts["llm_fence"], 1)
            self.assertEqual(report.content_defects, 1)

    def test_ellipsis_is_not_flagged_as_repeated_punct(self) -> None:
        """The refinement that dropped 147k false positives: `...` is legitimate."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            write_jsonl(
                path,
                [{"audio_filepath": "/a.wav", "text": "لحظة...", "duration": 2.0}],
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertEqual(report.artifacts.get("repeated_punct", 0), 0)

    def test_out_of_band_durations_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_ar_x.jsonl"
            write_jsonl(
                path,
                [
                    {"audio_filepath": "/a.wav", "text": "x", "duration": 0.05},   # too short
                    {"audio_filepath": "/b.wav", "text": "y", "duration": 40.0},   # too long
                    {"audio_filepath": "/c.wav", "text": "z", "duration": 3.0},    # fine
                ],
            )
            report = audit.audit_file(path, "ar", None, None, True, 8)
            self.assertEqual(report.too_short, 1)
            self.assertEqual(report.too_long_duration, 1)


class DuplicationAndOverlapTest(unittest.TestCase):
    def test_reused_audio_and_eval_overlap_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "ar" / "eval_ar_x.jsonl"
            train_path = root / "ar" / "train_ar_x.jsonl"
            write_jsonl(eval_path, [{"audio_filepath": "/shared.wav", "text": "e", "duration": 2.0}])
            write_jsonl(
                train_path,
                [
                    {"audio_filepath": "/shared.wav", "text": "t", "duration": 2.0},   # in eval too
                    {"audio_filepath": "/dup.wav", "text": "a", "duration": 2.0},
                    {"audio_filepath": "/dup.wav", "text": "b", "duration": 2.0},      # reused audio
                ],
            )
            eval_hashes = audit.collect_eval_hashes([(eval_path, "ar")], "ar")
            report = audit.audit_file(train_path, "ar", None, eval_hashes, True, 8)
            self.assertEqual(report.eval_overlap, 1)
            self.assertEqual(report.dupe_audio, 1)


class ManifestCrossCheckTest(unittest.TestCase):
    def _snapshot(self, tmp: str) -> Path:
        root = Path(tmp) / "v9.9"
        write_jsonl(
            root / "ar" / "train_ar_good.jsonl",
            [{"audio_filepath": "/a.wav", "text": "نص", "duration": 2.0}] * 3,
        )
        write_jsonl(
            root / "ar" / "train_ar_x_rejected_p0000.jsonl",
            [{"audio_filepath": "/b.wav", "text": "نص", "duration": 2.0}] * 2,
        )
        manifest = {
            "version": "v9.9",
            "created": "now",
            "input_dir": "/somewhere",
            "languages": {
                "ar": {
                    "corpora": {
                        "train_ar_good": {"total": 3, "files": ["train_ar_good.jsonl"]},
                        # The bug: a rejected pile registered as a training corpus.
                        "train_ar_x_rejected_p0000": {
                            "total": 2,
                            "files": ["train_ar_x_rejected_p0000.jsonl"],
                        },
                    }
                }
            },
        }
        (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_registered_rejected_pile_is_reported_as_leaked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._snapshot(tmp)
            reports = [
                audit.audit_file(p, p.parent.name, None, None, True, 8)
                for p in sorted(root.rglob("*.jsonl"))
            ]
            info = audit.cross_check_manifest(root, reports)
            self.assertTrue(info["present"])
            self.assertEqual(len(info["leaked"]), 1)
            self.assertEqual(info["leaked"][0][1], "train_ar_x_rejected_p0000.jsonl")

    def test_count_mismatch_is_reported_for_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._snapshot(tmp)
            # Corrupt the good corpus count so on-disk (3) != registered (99).
            manifest = json.loads((root / "MANIFEST.json").read_text())
            manifest["languages"]["ar"]["corpora"]["train_ar_good"]["total"] = 99
            (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            reports = [
                audit.audit_file(p, p.parent.name, None, None, True, 8)
                for p in sorted(root.rglob("*.jsonl"))
            ]
            info = audit.cross_check_manifest(root, reports)
            keys = {m[0] for m in info["mismatched"]}
            self.assertIn("ar/train_ar_good.jsonl", keys)

    def test_end_to_end_main_runs_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._snapshot(tmp)
            self.assertEqual(audit.main([str(root)]), 0)


if __name__ == "__main__":
    unittest.main()
