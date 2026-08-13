import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qasr.audio import AudioLoadingError
from qasr.data import (
    CombinedSpeechDataset,
    JsonlSpeechDataset,
    ManifestError,
    ResilientAudioDataset,
)


class JsonlSpeechDatasetTest(unittest.TestCase):
    def test_aliases_relative_paths_and_duration_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            rows = [
                {"audio_filepath": "a.wav", "text": "مرحبا", "duration": 1.5},
                {"wav_path": "b.wav", "transcript": "أهلا", "duration": 2.5},
                {"wav_path": "too_long.wav", "transcript": "نص", "duration": 40.0},
            ]
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["text"], "مرحبا")
            self.assertEqual(dataset[1]["text"], "أهلا")
            self.assertEqual(Path(dataset[0]["audio_path"]), (root / "a.wav").resolve())
            self.assertEqual(dataset.stats.skipped_by_duration, 1)
            self.assertEqual(dataset.stats.skipped_by_text, 0)

    def test_missing_blank_and_punctuation_only_transcripts_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            rows = [
                {"wav_path": "missing.wav", "duration": 1.0},
                {"wav_path": "empty.wav", "text": "", "duration": 1.0},
                {"wav_path": "spaces.wav", "transcript": "  \t ", "duration": 1.0},
                {"wav_path": "ascii-punctuation.wav", "text": "...?!", "duration": 1.0},
                {"wav_path": "arabic-punctuation.wav", "text": "،؛؟", "duration": 1.0},
                {
                    "wav_path": "fallback.wav",
                    "text": " ",
                    "transcript": "valid fallback",
                    "duration": 1.0,
                },
                {"wav_path": "arabic.wav", "text": "مرحبا!", "duration": 1.0},
            ]
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["text"], "valid fallback")
            self.assertEqual(dataset[1]["text"], "مرحبا!")
            self.assertEqual(dataset.stats.total_records, 7)
            self.assertEqual(dataset.stats.skipped_by_text, 5)
            self.assertEqual(dataset.stats.skipped_by_duration, 0)

    def test_bad_record_has_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.jsonl"
            manifest.write_text('{"text": "missing audio", "duration": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, r"bad\.jsonl:1"):
                JsonlSpeechDataset(
                    manifest,
                    min_duration_seconds=0.1,
                    max_duration_seconds=30.0,
                )

    def test_vast_prefix_is_rewritten_for_both_audio_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            rows = [
                {"wav_path": "/vast/a.wav", "text": "أ", "duration": 1.0},
                {"audio_filepath": "/vast/b.wav", "text": "ب", "duration": 1.0},
            ]
            manifest.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
            )

            self.assertEqual(dataset[0]["audio_path"], "/lustrefs/taiga/vast40/a.wav")
            self.assertEqual(dataset[1]["audio_path"], "/lustrefs/taiga/vast40/b.wav")

    def test_manifest_language_is_attached_to_every_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "english.jsonl"
            manifest.write_text(
                json.dumps({"wav_path": "a.wav", "text": "hello", "duration": 1.0}) + "\n",
                encoding="utf-8",
            )
            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
                language="en",
            )

            self.assertEqual(dataset[0]["language"], "en")

    def test_missing_duration_is_validated_only_when_sample_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            rows = [
                {"wav_path": "a.wav", "text": "first"},
                {"wav_path": "b.wav", "text": "second", "duration": None},
            ]
            manifest.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with patch("qasr.data.load_mono_audio") as load_audio:
                indexed = JsonlSpeechDataset(
                    manifest,
                    min_duration_seconds=0.1,
                    max_duration_seconds=1.0,
                )

            self.assertEqual(len(indexed), 2)
            self.assertIsNone(indexed[0]["duration"])
            self.assertEqual(indexed.stats.missing_duration, 2)
            self.assertEqual(indexed.stats.total_kept_hours, 0.0)
            load_audio.assert_not_called()

            dataset = ResilientAudioDataset(
                indexed,
                sampling_rate=1_000,
                min_audio_seconds=0.1,
                max_audio_seconds=1.0,
            )
            with patch(
                "qasr.data.load_mono_audio",
                return_value=np.zeros(200, dtype=np.float32),
            ) as load_audio:
                sample = dataset[1]

            self.assertEqual(sample["text"], "second")
            load_audio.assert_called_once_with(str((root / "b.wav").resolve()), 1_000)

    def test_manifest_file_handle_is_reopened_after_process_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text(
                json.dumps({"wav_path": "a.wav", "text": "hello", "duration": 1.0})
                + "\n",
                encoding="utf-8",
            )
            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
            )
            first_handle = dataset._get_file()
            self.assertEqual(dataset._file_pid, os.getpid())

            # Simulate a forked worker inheriting a handle opened by its parent.
            dataset._file_pid = os.getpid() - 1
            second_handle = dataset._get_file()

            self.assertTrue(first_handle.closed)
            self.assertIsNot(first_handle, second_handle)
            self.assertEqual(dataset._file_pid, os.getpid())
            self.assertEqual(dataset[0]["text"], "hello")

    def test_multiple_manifests_are_concatenated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first_manifest = first_dir / "train.jsonl"
            second_manifest = second_dir / "train.jsonl"
            first_manifest.write_text(
                json.dumps({"wav_path": "a.wav", "text": "أ", "duration": 1.0}) + "\n",
                encoding="utf-8",
            )
            second_manifest.write_text(
                json.dumps({"wav_path": "b.wav", "text": "ب", "duration": 2.0}) + "\n",
                encoding="utf-8",
            )
            kwargs = {"min_duration_seconds": 0.1, "max_duration_seconds": 30.0}

            dataset = CombinedSpeechDataset(
                [
                    JsonlSpeechDataset(first_manifest, **kwargs),
                    JsonlSpeechDataset(second_manifest, **kwargs),
                ]
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["text"], "أ")
            self.assertEqual(dataset[1]["text"], "ب")
            self.assertEqual(Path(dataset[0]["audio_path"]), (first_dir / "a.wav").resolve())
            self.assertEqual(Path(dataset[1]["audio_path"]), (second_dir / "b.wav").resolve())
            self.assertAlmostEqual(dataset.stats.total_kept_hours, 3.0 / 3600.0)


class ResilientAudioDatasetTest(unittest.TestCase):
    def _dataset(self) -> ResilientAudioDataset:
        records = [
            {"audio_path": "too-short.wav", "text": "bad"},
            {"audio_path": "good.wav", "text": "replacement"},
        ]
        return ResilientAudioDataset(
            records,
            sampling_rate=1_000,
            min_audio_seconds=0.1,
            max_audio_seconds=1.0,
        )

    def test_too_short_audio_is_replaced_with_another_sample(self) -> None:
        waveforms = {
            "too-short.wav": np.zeros(28, dtype=np.float32),
            "good.wav": np.zeros(200, dtype=np.float32),
        }
        with patch(
            "qasr.data.load_mono_audio",
            side_effect=lambda path, _: waveforms[path],
        ):
            sample = self._dataset()[0]

        self.assertEqual(sample["audio_path"], "good.wav")
        self.assertEqual(sample["text"], "replacement")
        self.assertEqual(len(sample["waveform"]), 200)

    def test_decode_failure_is_replaced_and_cached(self) -> None:
        calls: list[str] = []

        def load(path: str, _: int) -> np.ndarray:
            calls.append(path)
            if path == "too-short.wav":
                raise AudioLoadingError("broken file")
            return np.zeros(200, dtype=np.float32)

        dataset = self._dataset()
        with patch("qasr.data.load_mono_audio", side_effect=load):
            self.assertEqual(dataset[0]["audio_path"], "good.wav")
            self.assertEqual(dataset[0]["audio_path"], "good.wav")

        self.assertEqual(calls.count("too-short.wav"), 1)
        self.assertEqual(calls.count("good.wav"), 2)

    def test_fails_only_after_every_sample_is_known_to_be_invalid(self) -> None:
        with patch(
            "qasr.data.load_mono_audio",
            return_value=np.zeros(10, dtype=np.float32),
        ), self.assertRaisesRegex(RuntimeError, "checking every sample"):
            self._dataset()[0]


if __name__ == "__main__":
    unittest.main()
