"""Tests for the per-language WER/CER evaluation callback.

Previously untested — and broken twice over: transformers' CallbackHandler
never passes a ``trainer`` kwarg, so the decode pass silently never ran, and
``random.Random((seed, language))`` raises TypeError on the tuple seed. These
pin the bind_trainer contract and the deterministic sampling fix.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from torch import nn

from qasr.eval_callback import WEREvalCallback


def _write_manifest(directory: Path, language: str, count: int = 4) -> str:
    path = directory / f"eval_{language}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": f"/nonexistent/{language}_{index}.wav",
                        "text": f"reference text {index}",
                        "duration": 2.0,
                    }
                )
                + "\n"
            )
    return str(path)


class _FakeTrainer:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 2)
        self.logged: list[dict] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)


def _callback(manifest: str, language: str) -> WEREvalCallback:
    return WEREvalCallback(
        processor=SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=16000)
        ),
        eval_manifest_specs=[(manifest, language)],
        samples_per_language=2,
        seed=42,
        dataset_kwargs={"min_duration_seconds": 0.1, "max_duration_seconds": 35.0},
    )


_STATE = SimpleNamespace(is_world_process_zero=True)


class WEREvalCallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manifest = _write_manifest(Path(self._tmp.name), "ar")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bound_trainer_fires_without_kwarg(self) -> None:
        """The handler passes no trainer kwarg; the bound reference must be
        used — this is the regression that silently disabled the callback."""
        trainer = _FakeTrainer()
        callback = _callback(self.manifest, "ar").bind_trainer(trainer)
        callback._transcribe = lambda t, feature: feature["text"]  # perfect ASR

        callback.on_evaluate(args=None, state=_STATE, control=None)

        self.assertEqual(len(trainer.logged), 1)
        metrics = trainer.logged[0]
        self.assertEqual(metrics["eval_wer_ar"], 0.0)
        self.assertEqual(metrics["eval_cer_ar"], 0.0)
        self.assertEqual(metrics["eval_wer_macro"], 0.0)

    def test_unbound_callback_warns_and_survives(self) -> None:
        callback = _callback(self.manifest, "ar")
        with self.assertLogs("qasr.eval_callback", level="WARNING"):
            callback.on_evaluate(args=None, state=_STATE, control=None)

    def test_sample_indices_deterministic_and_no_tuple_seed_crash(self) -> None:
        """random.Random rejects tuple seeds; the crc32 mix must be stable
        across instances so every evaluation decodes the same utterances."""
        first = _callback(self.manifest, "ar")
        second = _callback(self.manifest, "ar")
        dataset = first._dataset_for(self.manifest, "ar")

        picks_a = first._sample_indices(dataset, "ar")
        picks_b = second._sample_indices(second._dataset_for(self.manifest, "ar"), "ar")

        self.assertEqual(picks_a, picks_b)
        self.assertEqual(len(picks_a), 2)
        self.assertNotEqual(
            picks_a, first._sample_indices(dataset, "en"),
            "different languages should sample different indices",
        )

    def test_imperfect_hypothesis_produces_nonzero_wer(self) -> None:
        trainer = _FakeTrainer()
        callback = _callback(self.manifest, "ar").bind_trainer(trainer)
        callback._transcribe = lambda t, feature: "completely wrong words here"

        callback.on_evaluate(args=None, state=_STATE, control=None)

        self.assertGreater(trainer.logged[0]["eval_wer_ar"], 0.0)


if __name__ == "__main__":
    unittest.main()
