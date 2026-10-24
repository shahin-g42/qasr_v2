from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_audio_types import artifact

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.audio_types import upgrade
from audm.eval_callback import AUDWEREvalCallback


@pytest.mark.parametrize("variant", ["audflash", "audpro", "audturbo"])
def test_batched_generation_matches_individual_unequal_length_audio(tmp_path, tokenizer, variant):
    torch.manual_seed(42)
    source = artifact(tmp_path, tokenizer, variant)
    upgrade(source, tmp_path / "typed")
    processor = AUDProcessor.from_pretrained(tmp_path / "typed")
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "typed").eval()
    trainer = SimpleNamespace(model=model, _prepare_inputs=lambda batch: batch)
    callback = AUDWEREvalCallback(
        processor=processor, eval_manifest_specs=[], batch_size=40, max_new_tokens=6
    )
    waves = [
        np.random.default_rng(i).normal(0, 0.01, n).astype(np.float32)
        for i, n in enumerate([3200, 7200, 4800])
    ]
    features = [dict(language="en") for _ in waves]
    with torch.inference_mode():
        batched = callback._transcribe_batch(trainer, features, waves)
        individual = [
            callback._transcribe_batch(trainer, [f], [w])[0]
            for f, w in zip(features, waves, strict=True)
        ]
    assert batched == individual


def test_full_batches_tail_and_one_bad_file_do_not_change_metrics(monkeypatch):
    processor = SimpleNamespace(_waves=lambda waves, sr: waves)
    callback = AUDWEREvalCallback(
        processor=processor, eval_manifest_specs=[], batch_size=40, samples_per_language=100
    )
    dataset = [dict(audio_path=str(i), text=f"example {i}", language="en") for i in range(43)]
    callback._dataset_for = lambda *args: dataset

    def load(path, sr):
        if path == "17":
            raise OSError("bad file")
        return np.zeros(1600)

    monkeypatch.setattr("audm.eval_callback.load_mono_audio", load)
    batches, seen = [], []

    def transcribe(trainer, features, waves):
        batches.append(len(features))
        seen.extend(f["audio_path"] for f in features)
        return [f["text"] for f in features]

    callback._transcribe_batch = transcribe
    result = callback._evaluate_language(None, "en", [("manifest", "en")])
    assert batches == [40, 2]
    assert result == dict(wer=0, cer=0, samples=42)
    assert set(seen) == {str(i) for i in range(43)} - {"17"}


def test_model_failure_is_not_silently_excluded(monkeypatch):
    callback = AUDWEREvalCallback(
        processor=SimpleNamespace(_waves=lambda waves, sr: waves),
        eval_manifest_specs=[],
        batch_size=40,
    )
    callback._dataset_for = lambda *args: [dict(audio_path="a", text="hello", language="en")]
    monkeypatch.setattr("audm.eval_callback.load_mono_audio", lambda *args: np.zeros(1600))

    def fail(*args):
        raise torch.OutOfMemoryError("test")

    callback._transcribe_batch = fail
    with pytest.raises(torch.OutOfMemoryError):
        callback._evaluate_language(None, "en", [("manifest", "en")])
