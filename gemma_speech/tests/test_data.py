import json
from dataclasses import replace

import pytest
import torch

from gemma_speech.config import TrainConfig
from gemma_speech.data import (
    Record,
    WeightedDistributedSampler,
    check_disjoint,
    read_manifest,
    sampling_weights,
)
from gemma_speech.metrics import score_predictions
from gemma_speech.text import normalize_text, train_tokenizer


def test_multilingual_tokenizer_roundtrips_and_keeps_diacritics(tmp_path):
    texts = ["السَّلَامُ عَلَيْكُمْ", "hello world", "नमस्ते", "你好", "നമസ്കാരം"]
    tokenizer = train_tokenizer(texts * 10, tmp_path / "speech.model", 400)
    for text in texts + ["unseen 🐪", "أإآ ة ى ؤ ئ 123"]:
        ids = tokenizer.encode(text)
        assert 0 not in ids and 1 not in ids
        assert tokenizer.decode(ids) == normalize_text(text)


def test_distributed_sampling_reconstructs_global_draw_and_resume():
    weights = torch.tensor([0.1, 0.2, 0.7], dtype=torch.double)
    rank0 = WeightedDistributedSampler(weights, 20, 2, 2, rank=0, world=2)
    rank1 = WeightedDistributedSampler(weights, 20, 2, 2, rank=1, world=2)
    global_sampler = WeightedDistributedSampler(weights, 24, 2, 2)
    combined = [value for pair in zip(list(rank0), list(rank1), strict=True) for value in pair]
    assert combined == list(global_sampler)
    original = list(rank0)
    rank0.offset = 4
    assert list(rank0) == original[4:]
    rank0.epoch = 1
    assert list(rank0) != original[4:]


def test_arabic_sampling_mass_is_independent_of_corpus_size():
    arabic = Record("a", "/a.wav", "مرحبا", "ar", "s", "مرحبا")
    english = replace(arabic, id="e", audio="/e.wav", text="hello", language="en")
    weights = sampling_weights([arabic] * 9 + [english], {"ar": 0.7, "en": 0.3})
    assert float(weights[:9].sum() / weights.sum()) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="exactly"):
        sampling_weights([arabic, english], {"ar": 1.0})


def test_manifest_relative_paths_aliases_and_leakage(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "wav_path": "x.wav",
                "transcript": "سَلَام",
                "language": "ar",
                "speaker_id": "one",
                "recording_id": "conversation1",
            }
        )
        + "\n"
    )
    row = read_manifest(path, "tts")[0]
    assert row.audio == str(tmp_path / "x.wav")
    assert row.tts_text == "سَلَام"
    with pytest.raises(ValueError, match="recording_id"):
        check_disjoint([row], [replace(row, id="other", audio="/different.wav")])


def test_scores_are_corpus_weighted_and_do_not_hide_silence_hallucination():
    rows = [
        {"id": "a", "reference": "أَهْلًا", "hypothesis": "اهلا", "language": "ar"},
        {"id": "b", "reference": "", "hypothesis": "هلوسة", "language": "ar"},
    ]
    scores = score_predictions(rows)
    assert scores["strict/all"]["wer"] == 2.0
    assert scores["arabic_normalized/all"]["wer"] == 1.0
    assert scores["strict/all"]["nonempty_on_empty"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(rows + rows)


def test_fp16_is_rejected():
    with pytest.raises(ValueError, match="fp16"):
        TrainConfig(precision="fp16")
