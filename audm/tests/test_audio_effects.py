import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from audm import audio_effects
from audm.robustness import RobustAugmenter


def tone(seconds=1):
    t = np.arange(round(16000 * seconds)) / 16000
    return (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def policy(name, **settings):
    return RobustAugmenter(
        {"clean_probability": 0, "effects": {name: {"p": 1, **settings}}}, {}
    )


@pytest.mark.parametrize("category", ["speech", "music", "noise", "non_speech", None])
def test_silence_preserves_every_input_sample(category):
    original = np.linspace(-0.9, 0.9, 3200, dtype=np.float32)
    aug = policy("silence", start_ms=[100, 100], end_ms=[50, 50])
    result = aug.augment_waveform(original, max_audio_seconds=30, audio_type=category)
    assert result.dtype == np.float32 and len(result) == 5600
    assert not result[:1600].any() and not result[-800:].any()
    np.testing.assert_array_equal(result[1600:-800], original)
    assert aug.stats["silence:changed"] == 1


@pytest.mark.parametrize("budget", [0, 1, 79, 160, 3200])
def test_silence_reduces_only_padding_to_fit_duration_budget(budget):
    original = np.ones(480000 - budget, dtype=np.float32)
    aug = policy("silence", start_ms=[100, 100], end_ms=[100, 100])
    result = aug.augment_waveform(original, max_audio_seconds=30)
    assert len(result) == 480000
    start = budget // 2
    np.testing.assert_array_equal(result[start : start + len(original)], original)
    assert np.count_nonzero(result) == len(original)
    assert aug.stats["silence:changed"] == int(budget > 0)
    assert not aug.stats["silence:duration_skip"]


def test_zero_padding_and_typed_silence_are_identity():
    wave = tone(0.1)
    aug = policy("silence", start_ms=[0, 0], end_ms=[0, 0])
    np.testing.assert_array_equal(aug.augment_waveform(wave), wave)
    aug = policy("silence", start_ms=[100, 100], end_ms=[100, 100])
    zero = np.zeros(1600, dtype=np.float32)
    np.testing.assert_array_equal(aug.augment_waveform(zero, audio_type="silence"), zero)
    assert aug.stats["clean"] == 1 and not aug.stats["silence:attempted"]


def test_pitch_changes_frequency_without_changing_duration():
    original = tone()
    result = policy("pitch", semitones=[1, 1]).augment_waveform(original)
    assert result.shape == original.shape and result.dtype == np.float32
    assert np.isfinite(result).all()
    frequency = np.fft.rfftfreq(len(result), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(result)))]
    assert abs(frequency - 440 * 2 ** (1 / 12)) < 2
    identity = policy("pitch", semitones=[0, 0]).augment_waveform(original)
    np.testing.assert_array_equal(identity, original)


def test_clipping_has_no_makeup_gain_and_preserves_shape():
    original = tone(0.1)
    aug = policy("clipping", peak_fraction=[0.7, 0.7])
    result = aug.augment_waveform(original)
    assert result.shape == original.shape
    assert np.abs(result).max() == pytest.approx(np.abs(original).max() * 0.7)
    unchanged = np.abs(original) < 0.5
    np.testing.assert_array_equal(result[unchanged], original[unchanged])
    zero = np.zeros(1600, dtype=np.float32)
    np.testing.assert_array_equal(aug.augment_waveform(zero), zero)


@pytest.mark.parametrize("length,fraction", [(1, 0.01), (160, 0.01), (16000, 0.01), (16000, 0)])
def test_dropouts_are_bounded_and_do_not_modify_source(length, fraction):
    original = np.ones(length, dtype=np.float32)
    aug = policy("dropout", duration_ms=[20, 20], max_drops=2, max_fraction=fraction)
    result = aug.augment_waveform(original)
    assert result.shape == original.shape and np.isfinite(result).all()
    assert np.count_nonzero(result == 0) <= int(length * fraction)
    if int(length * fraction):
        assert np.count_nonzero(result == 0) > 0
    np.testing.assert_array_equal(original, np.ones(length))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
def test_actual_compression_attenuates_loud_audio_without_changing_duration():
    original = tone()
    aug = policy("compression", threshold_db=[-24, -24], ratio=[4, 4])
    result = aug.augment_waveform(original)
    assert result.shape == original.shape and result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.sqrt(np.mean(result[1600:] ** 2)) < np.sqrt(np.mean(original[1600:] ** 2)) * 0.5
    assert aug.scan_assets()["extra_effects_checked"] == ["compression"]


def test_compression_missing_binary_fails_only_when_enabled(monkeypatch):
    monkeypatch.setattr("audm.robustness.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Compression requires ffmpeg"):
        policy("compression")
    RobustAugmenter({"effects": {"compression": {"p": 0}}}, {})


@pytest.mark.parametrize(
    "name,settings",
    [
        ("silence", {"start_ms": [-1, 100]}),
        ("silence", {"end_ms": [100, 0]}),
        ("silence", {"start_ms": [0, float("nan")]}),
        ("silence", {"start_ms": [False, 100]}),
        ("silence", {"end_ms": [0, 1001]}),
        ("pitch", {"semitones": [0, 5]}),
        ("pitch", {"semitones": "1"}),
        ("clipping", {"peak_fraction": [0, 1]}),
        ("dropout", {"max_drops": True}),
        ("dropout", {"max_fraction": 0.1}),
        ("dropout", {"duration_ms": [0, 20]}),
        ("compression", {"ratio": [0, 4]}),
        ("compression", {"attack_ms": float("inf")}),
        ("compression", {"release_ms": 0}),
        ("compression", {"threshold_db": [-100, -12]}),
    ],
)
def test_invalid_settings_fail_even_when_disabled(name, settings):
    with pytest.raises(ValueError):
        RobustAugmenter({"effects": {name: {"p": 0, **settings}}}, {})


def test_exclusive_effects_and_cap(monkeypatch):
    effects = {key: {"p": 1} for key in ("speed", "time_stretch", "pitch", "shift", "silence")}
    effects["gain"] = {"p": 1, "db": [-3, -3]}
    aug = RobustAugmenter(
        {"clean_probability": 0, "max_effects": 2, "effects": effects},
        {"speed_perturb": {"enabled": True}},
    )
    for _ in range(30):
        calls = []
        monkeypatch.setattr(aug, "apply", lambda name, wave, cfg: calls.append(name) or wave)
        monkeypatch.setattr(audio_effects, "apply", lambda name, wave, *args: calls.append(name) or wave)
        aug.augment_waveform(tone(0.1))
        assert len(calls) == 2
        assert len(set(calls) & {"speed", "time_stretch", "pitch"}) <= 1
        assert len(set(calls) & {"shift", "silence"}) <= 1


def test_silence_after_time_stretch_uses_remaining_budget(monkeypatch):
    aug = RobustAugmenter(
        {"clean_probability": 0, "max_effects": 2, "effects": {
            "time_stretch": {"p": 1, "rate": [0.9, 0.9]},
            "silence": {"p": 1, "start_ms": [100, 100], "end_ms": [100, 100]},
        }}, {}
    )
    stretched = np.ones(15900, dtype=np.float32)
    monkeypatch.setattr(aug, "apply", lambda *args: stretched)
    result = aug.augment_waveform(tone(0.9), max_audio_seconds=1)
    assert len(result) == 16000
    np.testing.assert_array_equal(result[50:-50], stretched)


def test_effective_policy_and_clean_branch_counts_are_logged(caplog):
    with caplog.at_level(logging.INFO, logger="audm.robustness"):
        aug = RobustAugmenter({"clean_probability": 1, "effects": {"shift": {"p": 0.15}}}, {})
        aug.stats["samples"] = 999
        aug.augment_waveform(tone(0.1))
    assert "AUD waveform policy" in caplog.text and '"shift": {"p": 0.15}' in caplog.text
    assert "Worker-local augmentation counts" in caplog.text and "1000" in caplog.text


def test_enabled_extras_are_scanned_without_altering_counters():
    aug = policy("silence")
    assert aug.scan_assets()["extra_effects_checked"] == ["silence"]
    assert not aug.stats


def test_padded_gemma_batch_keeps_targets_lengths_and_gradients_finite(tmp_path, tokenizer):
    import torch
    from test_audio_types import artifact

    from audm import AUDForConditionalGeneration, AUDProcessor
    from audm.audio_types import upgrade
    from audm.collator import AUDDataCollator

    source = artifact(tmp_path, tokenizer, "audflash")
    upgrade(source, tmp_path / "typed")
    processor = AUDProcessor.from_pretrained(tmp_path / "typed")
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "typed")
    model.configure_training("full")
    model.train()
    augmenter = RobustAugmenter(
        {"clean_probability": 0, "effects": {
            "silence": {"p": 1, "start_ms": [100, 100], "end_ms": [100, 100]},
        }},
        {"spec_augment": {"enabled": True, "p": 1}},
    )
    common = dict(
        processor=processor, language="en", max_target_length=512,
        min_audio_seconds=0.1, max_audio_seconds=30,
    )
    clean = AUDDataCollator(**common)
    augmented = AUDDataCollator(**common, augmenter=augmenter)
    records = [
        dict(audio_path="unused-speech.wav", waveform=tone(), text="hello world",
             language="en", audio_type="speech"),
        dict(audio_path="unused-silence.wav", waveform=np.zeros(3200, dtype=np.float32),
             text="", language="en", audio_type="silence"),
    ]
    baseline, batch = clean(records), augmented(records)
    before = baseline["input_features_mask"].sum(-1)
    after = batch["input_features_mask"].sum(-1)
    assert after[0] > before[0] and after[1] == before[1]
    for old, new in zip(baseline["labels"], batch["labels"], strict=True):
        torch.testing.assert_close(old[old != -100], new[new != -100])
    assert not augmenter.clean_rows  # Waveform and SpecAugment decisions stay synchronized.
    loss = model(**batch).loss
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


def test_shipped_recipes_enable_silence_and_keep_distortions_opt_in():
    root = Path(__file__).resolve().parents[1] / "configs"
    for filename in ("audflash_audio_types_decoder.yaml", "audflash_robust_full.yaml"):
        recipe = yaml.safe_load((root / filename).read_text())
        effects = recipe["robustness"]["augmentation"]["effects"]
        assert effects["silence"] == {"p": 0.2, "start_ms": [0, 100], "end_ms": [0, 100]}
        for name in ("pitch", "dropout", "clipping", "compression"):
            assert effects[name]["p"] == 0
            audio_effects.validate(name, effects[name])
