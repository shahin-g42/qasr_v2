import json
import random
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from qasr.augmentation import build_augmenter
from test_audio_types import artifact
from test_training import data_recipe

from audm import AUDProcessor
from audm.audio_types import upgrade
from audm.collator import AUDDataCollator
from audm.robustness import NonSpeechPool, RobustAugmenter, validate_negative_record
from audm.training_config import load_recipe


@pytest.mark.parametrize(
    "policy",
    [
        {"unknown": 1},
        {"clean_probability": float("nan")},
        {"clean_probability": True},
        {"max_effects": 1.5},
        {"max_effects": 0},
        {"effects": []},
        {"effects": {"gain": {"p": 1, "dB": [-1, 1]}}},
        {"effects": {"gain": {"p": float("inf")}}},
        {"effects": {"gain": {"p": 1, "db": "loud"}}},
        {"effects": {"gain": {"db": [2, -2]}}},
        {"effects": {"time_stretch": {"rate": [0, 1]}}},
        {"effects": {"shift": {"max_ms": float("nan")}}},
        {"effects": {"shift": {"boundary_peak": -1}}},
        {"effects": {"transcode": {"codecs": []}}},
        {"effects": {"transcode": {"codecs": "opus"}}},
        {"effects": {"transcode": {"codecs": ["made-up"]}}},
    ],
)
def test_policy_rejects_invalid_values_and_typos(policy):
    with pytest.raises(ValueError):
        RobustAugmenter(policy, {})


@pytest.mark.parametrize(
    "wave", [np.array([]), np.ones((2, 10)), np.array([np.nan]), np.array([np.inf])]
)
def test_clean_branch_also_validates_waveform(wave):
    aug = RobustAugmenter({"clean_probability": 1}, {})
    with pytest.raises(ValueError, match="waveform"):
        aug.augment_waveform(wave)


def test_policy_queue_and_effect_failures(monkeypatch):
    wave = np.ones(3200, dtype=np.float32)
    disabled = RobustAugmenter({}, {})
    features = torch.ones(1, 50, 128)
    assert disabled.augment_features(features, torch.ones(1, 50)) is features
    aug = RobustAugmenter(
        {"clean_probability": 0, "effects": {"gain": {"p": 1}}},
        {"spec_augment": {"enabled": True, "p": 1}},
    )
    with pytest.raises(ValueError, match="decision"):
        aug.augment_features(features, torch.ones(1, 50))
    monkeypatch.setattr(aug, "apply", lambda *args: np.full(3200, np.nan))
    with pytest.raises(ValueError, match="invalid waveform"):
        aug.augment_waveform(wave)
    assert not aug.clean_rows
    with pytest.raises(ValueError):
        aug.augment_waveform(wave, min_audio_seconds=1, max_audio_seconds=0.5)


def test_no_additive_effect_can_relabel_music_or_silence(monkeypatch):
    aug = RobustAugmenter({"clean_probability": 0, "max_effects": 3, "effects": {}}, {})
    aug.effects = {"noise": {"p": 1}, "white_noise": {"p": 1}, "gain": {"p": 1}}
    calls = []
    monkeypatch.setattr(aug, "apply", lambda name, wave, cfg: calls.append(name) or wave)
    for category, expected in [
        ("music", ["gain"]),
        ("silence", []),
        ("noise", ["gain", "white_noise"]),
        ("non_speech", ["gain", "white_noise"]),
    ]:
        calls.clear()
        aug.augment_waveform(np.zeros(3200), audio_type=category)
        assert calls == expected


@pytest.mark.parametrize(
    "row",
    [
        {},
        [],
        {"text": None},
        {"text": False},
        {"text": 0},
        {"text": []},
        {"text": "."},
        {"text": "", "transcript": "speech"},
        {"text": "", "audio_type": "speech"},
        {"text": "", "audio_type": "unknown"},
    ],
)
def test_negative_labels_never_silently_discard_content(row):
    with pytest.raises(ValueError):
        validate_negative_record(row)


def test_negative_metadata_filtering_sampling_and_decoded_duration(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(3200), 16000)
    manifest = tmp_path / "negative.jsonl"
    rows = [
        dict(audio_filepath="clip.wav", text="", duration=40, audio_type="music"),
        dict(audio_filepath="clip.wav", text="", duration=0.2, audio_type="silence"),
    ]
    manifest.write_text("\n" + "\n\n".join(map(json.dumps, rows)) + "\n")
    cfg = dict(probability=0.03, manifests=[str(manifest)])
    pool = NonSpeechPool(cfg, 0.1, 30, require_audio_type=True)
    assert len(pool.parts[0]) == 1
    assert pool.scan_audio()["categories"] == {"silence": 1}
    original = [{"text": "hello", "language": "ml"} for _ in range(2000)]
    random.seed(42)
    replaced = pool.replace(original)
    assert 30 < sum(not r["text"] for r in replaced) < 90
    assert all(r["text"] == "hello" for r in original)
    assert all(r["language"] == "ml" for r in replaced)
    monkeypatch.setattr(random, "random", lambda: 0)
    assert pool.replace([{"text": "hello"}])[0]["language"] is None
    sf.write(wav, np.zeros(160), 16000)  # Metadata claimed 0.2s; bytes actually 0.01s.
    with pytest.raises(ValueError, match="shorter"):
        pool.scan_audio()
    with pytest.raises(ValueError, match="Duplicate"):
        NonSpeechPool(dict(cfg, manifests=[str(manifest)] * 2), 0.1, 30)


def test_rir_scan_and_full_tail_duration_guard(tmp_path):
    path = tmp_path / "rir.wav"
    rir = np.zeros(1000, dtype=np.float32)
    rir[100], rir[900] = 1, 0.5
    sf.write(path, rir, 16000, subtype="FLOAT")
    manifest = tmp_path / "rirs.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "rir.wav"}) + "\n")
    aug = RobustAugmenter(
        dict(clean_probability=0, effects={"rir": {"p": 1, "manifest": str(manifest)}}), {}
    )
    assert aug.scan_assets()["rirs"] == 1
    wave = np.ones(3200, dtype=np.float32)
    result = aug.augment_waveform(wave, max_audio_seconds=1)
    assert len(result) == len(wave) + 899
    np.testing.assert_allclose(np.sqrt(np.mean(result**2)), 1, rtol=1e-5)
    np.testing.assert_array_equal(aug.augment_waveform(wave, max_audio_seconds=0.2), wave)
    assert aug.stats["rir:duration_skip"] == 1
    sf.write(path, np.zeros(1000), 16000)
    with pytest.raises(ValueError, match="zero RIR"):
        aug.scan_assets()


def test_codec_encoder_detection_and_subprocess_failure(monkeypatch):
    monkeypatch.setattr("audm.robustness.shutil.which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, " A..... libfdk_aac AAC encoder\n"),
    )
    with pytest.raises(RuntimeError, match="unavailable: aac"):
        RobustAugmenter({"effects": {"transcode": {"p": 1, "codecs": ["aac"]}}}, {})
    aug = RobustAugmenter({}, {})

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        aug.transcode(np.zeros(3200), {"codecs": ["opus"]})


def test_asset_paths_resolve_beside_recipe_and_environment_is_required(tmp_path, monkeypatch):
    manifest = tmp_path / "negative.jsonl"
    manifest.write_text("{}\n")
    recipe = tmp_path / "recipe.yaml"
    data = dict(
        data_recipe=str(data_recipe(tmp_path)),
        stage="decoder",
        training={},
        robustness=dict(non_speech=dict(probability=0.03, manifests=["negative.jsonl"])),
    )
    recipe.write_text(yaml.safe_dump(data))
    cfg, _, receipt = load_recipe(recipe, model="/tmp/model", output="/tmp/run")
    assert cfg.aud_robustness["non_speech"]["manifests"] == [str(manifest)]
    assert str(manifest) in receipt["robustness_assets"]
    monkeypatch.delenv("AUD_MISSING_ASSET_TEST", raising=False)
    data["robustness"]["non_speech"]["manifests"] = ["${AUD_MISSING_ASSET_TEST}/negative.jsonl"]
    recipe.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Unset environment"):
        load_recipe(recipe, model="/tmp/model", output="/tmp/run")


@pytest.mark.parametrize("variant", ["audpro", "audturbo", "audflash"])
@pytest.mark.parametrize("typed", [False, True])
def test_negative_collation_augmentation_and_codeswitch(tmp_path, tokenizer, variant, typed):
    source = artifact(tmp_path, tokenizer, variant)
    if typed:
        upgrade(source, tmp_path / "typed")
        source = tmp_path / "typed"
    proc = AUDProcessor.from_pretrained(source)
    wave = np.zeros(3200, dtype=np.float32)
    records = [
        dict(audio_path="memory.wav", waveform=wave, text="", language="en", audio_type="silence")
    ]
    calls = []

    class Mixer:
        def augment(self, f, p):
            calls.append(f)
            raise AssertionError("Never code-switch non-speech")

    opts = dict(
        processor=proc,
        language="en",
        min_audio_seconds=0.1,
        max_audio_seconds=30,
        max_target_length=512,
    )
    clean = AUDDataCollator(**opts)(records)
    for aug in [
        build_augmenter({"spec_augment": {"enabled": True, "p": 1}}, 16000),
        RobustAugmenter(
            {"clean_probability": 0, "effects": {"white_noise": {"p": 1}}},
            {"spec_augment": {"enabled": True, "p": 1}},
        ),
    ]:
        collator = AUDDataCollator(**opts, augmenter=aug, code_switcher=Mixer())
        batch = collator(records)
        assert not calls
        torch.testing.assert_close(batch["input_features"], clean["input_features"], atol=0, rtol=0)
        torch.testing.assert_close(batch["input_features_mask"], clean["input_features_mask"])
        assert batch["labels"].ne(-100).sum() == (2 if typed else clean["labels"].ne(-100).sum())
        assert not getattr(aug, "clean_rows", [])
    if typed:
        ids = [proc.config.audio_type_token_ids["noise"], proc.config.eos_token_id]
        assert proc.decode_audio_event(ids + [proc.tokenizer.pad_token_id])["format_valid"]
        invalid = proc.decode_audio_event(ids + tokenizer.encode("hello"))
        assert not invalid["format_valid"] and "hello" in invalid["text"]
