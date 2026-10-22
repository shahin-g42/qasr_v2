import json
import random
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch
from test_models import encoder_source, gqasr_source

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.collator import AUDDataCollator
from audm.eval_non_speech import evaluate
from audm.export import export_model
from audm.robustness import NonSpeechPool, RobustAugmenter


def wave():
    t = np.arange(8000) / 16000
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.mark.parametrize(
    "effect,settings",
    [
        ("gain", {"db": [-6, -6]}),
        ("white_noise", {"db": [-46, -46]}),
        ("time_stretch", {"rate": [0.9, 0.9]}),
    ],
)
def test_effects_finite_and_effective(effect, settings):
    aug = RobustAugmenter(dict(clean_probability=0, effects={effect: dict(p=1, **settings)}), {})
    result = aug.augment_waveform(wave(), max_audio_seconds=1)
    assert np.isfinite(result).all() and result.dtype == np.float32
    assert len(result) != len(wave()) or not np.array_equal(result, wave())
    assert aug.stats[f"{effect}:changed"] == 1
    assert not aug.clean_rows  # no growing queue when SpecAugment is disabled


def test_clean_branch_and_specaugment():
    aug = RobustAugmenter(
        dict(clean_probability=1, effects={"gain": {"p": 1}}),
        {"spec_augment": {"enabled": True, "p": 1}},
    )
    result = aug.augment_waveform(wave())
    assert np.array_equal(result, wave())
    features = torch.ones(1, 50, 128)
    assert torch.equal(aug.augment_features(features, torch.ones(1, 50)), features)
    assert not aug.clean_rows


def test_boundary_shift_does_not_remove_speech(monkeypatch):
    aug = RobustAugmenter(dict(effects={}), {})
    monkeypatch.setattr(random, "uniform", lambda *args: 5)
    original = np.ones(3200, dtype=np.float32)
    assert np.array_equal(aug.apply("shift", original, {}), original)
    padded = np.pad(original, (0, 80))
    result = aug.apply("shift", padded, {})
    assert len(result) == len(padded)
    assert result[:80].sum() == 0
    assert result.sum() == padded.sum()


def test_duration_budget_and_exclusive_effects(monkeypatch):
    aug = RobustAugmenter(
        dict(
            clean_probability=0,
            max_effects=2,
            effects={"time_stretch": {"p": 1, "rate": [0.5, 0.5]}},
        ),
        {},
    )
    assert np.array_equal(aug.augment_waveform(wave(), max_audio_seconds=0.6), wave())
    assert aug.stats["time_stretch:duration_skip"] == 1
    aug = RobustAugmenter(
        dict(
            clean_probability=0,
            max_effects=2,
            effects={"speed": {"p": 1}, "time_stretch": {"p": 1}, "gain": {"p": 1}},
        ),
        {"speed_perturb": {"enabled": True}},
    )
    calls = []
    monkeypatch.setattr(aug, "apply", lambda name, wave, cfg: calls.append(name) or wave)
    aug.augment_waveform(wave())
    assert len(calls) <= 2
    assert not {"speed", "time_stretch"}.issubset(calls)


def test_rir_full_tail_and_invalid_assets(tmp_path):
    rir = np.zeros(1000, dtype=np.float32)
    rir[0], rir[500] = 1, 0.5
    sf.write(tmp_path / "rir.wav", rir, 16000, subtype="FLOAT")
    manifest = tmp_path / "rir.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "rir.wav"}) + "\n")
    aug = RobustAugmenter(
        dict(clean_probability=0, effects={"rir": {"p": 1, "manifest": str(manifest)}}), {}
    )
    result = aug.augment_waveform(wave(), max_audio_seconds=1)
    assert len(result) == len(wave()) + len(rir) - 1
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="nonempty"):
        RobustAugmenter(
            dict(effects={"noise": {"p": 1}}),
            {"noise_injection": {"enabled": True, "noise_dir": str(tmp_path / "missing")}},
        )


@pytest.mark.parametrize("codec", ["opus", "aac", "mp3", "g711"])
def test_actual_codec_round_trip(codec):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    aug = RobustAugmenter(
        dict(clean_probability=0, effects={"transcode": {"p": 1, "codecs": [codec]}}), {}
    )
    result = aug.augment_waveform(wave(), max_audio_seconds=2)
    assert result.ndim == 1 and len(result) > 0 and np.isfinite(result).all()
    assert result.shape != wave().shape or not np.array_equal(result, wave())
    assert aug.stats[f"codec:{codec}"] == 1


def non_speech(tmp_path):
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(3200, dtype=np.float32), 16000)
    manifest = tmp_path / "negative.jsonl"
    manifest.write_text(json.dumps(dict(audio_filepath=str(audio), duration=0.2, text="")) + "\n")
    return manifest


def test_non_speech_supervision_and_evaluation(tmp_path, tokenizer, monkeypatch):
    manifest = non_speech(tmp_path)
    pool = NonSpeechPool(dict(probability=0.03, manifests=[str(manifest)]), 0.1, 30)
    monkeypatch.setattr(random, "random", lambda: 0)
    records = pool.replace([dict(language="ml", text="hello", waveform=wave())] * 3)
    assert len(records) == 3 and all(r["text"] == "" and r["language"] == "ml" for r in records)
    gqasr_source(tmp_path / "text", tokenizer)
    encoder_source(tmp_path / "encoder", "audflash_gemma")
    export_model(
        "audflash",
        tmp_path / "aud",
        gqasr_checkpoint=tmp_path / "text",
        encoder_source=tmp_path / "encoder",
        dtype="float32",
    )
    proc = AUDProcessor.from_pretrained(tmp_path / "aud")
    collator = AUDDataCollator(
        processor=proc,
        language="en",
        min_audio_seconds=0.1,
        max_audio_seconds=30,
        max_target_length=512,
    )
    collator.non_speech_pool = pool
    batch = collator([dict(language="en", text="hello", waveform=wave())])
    labels = batch["labels"][0]
    supervised = labels[labels != -100]
    assert len(supervised) > 1 and supervised[-1] == proc.config.eos_token_id
    assert "hello" not in tokenizer.decode(supervised)
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "aud")
    model.configure_training("adapter")
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.aud_projector.parameters()
    )
    result = evaluate(
        tmp_path / "aud", [str(manifest)], tmp_path / "eval.json", max_new_tokens=2, limit=1
    )
    assert result["samples"] == 1 and 0 <= result["nonempty_transcript_rate"] <= 1
    manifest.write_text(manifest.read_text().replace('"text": ""', '"text": "speech"'))
    with pytest.raises(ValueError, match="empty"):
        NonSpeechPool(dict(probability=0.03, manifests=[str(manifest)]), 0.1, 30)


def test_opt_in_trainer_and_asset_receipts(tmp_path, tokenizer):
    import yaml
    from test_training import data_recipe

    from audm.train import preflight, run
    from audm.training_config import load_recipe

    manifest = non_speech(tmp_path)
    gqasr_source(tmp_path / "text", tokenizer)
    encoder_source(tmp_path / "encoder", "audflash_gemma")
    export_model(
        "audflash",
        tmp_path / "aud",
        gqasr_checkpoint=tmp_path / "text",
        encoder_source=tmp_path / "encoder",
        dtype="float32",
    )
    recipe = tmp_path / "robust.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage="adapter",
                robustness=dict(
                    non_speech=dict(probability=0.5, manifests=[str(manifest)]),
                    augmentation=dict(
                        clean_probability=0.2,
                        max_effects=1,
                        effects=dict(gain=dict(p=1, db=[-3, 3])),
                    ),
                ),
                training=dict(
                    max_duration_seconds=30,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    per_device_eval_batch_size=1,
                    expected_world_size=1,
                    max_steps=2,
                    save_steps=1,
                    eval_steps=2,
                    report_to="none",
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                    bf16=False,
                    tf32=False,
                    deepspeed=None,
                    gradient_checkpointing=True,
                    eval_log_samples=0,
                    eval_wer_samples_per_language=0,
                ),
            )
        )
    )
    report = preflight(recipe, tmp_path / "aud", tmp_path / "run", scan=True)
    assert report["recipe"]["robustness_assets"]
    run(recipe, tmp_path / "aud", tmp_path / "run")
    assert (tmp_path / "run/model.safetensors").is_file()
    _, _, before = load_recipe(recipe, tmp_path / "aud", tmp_path / "run")
    manifest.write_text(manifest.read_text() + "\n")
    _, _, after = load_recipe(recipe, tmp_path / "aud", tmp_path / "run")
    assert before["robustness_assets"] != after["robustness_assets"]
