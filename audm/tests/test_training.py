import json
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from qasr.augmentation import build_augmenter
from qasr.sampling import WeightedStratifiedLanguageSampler
from test_models import encoder_source, gqasr_source, qasr_source

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.collator import AUDDataCollator, spectral_mask_waveform
from audm.export import export_model
from audm.train import preflight, run
from audm.training_config import load_recipe


def data_recipe(tmp_path):
    train, evaluation = {}, {}
    for language in ("ar", "en"):
        lines = []
        for i in range(2):
            wav = tmp_path / f"{language}-{i}.wav"
            sf.write(wav, np.random.default_rng(i).normal(0, 0.01, 3200), 16000)
            lines.append(
                json.dumps({"audio_filepath": str(wav), "text": "hello world", "duration": 0.2})
            )
        manifest = tmp_path / f"train_{language}.jsonl"
        manifest.write_text("\n".join(lines) + "\n")
        train[language] = str(manifest)
        evaluation[language] = str(manifest)
    source = dict(
        train_manifest=train,
        eval_manifest=evaluation,
        min_duration_seconds=0.1,
        max_duration_seconds=35.0,
        max_target_length=64,
        language="ar",
        balanced_sampling=True,
        sampling_strategy="weighted_stratified",
        language_batch_counts={"ar": 1, "en": 1},
        code_switch_index=None,
        augmentation=None,
    )
    path = tmp_path / "stage02.yaml"
    path.write_text(yaml.safe_dump(source))
    return path


@pytest.mark.parametrize("variant", ["audpro", "audturbo", "audflash"])
def test_two_updates_checkpoint_resume_and_clean_eval(tmp_path, tokenizer, variant):
    source = tmp_path / "source"
    if variant == "audflash":
        gqasr_source(source, tokenizer)
        kwargs = dict(gqasr_checkpoint=source)
    else:
        qasr_source(source, tokenizer)
        kwargs = dict(qasr_checkpoint=source)
    if variant != "audpro":
        encoder_source(tmp_path / "encoder", variant)
        kwargs["encoder_source"] = tmp_path / "encoder"
    export_model(variant, tmp_path / "aud", dtype="float32", **kwargs)
    stage02 = data_recipe(tmp_path)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(stage02),
                stage="full",
                training=dict(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    per_device_eval_batch_size=1,
                    expected_world_size=1,
                    num_train_epochs=1,
                    max_steps=-1,
                    learning_rate=1e-4,
                    warmup_steps=0,
                    save_steps=1,
                    eval_steps=2,
                    logging_steps=1,
                    report_to="none",
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                    bf16=False,
                    tf32=False,
                    deepspeed=None,
                    gradient_checkpointing=True,
                    label_smoothing=0.1,
                    eval_log_samples=0,
                    eval_wer_samples_per_language=0,
                ),
            )
        )
    )
    report = preflight(recipe, tmp_path / "aud", tmp_path / "run", scan=True)
    assert report["updates_per_coverage_epoch"] == 2
    run(recipe, model=tmp_path / "aud", output=tmp_path / "run")
    final = AUDForConditionalGeneration.from_pretrained(tmp_path / "run", local_files_only=True)
    reference = {k: v.clone() for k, v in final.state_dict().items()}
    assert Path(tmp_path / "run/checkpoint-1/optimizer.pt").is_file()
    assert Path(tmp_path / "run/checkpoint-1/preprocessor_config.json").is_file()
    run(
        recipe,
        model=tmp_path / "aud",
        output=tmp_path / "run",
        resume=str(tmp_path / "run/checkpoint-1"),
    )
    resumed = AUDForConditionalGeneration.from_pretrained(tmp_path / "run", local_files_only=True)
    for key in reference:
        torch.testing.assert_close(resumed.state_dict()[key], reference[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError, match="not empty"):
        run(recipe, model=tmp_path / "aud", output=tmp_path / "run")


def test_accumulation_window_preserves_96_rank_coverage():
    parts = []
    quotas = {"ar": 10, "en": 6, "hi": 2, "ml": 1, "zh": 1}

    class Part:
        def __init__(self, language, size):
            self.language, self.size = language, size

        def __len__(self):
            return self.size

    parts = [Part(lang, quota * 96 + 3) for lang, quota in quotas.items()]
    ends = np.cumsum([len(p) for p in parts])
    seen = set()
    for rank in range(96):
        sampler = WeightedStratifiedLanguageSampler(
            parts, batch_size=20, seed=42, num_replicas=96, rank=rank, language_batch_counts=quotas
        )
        indices = list(sampler)
        seen.update(indices)
        for offset in range(0, len(indices), 20):
            window = indices[offset : offset + 20]
            actual = {p.language: 0 for p in parts}
            for index in window:
                actual[parts[int(np.searchsorted(ends, index, side="right"))].language] += 1
            assert actual == quotas
    assert seen == set(range(sum(map(len, parts))))


def test_production_recipes_reuse_stage02_data_and_augmentation():
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root.parent / "configs/v8.1/02_full_12node.yaml").read_text())
    for path in (root / "configs").glob("*_12node.yaml"):
        config, stage, receipt = load_recipe(path, model="/tmp/export", output="/tmp/run")
        assert config.train_manifest == original["train_manifest"]
        assert config.eval_manifest == original["eval_manifest"]
        assert config.augmentation == original["augmentation"]
        assert config.code_switch_index == original["code_switch_index"]
        assert config.code_switch_probability == original["code_switch_probability"]
        assert config.expected_world_size == 96
        assert config.per_device_train_batch_size * config.gradient_accumulation_steps * 96 == 1920
        assert receipt["quota_window"] == 20


@pytest.mark.parametrize("variant", ["audpro", "audturbo", "audflash"])
def test_augmentation_and_codeswitch_use_correct_feature_axis(tmp_path, tokenizer, variant):
    if variant == "audflash":
        gqasr_source(tmp_path / "text", tokenizer)
        kwargs = dict(gqasr_checkpoint=tmp_path / "text")
    else:
        qasr_source(tmp_path / "text", tokenizer)
        kwargs = dict(qasr_checkpoint=tmp_path / "text")
    if variant != "audpro":
        encoder_source(tmp_path / "encoder", variant)
        kwargs["encoder_source"] = tmp_path / "encoder"
    export_model(variant, tmp_path / "aud", dtype="float32", **kwargs)
    proc = AUDProcessor.from_pretrained(tmp_path / "aud")
    wave = np.random.default_rng(12).normal(0, 0.05, 3200).astype(np.float32)
    records = [{"audio_path": "in-memory.wav", "waveform": wave, "text": "hello", "language": "en"}]
    calls = []

    class Mixer:
        def augment(self, record, probability):
            calls.append(probability)
            return dict(record, text="hello world"), {"mixed": True}

    augmenter = build_augmenter(
        {
            "spec_augment": {
                "enabled": True,
                "p": 1.0,
                "num_time_masks": 2,
                "num_freq_masks": 2,
                "max_time_mask_ratio": 0.2,
                "max_freq_mask_ratio": 0.3,
            }
        },
        sampling_rate=16000,
    )
    kwargs = dict(
        processor=proc,
        language="en",
        min_audio_seconds=0.1,
        max_audio_seconds=35,
        max_target_length=64,
    )
    collator = AUDDataCollator(
        **kwargs, augmenter=augmenter, code_switcher=Mixer(), code_switch_probability=0.4
    )
    clean = AUDDataCollator(**kwargs)
    boundary = AUDDataCollator(
        **{**kwargs, "max_target_length": len(tokenizer.encode("hello", add_special_tokens=False))}
    )
    boundary_batch = boundary(records)
    assert boundary_batch["labels"].ne(-100).sum() > boundary.max_target_length
    random.seed(12)
    batch = collator(records)
    assert calls == [0.4]
    clean_batch = clean(records)
    assert not torch.equal(batch["input_features"], clean_batch["input_features"])
    assert torch.equal(clean(records)["input_features"], clean_batch["input_features"])
    assert torch.isfinite(batch["input_features"]).all()
    # Inference never invokes the mixer or augmenter.
    collator.generation_batch(records)
    assert calls == [0.4]


def test_raw_spectral_identity_when_probability_zero():
    augmenter = build_augmenter({"spec_augment": {"enabled": True, "p": 0.0}}, sampling_rate=16000)
    wave = np.random.default_rng(42).normal(0, 0.01, 3217).astype(np.float32)
    masked = spectral_mask_waveform(wave, augmenter)
    np.testing.assert_allclose(masked, wave, atol=1e-8, rtol=1e-5)
