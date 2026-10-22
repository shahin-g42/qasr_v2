import json

import pytest
import soundfile as sf
import torch
import yaml
from test_audio_types import artifact
from test_training import data_recipe

from audm.memory_probe import prepare
from audm.processing import AUDProcessor
from audm.train import run
from audm.training_config import load_recipe


def test_probe_has_max_shapes_and_keeps_training_contract(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    source = artifact(tmp_path, tokenizer, "audflash")
    recipe = tmp_path / "full.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage="full",
                robustness=dict(
                    non_speech=dict(probability=0.03, manifests=["${UNSET_UNUSED_ASSET}"])
                ),
                training=dict(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    per_device_eval_batch_size=1,
                    expected_world_size=1,
                    dataloader_num_workers=4,
                    bf16=False,
                    tf32=False,
                    deepspeed=None,
                    gradient_checkpointing=True,
                    label_smoothing=0.1,
                    max_duration_seconds=1.0,
                ),
            )
        )
    )
    before = recipe.read_bytes()
    report = prepare(recipe, source, tmp_path / "probe")
    cfg, stage, receipt = load_recipe(report["config"])
    assert stage == "full" and cfg.max_steps == 2
    assert cfg.per_device_train_batch_size == 1 and cfg.gradient_accumulation_steps == 2
    assert cfg.dataloader_num_workers == 4
    assert cfg.gradient_checkpointing and cfg.label_smoothing == 0.1
    assert not cfg.eval_manifest_specs and not cfg.aud_robustness
    assert cfg.code_switch_probability == 0
    assert all(not settings["enabled"] for settings in cfg.augmentation.values())
    records = [
        json.loads(line) for line in (tmp_path / "probe/synthetic.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    record = records[0]
    wave, sr = sf.read(record["audio_filepath"])
    assert len(wave) / sr == report["audio_seconds"] == cfg.max_duration_seconds
    processor = AUDProcessor.from_pretrained(source)
    count = len(processor.tokenizer.encode(record["text"], add_special_tokens=False))
    assert cfg.max_target_length - 8 <= count <= cfg.max_target_length
    assert recipe.read_bytes() == before
    run(report["config"])
    state = json.loads((tmp_path / "probe/run/trainer_state.json").read_text())
    assert state["global_step"] == 2
    with pytest.raises(FileExistsError):
        prepare(recipe, source, tmp_path / "probe")
