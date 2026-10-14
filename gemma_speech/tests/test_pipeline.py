import importlib
import json
import shutil

import pytest
import torch

from gemma_speech.checkpoint import load_checkpoint
from gemma_speech.cli import inspect_config
from gemma_speech.inference import synthesize, transcribe
from gemma_speech.smoke import prepare_smoke
from gemma_speech.train import train


@pytest.mark.parametrize("task", ["asr", "tts"])
def test_training_checkpoint_resume_is_exact_and_offline(tmp_path, monkeypatch, task):
    directory, configs = prepare_smoke(tmp_path / "run")
    inspect_config(configs[task], validate=True)
    module = importlib.import_module("gemma_speech.train")
    save = module.save_checkpoint

    def save_and_keep(*args, **kwargs):
        save(*args, **kwargs)
        if args[7] == 2:
            shutil.copy2(args[0], directory / "step2.pt")

    monkeypatch.setattr(module, "save_checkpoint", save_and_keep)
    train(configs[task])
    # Disable all network access for checkpoint inference and continuation.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    original, _, _ = load_checkpoint(directory / task / "last.pt")
    config = json.loads(configs[task].read_text())
    config["output_dir"] = str(directory / "resumed")
    resumed_config = directory / "resume.json"
    resumed_config.write_text(json.dumps(config))
    train(resumed_config, resume=directory / "step2.pt")
    resumed, _, _ = load_checkpoint(directory / "resumed/last.pt")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    if task == "asr":
        report = transcribe(
            directory / task / "last.pt",
            directory / "validation.jsonl",
            directory / "predictions.jsonl",
        )
        assert report["scores"]["strict/all"]["utterances"] == 2
    else:
        result = synthesize(
            directory / task / "last.pt",
            "مرحبا",
            "test-speaker",
            "ar",
            directory / "mel.pt",
            seconds=0.5,
            steps=2,
        )
        assert result["frames"] == 47
    with pytest.raises(FileExistsError):
        train(configs[task])
