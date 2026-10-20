import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from gqasr.config import load_config
from gqasr.export_checkpoint import digest, export_checkpoint
from gqasr.modeling import GQASRForSpeechRecognition
from gqasr.train import train
from test_training import artifact, fixture

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_export_preserves_weights_and_starts_fresh_optimizer(tmp_path, monkeypatch):
    path, cfg = fixture(tmp_path)
    cfg.update(acoustic_layers=2, decoder_layers=2, train_embeddings=True)
    path.write_text(yaml.safe_dump(cfg))
    train(path)
    source = tmp_path / "out/checkpoint-00000001"
    saved = torch.load(source / "training_state.pt", weights_only=True)
    assert all(state["step"].item() == 1 for state in saved["optimizer"]["state"].values())
    original_state_hash = digest(source / "training_state.pt")
    original_load = torch.load

    def no_training_state_load(file, *args, **kwargs):
        if str(file).endswith("training_state.pt"):
            pytest.fail("Fresh export/training must not load optimizer or RNG state")
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", no_training_state_load)
    destination = tmp_path / "exported"
    result = export_checkpoint(source, destination)
    assert result["source_optimizer_updates"] == 1
    assert not result["optimizer_state_included"] and not result["rng_state_included"]
    assert result["finite_weights"] and result["expected_infinite_clip_bounds"]
    assert digest(destination / "model.safetensors") == digest(source / "model.safetensors")
    assert digest(source / "training_state.pt") == original_state_hash
    assert not (destination / "training_state.pt").exists()
    assert not (destination / "training_config.json").exists()
    for name in ("config.json", "preprocessor_config.json", "tokenizer/tokenizer.json"):
        assert digest(destination / name) == digest(source / name)
    checksums = json.loads((destination / "checksums.json").read_text())
    assert all(digest(destination / name) == expected for name, expected in checksums.items())
    original_model = GQASRForSpeechRecognition.from_pretrained(source)
    exported_model = GQASRForSpeechRecognition.from_pretrained(destination)
    for key, value in original_model.state_dict().items():
        torch.testing.assert_close(value, exported_model.state_dict()[key], rtol=0, atol=0)

    before_first_step = []
    original_step = torch.optim.AdamW.step

    def capture(optimizer, *args, **kwargs):
        if not before_first_step:
            assert not optimizer.state
            before_first_step.append([group["lr"] for group in optimizer.param_groups])
            assert all(group["lr"] == group["base_lr"] / 2 for group in optimizer.param_groups)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", capture)
    cfg.update(base_model=str(destination), output_dir=str(tmp_path / "fresh"),
               warmup_steps=2, max_steps=2, workers=1, eval_at_start=True, logging_steps=1)
    path.write_text(yaml.safe_dump(cfg))
    train(path)
    assert before_first_step
    assert not (tmp_path / "fresh/resume.json").exists()
    metrics = [json.loads(line) for line in (tmp_path / "fresh/metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in metrics] == [0, 1, 2]
    fresh = original_load(tmp_path / "fresh/final/training_state.pt", weights_only=True)
    assert fresh["step"] == 2
    assert all(state["step"].item() == 2 for state in fresh["optimizer"]["state"].values())


def test_export_rejects_overwrite_and_incomplete_source(tmp_path):
    source = tmp_path / "source"
    artifact(source)
    output = tmp_path / "export"
    export_checkpoint(source, output)
    before = digest(output / "model.safetensors")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, output)
    assert digest(output / "model.safetensors") == before
    with pytest.raises(ValueError, match="separate"):
        export_checkpoint(source, source / "nested")
    (source / "preprocessor_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        export_checkpoint(source, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_export_rejects_nonfinite_weights_without_publishing(tmp_path, bad_value):
    source = tmp_path / "source"
    artifact(source)
    save_file({"gqasr_bad": torch.tensor([bad_value])}, source / "model.safetensors")
    with pytest.raises(ValueError, match="Nonfinite"):
        export_checkpoint(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.glob(".gqasr-checkpoint-export-*"))
    assert not (tmp_path / ".export.export.lock").exists()


@pytest.mark.parametrize("bad_value", [float("nan"), -float("inf")])
def test_export_rejects_invalid_clipping_buffers(tmp_path, bad_value):
    source = tmp_path / "source"
    artifact(source)
    tensors = load_file(source / "model.safetensors")
    key = next(key for key in tensors if key.endswith(".input_max"))
    tensors[key] = torch.tensor(bad_value)
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(ValueError, match="Nonfinite"):
        export_checkpoint(source, tmp_path / "export")


def test_fresh_config_and_launcher_never_resume():
    cfg = load_config(ROOT / "gemma_speech/gqasr/configs/04_encoder12_b120_fresh_4node.yaml")
    assert cfg["max_steps"] == 31446 and cfg["warmup_steps"] == 1000
    assert cfg["batch_size"] == 120 and cfg["accumulation_steps"] == 1
    assert cfg["expected_world_size"] == 32 and cfg["workers"] == 8
    assert cfg["prefetch_factor"] == 2 and not cfg["online_code_switching"]
    assert cfg["base_model"].endswith("/models/gqasr-encoder12-step7500")
    assert cfg["output_dir"].endswith("/outputs/encoder12_b120_4node_fresh7500")
    assert cfg["training_data_seed"] > 43 + 7500 * 96
    env = dict(os.environ, DRY_RUN="1", NNODES="12", GQASR_LAUNCH_NNODES="12",
               GQASR_CONFIG="/stale/resume.yaml", GQASR_RESUME_CHECKPOINT="/stale/checkpoint")
    for key in ("PROJECT_DIR", "GQASR_FRESH_CONFIG", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    script = str(ROOT / "scripts/start_gqasr_encoder_4node.sh")
    for rank in range(4):
        result = subprocess.run(["bash", script, str(rank)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        for value in ("--nnodes=4", "--nproc_per_node=8", "--master_port=29614",
                      "04_encoder12_b120_fresh_4node.yaml", f"--node_rank={rank}"):
            assert value in result.stdout
        assert "--resume" not in result.stdout and "--allow-" not in result.stdout
        assert "/stale/" not in result.stdout
    result = subprocess.run(["bash", script, "0", "--resume"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
