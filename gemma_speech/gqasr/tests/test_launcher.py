import os
import subprocess
from pathlib import Path

import pytest
from gqasr.config import load_config

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts/run_gqasr_12node.sh"


def launch(*args, **overrides):
    env = dict(os.environ)
    for key in (
        "MASTER_ADDR", "MASTER_PORT", "NODE_RANK", "NNODES", "NPROC_PER_NODE",
        "GQASR_CONFIG", "PROJECT_DIR",
    ):
        env.pop(key, None)
    env.update(DRY_RUN="1", **overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True,
    )


def test_bridge_launch():
    result = launch("0")
    assert result.returncode == 0, result.stderr
    for expected in (
        "--nnodes=12", "--nproc_per_node=8", "--node_rank=0",
        "--master_addr=inception-H100-hpc-051", "--master_port=29611",
        "--rdzv_backend=static", "timeout=7200", "01_bridge_12node.yaml",
    ):
        assert expected in result.stdout


def test_full_launch_and_resume():
    result = launch("11", "master.example", "--stage", "full", "--resume", "/shared/checkpoint")
    assert result.returncode == 0, result.stderr
    for expected in (
        "--node_rank=11", "--master_addr=master.example", "--master_port=29612",
        "02_full_b40_12node.yaml", "--resume /shared/checkpoint",
    ):
        assert expected in result.stdout


def test_encoder_continuation_launch():
    result = launch("0", "--stage", "encoder")
    assert result.returncode == 0, result.stderr
    assert "03_encoder12_b80_12node.yaml" in result.stdout
    assert "--master_port=29613" in result.stdout
    assert "--resume" not in result.stdout
    result = launch("11", "--stage", "encoder", "--preflight")
    assert result.returncode == 0, result.stderr
    assert "03_encoder12_b80_12node.yaml" in result.stdout and "torchrun" not in result.stdout


def test_encoder_batch40_override():
    config = ROOT / "gemma_speech/gqasr/configs/03_encoder12_b40_12node.yaml"
    result = launch("0", "--stage", "encoder", GQASR_CONFIG=str(config))
    assert result.returncode == 0, result.stderr
    assert str(config) in result.stdout
    assert "03_encoder12_b80_12node.yaml" not in result.stdout


def test_preflight_does_not_launch_training():
    result = launch("0", "--preflight")
    assert result.returncode == 0, result.stderr
    assert "gqasr preflight --config" in result.stdout
    assert "torchrun" not in result.stdout


@pytest.mark.parametrize("args", [(), ("12",), ("-1",), ("abc",), ("0", "--stage", "bad")])
def test_invalid_arguments(args):
    assert launch(*args).returncode != 0


def test_wrong_world_size_rejected():
    assert launch("0", NNODES="1").returncode != 0


@pytest.mark.parametrize("name", ["01_bridge_12node.yaml", "02_full_12node.yaml", "02_full_b40_12node.yaml"])
def test_training_batch_and_existing_index(name):
    cfg = load_config(ROOT / "gemma_speech/gqasr/configs" / name)
    full = name.startswith("02")
    large = "_b40_" in name
    assert cfg["batch_size"] == (40 if large else 20)
    assert cfg["accumulation_steps"] == 1
    assert cfg["expected_world_size"] * cfg["batch_size"] == (3840 if large else 1920)
    assert cfg["language_batch_counts"] == (
        {"ar": 20, "en": 12, "hi": 4, "ml": 2, "zh": 2} if large else
        {"ar": 10, "en": 6, "hi": 2, "ml": 1, "zh": 1}
    )
    if full:
        assert cfg["acoustic_layers"] == 0
        assert cfg["max_steps"] == (31446 if large else 62891)
        assert cfg["warmup_steps"] == (417 if large else 833)
        assert cfg["decay_steps"] == (3125 if large else 6250)
        assert cfg["eval_steps"] == cfg["save_steps"] == (1042 if large else 2083)
    assert cfg["index_path"].endswith("/indexes/full_v81.sqlite")


def test_explicit_gpu_ids_override_inherited_single_gpu():
    result = launch("0", CUDA_VISIBLE_DEVICES="0", GPU_IDS="0,1,2,3,4,5,6,7")
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in result.stdout


def test_default_overrides_inherited_single_gpu():
    result = launch("0", CUDA_VISIBLE_DEVICES="3", GPU_IDS="")
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 " in result.stdout


def test_batch20_override():
    config = ROOT / "gemma_speech/gqasr/configs/02_full_12node.yaml"
    result = launch("0", "--stage", "full", GQASR_CONFIG=str(config))
    assert result.returncode == 0, result.stderr
    assert str(config) in result.stdout
    assert "02_full_b40_12node.yaml" not in result.stdout


def test_prepare_preserves_both_batch_recipes(tmp_path):
    from gqasr.config import prepare

    prepare(ROOT / "configs/v8.1/02_full_12node.yaml", tmp_path)
    small = load_config(tmp_path / "02_full_12node.yaml")
    large = load_config(tmp_path / "02_full_b40_12node.yaml")
    for cfg, batch in ((small, 20), (large, 40)):
        assert cfg["batch_size"] == batch
        assert cfg["accumulation_steps"] == 1
        assert cfg["acoustic_layers"] == 0
        assert sum(cfg["language_batch_counts"].values()) == batch
    assert small["index_path"] == large["index_path"]
    assert small["output_dir"] != large["output_dir"]
    assert abs(small["max_steps"] * 20 - large["max_steps"] * 40) <= 20
