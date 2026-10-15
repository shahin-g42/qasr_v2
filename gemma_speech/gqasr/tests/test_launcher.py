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
        "02_full_12node.yaml", "--resume /shared/checkpoint",
    ):
        assert expected in result.stdout


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


@pytest.mark.parametrize("name", ["01_bridge_12node.yaml", "02_full_12node.yaml"])
def test_training_batch_and_existing_index(name):
    cfg = load_config(ROOT / "gemma_speech/gqasr/configs" / name)
    assert cfg["batch_size"] == 20
    assert cfg["accumulation_steps"] == 1
    assert cfg["expected_world_size"] * cfg["batch_size"] == 1920
    assert cfg["language_batch_counts"] == {"ar": 10, "en": 6, "hi": 2, "ml": 1, "zh": 1}
    assert cfg["index_path"].endswith("/indexes/full_v81.sqlite")


def test_explicit_gpu_ids_override_inherited_single_gpu():
    result = launch("0", CUDA_VISIBLE_DEVICES="0", GPU_IDS="0,1,2,3,4,5,6,7")
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in result.stdout


def test_default_overrides_inherited_single_gpu():
    result = launch("0", CUDA_VISIBLE_DEVICES="3", GPU_IDS="")
    assert result.returncode == 0, result.stderr
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 " in result.stdout
