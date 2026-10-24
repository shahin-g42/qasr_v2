import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audturbo.sh"


def launch(*args, master="rank0.example"):
    env = dict(
        os.environ,
        AUD_DRY_RUN="1",
        AUD_MODEL="/shared/export model",
        AUD_RUN_ROOT="/shared/training runs",
    )
    env.pop("SLURM_JOB_ID", None)
    env.pop("AUD_MASTER_ADDR", None)
    if master:
        env["AUD_MASTER_ADDR"] = master
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)


def test_smoke_uses_export_and_full_uses_aligned_checkpoint():
    smoke = launch("smoke", "0")
    assert smoke.returncode == 0, smoke.stderr
    assert "nodes=8 GPUs/node=8" in smoke.stdout
    assert "audturbo_adapter_8node.yaml" in smoke.stdout
    assert "--smoke" in smoke.stdout
    assert r"/shared/export\ model" in smoke.stdout
    full = launch("full", "7", "--resume", "auto")
    assert full.returncode == 0, full.stderr
    assert r"--model /shared/training\ runs/adapter" in full.stdout
    assert r"--output /shared/training\ runs/full" in full.stdout
    assert "audturbo_full_8node.yaml" in full.stdout
    assert "--resume auto" in full.stdout


def test_nccl_has_all_64_ranks_and_invalid_launches_fail():
    result = launch("nccl", "0")
    assert result.returncode == 0, result.stderr
    assert "command=distributed-check" in result.stdout
    assert "--expected-world-size 64" in result.stdout
    assert launch("adapter", "8").returncode != 0
    assert launch("adapter", "0", master=None).returncode != 0


@pytest.mark.parametrize(
    "variant,rank,nodes,port",
    [("audturbo", "7", 8, 29591), ("audpro", "4", 5, 29601), ("audflash", "0", 1, 29611)],
)
def test_direct_launch_ignores_inherited_slurm_environment(tmp_path, variant, rank, nodes, port):
    import json

    capture = tmp_path / "args.json"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if sys.argv[1:2] == ['-m']:\n"
        "    with open(os.environ['CAPTURE'], 'w') as f:\n"
        "        json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    # These must never be invoked by a manual launch.
    for name in ("srun", "scontrol"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CAPTURE=str(capture),
        AUD_PYTHON=str(python),
        AUD_DRY_RUN="0",
        AUD_RUN_ROOT=str(tmp_path / "runs"),
        AUD_MASTER_ADDR="chosen-master",
        AUD_MASTER_PORT="29591",
        AUDPRO_RUN_ROOT=str(tmp_path / "pro-runs"),
        AUDPRO_TRITON_CACHE_DIR=str(tmp_path / "cache"),
        AUDPRO_MASTER_ADDR="chosen-master",
        AUDPRO_MASTER_PORT=str(port),
        AUDFLASH_RUN_ROOT=str(tmp_path / "flash-runs"),
        AUDFLASH_TRITON_CACHE_DIR=str(tmp_path / "flash-cache"),
        AUDFLASH_MASTER_ADDR="chosen-master",
        AUDFLASH_MASTER_PORT=str(port),
        AUD_USE_SLURM="1",
        SLURM_JOB_ID="123",
        SLURM_JOB_NODELIST="wrong-hosts",
        SLURM_JOB_NUM_NODES="12",
        SLURM_PROCID="0",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name(f"{variant}.sh")), "full", rank],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[:2] == ["-m", "torch.distributed.run"]
    assert f"--node-rank={rank}" in args
    assert f"--nnodes={nodes}" in args
    assert "--master-addr=chosen-master" in args
    assert f"--master-port={port}" in args
    assert "--nproc-per-node=8" in args


def test_audpro_defaults_are_isolated_from_audturbo(tmp_path):
    env = dict(
        os.environ,
        AUD_DRY_RUN="1",
        AUDPRO_MASTER_ADDR="pro-master",
        AUD_MODEL="/wrong/turbo",
        AUD_RUN_ROOT="/wrong/turbo-run",
        AUD_FULL_CONFIG="/wrong/turbo.yaml",
        AUD_MASTER_PORT="29591",
        AUDPRO_TRITON_CACHE_DIR=str(tmp_path / "cache"),
    )
    for key in ("AUDPRO_MODEL", "AUDPRO_RUN_ROOT", "AUDPRO_FULL_CONFIG", "AUDPRO_MASTER_PORT"):
        env.pop(key, None)

    def run(*args):
        return subprocess.run(
            ["bash", str(SCRIPT.with_name("audpro.sh")), *args],
            env=env,
            capture_output=True,
            text=True,
        )

    full = run("full", "4", "--resume", "auto")
    assert full.returncode == 0, full.stderr
    assert "nodes=5 GPUs/node=8 master=pro-master:29601" in full.stdout
    assert "audpro_full_5node.yaml" in full.stdout
    assert "models/AudPro-v1" in full.stdout
    assert "output/audpro-v1/full" in full.stdout
    assert "--resume auto" in full.stdout
    assert "/wrong/" not in full.stdout
    assert "--expected-world-size 40" in run("nccl", "0").stdout
    assert "--smoke" in run("smoke", "0").stdout
    assert run("full", "5").returncode != 0
    env.pop("AUDPRO_MASTER_ADDR")
    assert run("full", "0").returncode != 0


def test_audflash_stage_paths_and_single_node_defaults(tmp_path):
    env = dict(
        os.environ,
        AUD_DRY_RUN="1",
        AUD_MODEL="/wrong/model",
        AUD_RUN_ROOT="/wrong/run",
        AUD_MASTER_ADDR="wrong-master",
        AUD_MASTER_PORT="29591",
        AUD_ADAPTER_CONFIG="/wrong/adapter.yaml",
        AUD_FULL_CONFIG="/wrong/full.yaml",
        AUDFLASH_TRITON_CACHE_DIR=str(tmp_path / "cache"),
    )
    for key in ("MODEL", "RUN_ROOT", "MASTER_ADDR", "MASTER_PORT", "ADAPTER_CONFIG", "FULL_CONFIG"):
        env.pop("AUDFLASH_" + key, None)

    def run(*args):
        return subprocess.run(
            ["bash", str(SCRIPT.with_name("audflash.sh")), *args],
            env=env,
            capture_output=True,
            text=True,
        )

    adapter = run("adapter", "0")
    assert adapter.returncode == 0, adapter.stderr
    assert "nodes=1 GPUs/node=8 master=127.0.0.1:29611" in adapter.stdout
    assert "audflash_adapter_1node.yaml" in adapter.stdout
    assert "models/AudFlash-Gemma4-v1" in adapter.stdout
    assert "output/audflash-gemma4-v1/adapter" in adapter.stdout
    assert "/wrong/" not in adapter.stdout
    full = run("full", "0", "--resume", "auto")
    assert full.returncode == 0, full.stderr
    assert "audflash_full_1node.yaml" in full.stdout
    assert "output/audflash-gemma4-v1/adapter" in full.stdout
    assert "output/audflash-gemma4-v1/full" in full.stdout
    assert "--resume auto" in full.stdout
    assert "--smoke" in run("smoke", "0").stdout
    assert "--expected-world-size 8" in run("nccl", "0").stdout
    assert run("full", "1").returncode != 0


@pytest.mark.parametrize("encoder,stride", [("moonshine", 1), ("gemma4", 2)])
def test_audflash_rejects_incompatible_export(tmp_path, encoder, stride):
    import json
    import sys

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            dict(
                variant="audflash",
                flash_encoder=encoder,
                flash_audio_stride=stride,
            )
        )
    )
    env = dict(
        os.environ,
        AUD_PYTHON=sys.executable,
        AUDFLASH_MODEL=str(model),
        AUDFLASH_TRITON_CACHE_DIR=str(tmp_path / "cache"),
    )
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("audflash.sh")), "verify"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "fresh Gemma4 AudFlash stride-1 export" in result.stderr


def test_next_run_direct_launcher_uses_fresh_stage_sources(tmp_path):
    import json

    capture = tmp_path / "args.json"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['CAPTURE'], 'w') as f: json.dump({'args':sys.argv[1:], 'slurm':os.environ['AUD_USE_SLURM']}, f)\n"
    )
    python.chmod(0o755)
    env = dict(
        os.environ,
        AUD_PYTHON=str(python),
        CAPTURE=str(capture),
        AUD_NEXT_ROOT=str(tmp_path / "new run"),
        AUD_USE_SLURM="1",
        SLURM_JOB_ID="999",
        AUD_NNODES="12",
        AUD_MASTER_ADDR="127.0.0.1",
    )
    script = SCRIPT.with_name("audflash_next.sh")
    for stage, source in (("adaptation", "export"), ("full", "adaptation")):
        result = subprocess.run(
            ["bash", str(script), stage], env=env, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
        record = json.loads(capture.read_text())
        args = record["args"]
        assert record["slurm"] == "0" and "--nnodes=1" in args and "--nproc-per-node=8" in args
        assert args[args.index("--model") + 1] == str(tmp_path / "new run" / source)
        assert "--resume" not in args
