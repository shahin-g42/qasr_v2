import os
import subprocess
from pathlib import Path

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


def test_direct_launch_ignores_inherited_slurm_environment(tmp_path):
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
        AUD_USE_SLURM="1",
        SLURM_JOB_ID="123",
        SLURM_JOB_NODELIST="wrong-hosts",
        SLURM_JOB_NUM_NODES="12",
        SLURM_PROCID="0",
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "adapter", "7"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[:2] == ["-m", "torch.distributed.run"]
    assert "--node-rank=7" in args
    assert "--nnodes=8" in args
    assert "--master-addr=chosen-master" in args
    assert "--master-port=29591" in args
    assert "--nproc-per-node=8" in args
