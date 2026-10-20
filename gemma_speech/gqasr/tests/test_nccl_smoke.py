"""Exercise real CPU collectives, failure reporting and the hard hang deadline."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "gemma_speech/gqasr/src/gqasr/nccl_smoke.py"
LAUNCHER = ROOT / "scripts/test_gqasr_nccl.sh"


def launch_preview(*args, **overrides):
    env = {key: value for key, value in os.environ.items()
           if key not in {"NNODES", "NPROC_PER_NODE", "MASTER_ADDR", "GEMMA_SPEECH_ENV"}
           and not key.startswith("NCCL_TEST_")}
    env.update(DRY_RUN="1", **overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=10,
    )


def test_four_node_launcher_has_separate_port_and_no_restarts():
    result = launch_preview("3", "master.example", MASTER_PORT="29611")
    assert result.returncode == 0, result.stderr
    for expected in (
        "--nnodes=4", "--nproc_per_node=8", "--node_rank=3",
        "--master_addr=master.example", "--master_port=29691",
        "--rdzv_endpoint=master.example:29691", "--rdzv_conf=timeout=180",
        "--max_restarts=0", "--tee=3", "nccl_smoke.py",
    ):
        assert expected in result.stdout
    assert "--master_port=29611" not in result.stdout


def test_launcher_supports_local_and_cpu_comparisons():
    result = launch_preview(
        "0", "127.0.0.1", "--backend", "gloo", "--iterations", "2",
        NNODES="1", NPROC_PER_NODE="2", NCCL_TEST_PORT="29692",
        NCCL_TEST_PYTHON="/tmp/path with spaces/python",
    )
    assert result.returncode == 0, result.stderr
    assert "--nnodes=1" in result.stdout and "--nproc_per_node=2" in result.stdout
    assert "--backend gloo --iterations 2" in result.stdout
    assert "/tmp/path\\ with\\ spaces/python" in result.stdout


@pytest.mark.parametrize("args,overrides", [
    ((), {}),
    (("4", "master"), {}),
    (("-1", "master"), {}),
    (("0",), {}),
    (("0", "master"), {"NNODES": "1", "NPROC_PER_NODE": "1"}),
    (("0", "master"), {"NCCL_TEST_PORT": "70000"}),
    (("0", "master"), {"NCCL_TEST_RDZV_TIMEOUT": "0"}),
])
def test_launcher_rejects_invalid_topology(args, overrides):
    assert launch_preview(*args, **overrides).returncode == 2


def run_workers(tmp_path, prelude=""):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch Gloo required")
    worker = SOURCE
    if prelude:
        worker = tmp_path / "worker.py"
        worker.write_text(
            prelude + "\nimport runpy\n"
            f"runpy.run_path({str(SOURCE)!r}, run_name='__main__')\n"
        )
    logs = tmp_path / "logs"
    env = dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    # torchrun supplies worker ranks; do not reuse the caller's rendezvous.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes=1", "--nproc_per_node=2", "--max_restarts=0",
            "--rdzv_backend=static", "--master_addr=127.0.0.1", f"--master_port={port}",
            "--rdzv_conf=timeout=10",
            "--log-dir", str(logs), "--redirects=3", str(worker),
            "--backend", "gloo", "--timeout", "10", "--iterations", "2", "--tensor-mib", "1",
        ],
        env=env, capture_output=True, text=True, timeout=50,
    )
    records = []
    for path in logs.rglob("stdout.log"):
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                records.append(json.loads(line))
    stderr = result.stderr + "\n".join(path.read_text() for path in logs.rglob("stderr.log"))
    return result.returncode, records, stderr


def test_real_two_rank_gloo_collectives(tmp_path):
    code, records, stderr = run_workers(tmp_path)
    assert code == 0, stderr
    passed = [r for r in records if r["status"] == "PASS"]
    assert len(passed) == 2
    assert all(r["world_size"] == 2 and not r["nccl_tested"] for r in passed)
    assert len([r for r in records if r["phase"] == "payload_all_reduce" and r["status"] == "ok"]) == 4
    assert len([r for r in records if r["phase"] == "broadcast" and r["status"] == "ok"]) == 2


def test_different_worker_options_fail_before_tensor_collectives(tmp_path):
    code, records, stderr = run_workers(tmp_path, """
import os
import sys
if os.environ["RANK"] == "1":
    sys.argv.extend(["--iterations", "3"])
""")
    assert code != 0
    failed = [r for r in records if r["status"] == "FAIL"]
    assert failed, stderr
    assert all(r["phase"] == "rank_inventory" and "arguments differ" in r["error"] for r in failed)
    assert not any(r["phase"] == "initial_barrier" or r["status"] == "PASS" for r in records)


def test_bad_collective_result_is_not_reported_as_pass(tmp_path):
    code, records, stderr = run_workers(tmp_path, """
import torch.distributed as dist
original = dist.all_reduce
def corrupt(tensor, *args, **kwargs):
    original(tensor, *args, **kwargs)
    tensor.add_(1)
dist.all_reduce = corrupt
""")
    assert code != 0
    failed = [r for r in records if r["status"] == "FAIL"]
    assert failed, stderr
    assert all(r["phase"] == "scalar_all_reduce" and "expected 3" in r["error"] for r in failed)
    assert not any(r["status"] == "PASS" for r in records)


def test_default_nccl_check_never_silently_falls_back_to_cpu():
    pytest.importorskip("torch")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RANK="0", LOCAL_RANK="0",
               WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    result = subprocess.run(
        [sys.executable, str(SOURCE)], env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert records[-1]["status"] == "FAIL"
    assert records[-1]["phase"] == "local_device"
    assert "CUDA/NCCL unavailable" in records[-1]["error"]
    assert not any(r["status"] == "PASS" for r in records)


def test_hard_watchdog_exits_a_stuck_worker():
    code = f"""
import runpy
import time
module = runpy.run_path({str(SOURCE)!r})
progress = module["Progress"]("gloo", 1)
with progress.stage("deliberately_stuck"):
    time.sleep(120)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=25,
    )
    assert result.returncode != 0
    assert '"phase": "deliberately_stuck"' in result.stdout
    assert "Timeout" in result.stderr
    assert '"status": "ok"' not in result.stdout
