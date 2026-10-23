"""Verify rank-local numerical failures stop every rank before any update."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("RUN_AUD_DDP_TESTS") != "1", reason="Requires local Gloo sockets")
def test_rank_local_failures_abort_every_rank(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("""
import os
import torch
from audm.numerics import check_loss, check_gradients, install_zero2_guard
from test_numerics import Zero2

torch.distributed.init_process_group("gloo")
rank = torch.distributed.get_rank()
for mode in ("loss", "gradients", "zero2"):
    bad = float("nan") if rank == 1 else 1.
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    before = [p.detach().clone() for p in model.parameters()]
    raised = False
    try:
        if mode == "loss":
            check_loss(torch.tensor(bad))
        elif mode == "gradients":
            model(torch.ones(1, 1)).sum().backward()
            model.weight.grad.fill_(bad)
            check_gradients(model)
        else:
            zero = Zero2([-1. if rank == 1 else 1.])
            install_zero2_guard(zero)
            zero.step()
        optimizer.step()
    except FloatingPointError:
        raised = True
    assert raised, (rank, mode)
    assert not optimizer.state
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    if mode == "zero2":
        assert not zero.optimizer.state
        assert torch.equal(zero.parameter, torch.ones(1))
    torch.distributed.barrier()
check_loss(torch.tensor(1.))
torch.distributed.destroy_process_group()
""")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")
    ])
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1", f"--master-port={port}", "--nnodes=1", "--nproc-per-node=2", str(script)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
