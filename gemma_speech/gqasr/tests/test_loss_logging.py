"""Loss averaging across ranks, accumulation, log boundaries and resume."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gqasr.distributed import LossWindow, Runtime


def runtime(rank=0, world=1):
    # CPU-only: these diagnostics tests must not allocate GPUs on the cluster.
    rt = Runtime.__new__(Runtime)
    rt.rank, rt.world, rt.local_rank = rank, world, rank
    rt.device = torch.device("cpu")
    return rt


def test_loss_window_detaches_and_resets_after_resume():
    window = LossWindow(runtime(), start_step=7)
    for value in (2.0, 4.0, 6.0, 8.0):
        loss = torch.tensor(value, requires_grad=True)
        window.add(loss)
        loss.backward()
        assert loss.grad.item() == 1.0
    assert not window.totals.requires_grad
    assert window.flush(9) == {
        "loss": 5.0, "loss_window_start_step": 8,
        "loss_window_updates": 2, "loss_microbatches": 4,
    }
    window.add(torch.tensor(3.0))
    assert window.flush(10) == {
        "loss": 3.0, "loss_window_start_step": 10,
        "loss_window_updates": 1, "loss_microbatches": 1,
    }


def distributed_loss_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=30),
    )
    rt = runtime(rank, 2)
    try:
        window = LossWindow(rt)
        for value in ((1.0, 3.0) if rank == 0 else (5.0, 7.0)):
            window.add(torch.tensor(value))
        first = window.flush(1)
        # Unequal counts exercise SUM/count rather than averaging rank means.
        for value in ((2.0,) if rank == 0 else (10.0, 12.0)):
            window.add(torch.tensor(value))
        second = window.flush(2)
        (Path(root) / f"rank-{rank}.json").write_text(json.dumps([first, second]))
    finally:
        rt.close()


def test_two_real_gloo_ranks_reduce_loss_and_count(tmp_path):
    context = mp.spawn(distributed_loss_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Distributed loss logging timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    for rank in range(2):
        first, second = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert first["loss"] == 4.0 and first["loss_microbatches"] == 4
        assert second["loss"] == 8.0 and second["loss_microbatches"] == 3
        assert second["loss_window_start_step"] == 2


def test_trainer_logs_accumulated_loss_at_validation_final_and_resume(tmp_path, monkeypatch):
    import yaml

    from gqasr.modeling import GQASRForSpeechRecognition
    from gqasr.train import train
    from test_training import fixture

    monkeypatch.setenv("WANDB_MODE", "disabled")
    path, cfg = fixture(tmp_path)
    cfg.update(max_steps=4, logging_steps=5, eval_steps=3, save_steps=2,
               save_total_limit=4, eval_at_start=True)
    path.write_text(yaml.safe_dump(cfg))
    observed = []
    forward = GQASRForSpeechRecognition.forward

    def capture(self, *args, **kwargs):
        result = forward(self, *args, **kwargs)
        observed.append(float(result["loss"].detach()))
        return result

    monkeypatch.setattr(GQASRForSpeechRecognition, "forward", capture)
    train(path)
    records = [json.loads(line) for line in (tmp_path / "out/metrics.jsonl").read_text().splitlines()]
    assert "loss" not in records[0]  # step-zero validation is not a training loss
    assert [r["step"] for r in records] == [0, 1, 3, 4]
    assert len(observed) == 8  # four updates, two accumulated microbatches each
    for row, start, stop in zip(records[1:], (0, 2, 6), (2, 6, 8), strict=True):
        assert row["loss"] == pytest.approx(sum(observed[start:stop]) / (stop - start))
        assert row["loss_microbatches"] == stop - start
        assert row["loss_window_updates"] == (stop - start) // 2
        assert row["rank0_last_microbatch_loss"] == observed[stop - 1]

    # A checkpoint need not coincide with a log. The resumed mean explicitly
    # starts at the next update, without fabricating losses from before resume.
    observed.clear()
    cfg["output_dir"] = str(tmp_path / "resumed")
    path.write_text(yaml.safe_dump(cfg))
    train(path, tmp_path / "out/checkpoint-00000002")
    resumed = [json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in resumed] == [3, 4]
    assert resumed[0]["loss_window_start_step"] == 3
    assert resumed[0]["loss_window_updates"] == 1 and resumed[0]["loss_microbatches"] == 2
    assert resumed[0]["loss"] == pytest.approx(sum(observed[:2]) / 2)
    original = GQASRForSpeechRecognition.from_pretrained(tmp_path / "out/final")
    restarted = GQASRForSpeechRecognition.from_pretrained(tmp_path / "resumed/final")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restarted.state_dict()[key], rtol=0, atol=0)
