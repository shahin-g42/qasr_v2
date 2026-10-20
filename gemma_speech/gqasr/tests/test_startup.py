"""Startup imbalance must not consume the first training collective's timeout."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from gqasr.startup import Startup, prepared_batches, first_update_ready


ROOT = Path(__file__).resolve().parents[3]


def runtime(rank=0, world=1):
    return SimpleNamespace(rank=rank, world=world, main=rank == 0, device=torch.device("cpu"))


class WorkerNumbers(Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, i):
        # This PID proves the warmup exercised distinct spawned data workers.
        return i, os.getpid()


def test_actual_spawned_workers_are_warmed_without_losing_or_reordering_batches():
    loader = DataLoader(WorkerNumbers(), batch_size=1, num_workers=2,
                        multiprocessing_context="spawn", persistent_workers=True,
                        prefetch_factor=1, timeout=30)
    with Startup(runtime()) as startup:
        with prepared_batches(loader, startup, accumulation=4) as batches:
            assert startup.phase == "data_ready"
            workers = list(loader._iterator._workers)
            results = list(batches)
            assert [int(row[0]) for row in results] == list(range(9))
            assert len({int(row[1]) for row in results[:4]}) == 2
        assert all(not worker.is_alive() for worker in workers)


def test_worker_read_failure_shuts_down_children():
    class FailingLoader:
        num_workers = 2
        def __len__(self):
            return 4
        def __iter__(self):
            return self
        def __next__(self):
            raise OSError("synthetic unreadable audio")
        def _shutdown_workers(self):
            self.closed = True

    loader = FailingLoader()
    with pytest.raises(OSError, match="unreadable audio"):
        with Startup(runtime()) as startup:
            with prepared_batches(loader, startup, accumulation=4):
                pytest.fail("must not enter training")
    assert loader.closed


def test_missing_rank_reports_phase_and_startup_failure():
    with pytest.raises(TimeoutError, match="dataloader_spawn"):
        with Startup(runtime(), timeout=0.1) as startup:
            startup.rt = runtime(world=2)
            startup.store = dist.HashStore()
            startup.store.set("phase/1", "dataloader_spawn")
            startup.wait("data_ready")
    failure = json.loads(startup.store.get("failed"))
    assert failure["rank"] == 0 and failure["phase"] == "data_ready"


def test_peer_failure_is_reported_without_waiting_for_collective_timeout():
    with pytest.raises(RuntimeError, match="synthetic audio error"):
        with Startup(runtime(), timeout=5) as startup:
            startup.rt = runtime(world=2)
            startup.store = dist.HashStore()
            startup.store.set("failed", "rank 1: synthetic audio error")
            startup.wait("data_ready")


def delayed_rank(rank, root, gated):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(root) / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=1.5))
    result = {}
    try:
        with Startup(runtime(rank, 2), timeout=15) as startup:
            # Reproduce a slow data-worker startup greater than the collective
            # timeout. The default timeout is identical in fixed/unfixed cases.
            if rank == 1:
                time.sleep(3)
            if gated:
                startup.wait("data_ready")
                # Also cover cold first-update computation after data readiness.
                if rank == 1:
                    time.sleep(3)
                first_update_ready(startup)
            value = torch.tensor(float(rank + 1))
            dist.all_reduce(value)
            result = {"passed": True, "sum": value.item()}
    except RuntimeError as exc:
        result = {"passed": False, "error": str(exc)}
    finally:
        dist.destroy_process_group()
        (Path(root) / f"rank-{rank}.json").write_text(json.dumps(result))


@pytest.mark.parametrize("gated", [False, True])
def test_slow_rank_reproduces_timeout_then_passes_with_startup_readiness(tmp_path, gated):
    context = mp.spawn(delayed_rank, args=(str(tmp_path), gated), nprocs=2, join=False)
    deadline = time.monotonic() + 40
    try:
        while not context.join(timeout=1):
            if not gated and (tmp_path / "rank-0.json").exists():
                # The intentionally broken peer can remain inside Gloo after
                # rank zero aborts. Stop that test-owned process in finally.
                break
            if time.monotonic() > deadline:
                pytest.fail("startup regression hung")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    ranks = range(2) if gated else [0]
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in ranks]
    assert all(result["passed"] is gated for result in results)
    if gated:
        assert all(result["sum"] == 3 for result in results)
    else:
        assert "Timed out" in results[0]["error"]


def test_recovery_launcher_replaces_inherited_thread_oversubscription(tmp_path):
    # Only the --check path is executed with a fake Python that prints its env.
    envdir = tmp_path / "env/bin"
    envdir.mkdir(parents=True)
    python = envdir / "python"
    python.write_text('#!/bin/bash\nprintf "%s %s %s %s\\n" "$OMP_NUM_THREADS" '
                      '"$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMBA_NUM_THREADS"\n')
    python.chmod(0o755)
    env = dict(os.environ, GEMMA_SPEECH_ENV=str(envdir.parent),
               OMP_NUM_THREADS="24", OPENBLAS_NUM_THREADS="32", MKL_NUM_THREADS="32")
    env.pop("GQASR_CPU_THREADS", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/resume_gqasr_encoder_12node.sh"), "check"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "1 1 1 1"
