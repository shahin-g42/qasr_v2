"""Prepare cold data workers before entering timed training collectives."""

from collections import deque
from contextlib import contextmanager
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import time

import torch
import torch.distributed as dist


def worker_context(workers):
    """Use a fresh CPU forkserver when requested; never fork the CUDA trainer."""
    if not workers:
        return None
    method = os.environ.get("GQASR_WORKER_START_METHOD", "spawn")
    if method not in {"spawn", "forkserver"}:
        raise ValueError("GQASR_WORKER_START_METHOD must be spawn or forkserver")
    if method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"DataLoader start method {method!r} is unavailable on this platform")
    if method == "forkserver":
        # Exclude __main__: the server only imports the CPU data pipeline, not
        # the CLI/trainer. Imports are amortized across this rank's workers.
        multiprocessing.set_forkserver_preload(["gqasr.worker_preload"])
    return multiprocessing.get_context(method)


def worker_ready(worker_id):
    """Report completion of deserialization, before the first batch is read."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU data worker unexpectedly initialized CUDA")
    method = multiprocessing.get_start_method()
    preload = sys.modules.get("gqasr.worker_preload")
    server_pid = getattr(preload, "PRELOAD_PID", None)
    if method == "forkserver" and (server_pid is None or server_pid == os.getpid()):
        raise RuntimeError("DataLoader forkserver preload was not inherited; "
                           "start a fresh trainer process")
    print(json.dumps({
        "event": "data_worker_ready", "rank": int(os.environ.get("RANK", "0")),
        "worker": worker_id, "pid": os.getpid(), "start_method": method,
        "preload_server_pid": server_pid,
    }), flush=True)


class Startup:
    """One-time readiness via the rendezvous store, without a GPU collective.

    The normal NCCL timeout is unchanged. A separate, bounded startup deadline
    reports which ranks and phases are missing instead of timing out in a loss
    reduction while another node is still constructing DataLoader workers.
    """

    def __init__(self, runtime, timeout=None):
        self.rt = runtime
        self.timeout = float(os.environ.get("GQASR_STARTUP_TIMEOUT", "1200")
                             if timeout is None else timeout)
        if not 0 < self.timeout < float("inf"):
            raise ValueError("GQASR_STARTUP_TIMEOUT must be finite and positive")
        self.store = None
        if runtime.world > 1:
            from torch.distributed.distributed_c10d import _get_default_store

            restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
            self.store = dist.PrefixStore(f"gqasr-startup-v1/{restart}/", _get_default_store())
        self.active = True
        self.phase = "created"
        self.started = time.monotonic()
        self.trace_file = None

    def __enter__(self):
        directory = os.environ.get("GQASR_LOG_DIR")
        if directory:
            path = Path(directory) / f"startup-rank-{self.rt.rank}-pid-{os.getpid()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_file = path.open("a", buffering=1)
            print(json.dumps({"event": "training_startup_trace", "rank": self.rt.rank,
                              "hostname": socket.gethostname(), "path": str(path)}), flush=True)
        # This catches time inside Process.start()/pickle and native reads,
        # which DataLoader's batch-delivery timeout does not cover.
        self.arm_trace()
        return self

    def arm_trace(self):
        faulthandler.dump_traceback_later(180, repeat=True, file=self.trace_file)

    def mark(self, phase):
        if not self.active:
            return
        self.phase = phase
        if self.trace_file is not None:
            self.trace_file.write(f"phase={phase} elapsed={time.monotonic() - self.started:.2f}s\n")
        if self.store is not None:
            self.store.set(f"phase/{self.rt.rank}", phase)

    def missing(self, keys):
        result = {}
        for rank, key in enumerate(keys):
            if not self.store.check([key]):
                phase = f"phase/{rank}"
                result[str(rank)] = (
                    self.store.get(phase).decode() if self.store.check([phase]) else "not_started"
                )
        return result

    def wait(self, stage):
        self.mark(stage)
        # Waiting ranks already have explicit progress. Dump stacks only while
        # doing startup work, so slow workers/computation get the useful traces.
        faulthandler.cancel_dump_traceback_later()
        elapsed = round(time.monotonic() - self.started, 2)
        print(json.dumps({"event": "training_startup_ready", "rank": self.rt.rank,
                          "hostname": socket.gethostname(), "stage": stage,
                          "elapsed_seconds": elapsed}), flush=True)
        if self.store is None:
            self.arm_trace()
            return
        keys = [f"{stage}/{rank}" for rank in range(self.rt.world)]
        self.store.set(keys[self.rt.rank], "ready")
        started, last_progress = time.monotonic(), 0.0
        while True:
            if self.store.check(["failed"]):
                raise RuntimeError("Training startup failed: " + self.store.get("failed").decode())
            if self.store.check(keys):
                if self.rt.main:
                    print(json.dumps({"event": "training_startup_all_ready", "stage": stage,
                                      "ready_ranks": self.rt.world}), flush=True)
                self.arm_trace()
                return
            now = time.monotonic()
            if now - started >= self.timeout:
                raise TimeoutError(f"Training startup {stage} exceeded {self.timeout:g}s; "
                                   f"waiting ranks/phases: {self.missing(keys)}")
            if self.rt.main and now - last_progress >= 15:
                missing = self.missing(keys)
                print(json.dumps({"event": "training_startup_wait", "stage": stage,
                                  "ready_ranks": self.rt.world - len(missing),
                                  "world_size": self.rt.world, "waiting": missing,
                                  "elapsed_seconds": round(now - started, 1)}), flush=True)
                last_progress = now
            time.sleep(0.2)

    def finish(self):
        self.active = False
        faulthandler.cancel_dump_traceback_later()
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None and self.active and self.store is not None:
            try:
                self.store.set("failed", json.dumps({
                    "rank": self.rt.rank, "phase": self.phase,
                    "error": f"{exc_type.__name__}: {exc}"[:2000],
                }))
            except Exception:
                pass  # Preserve the original error if rendezvous has also failed.
        self.finish()


@contextmanager
def prepared_batches(loader, startup, accumulation):
    """Warm every worker and retain the first batches, in their original order."""
    iterator = None
    try:
        startup.mark("dataloader_spawn")
        print(json.dumps({
            "event": "data_workers_starting", "rank": startup.rt.rank,
            "workers": loader.num_workers,
            "start_method": (loader.multiprocessing_context.get_start_method()
                             if loader.num_workers else None),
        }), flush=True)
        iterator = iter(loader)
        startup.mark("dataloader_first_batches")
        # At least one actual batch per worker exercises imports, SQLite, audio
        # decoding, augmentation and the frontend. Also prepare the first update.
        count = min(len(loader), max(1, loader.num_workers, accumulation))
        buffered = deque(next(iterator) for _ in range(count))
        startup.wait("data_ready")

        def batches():
            while buffered:
                yield buffered.popleft()
            yield from iterator

        yield batches()
    finally:
        if iterator is not None and loader.num_workers:
            iterator._shutdown_workers()


def first_update_ready(startup):
    if startup.active:
        startup.mark("first_update_device_sync")
        if startup.rt.device.type == "cuda":
            torch.cuda.synchronize(startup.rt.device)
        startup.wait("first_update_ready")
        startup.finish()
