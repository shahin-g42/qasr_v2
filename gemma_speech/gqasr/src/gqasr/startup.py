"""Prepare cold data workers before entering timed training collectives."""

from collections import deque
from contextlib import contextmanager
import faulthandler
import json
import os
import socket
import time

import torch
import torch.distributed as dist


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

    def __enter__(self):
        # This catches time inside Process.start()/pickle and native reads,
        # which DataLoader's batch-delivery timeout does not cover.
        faulthandler.dump_traceback_later(180, repeat=True)
        return self

    def mark(self, phase):
        if not self.active:
            return
        self.phase = phase
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
            faulthandler.dump_traceback_later(180, repeat=True)
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
                faulthandler.dump_traceback_later(180, repeat=True)
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
