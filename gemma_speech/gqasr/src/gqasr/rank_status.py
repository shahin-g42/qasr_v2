"""Best-effort startup observations; never a training synchronization gate."""
import json
import os
import socket
import threading
from collections import Counter


class RankStatus:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.enabled = os.environ.get("GQASR_RANK_STATUS", "1") != "0"
        self.record = {"rank": self.rank, "host": socket.gethostname(), "reached": []}
        self.stop = threading.Event()
        self.finished = False
        self.thread = None

    def mark(self, phase):
        if not self.enabled or self.finished:
            return
        self.record = {**self.record, "phase": phase,
                       "reached": [*self.record["reached"], phase]}
        print(json.dumps({"event": "rank_startup", **self.record}), flush=True)
        if phase == "first_update_complete":
            self.finished = True

    def attach(self):
        """Called only after ordinary process-group initialization returns."""
        if not self.enabled or self.world == 1:
            return
        try:
            import torch.distributed as dist
            from torch.distributed.distributed_c10d import _get_default_store
            store = dist.PrefixStore("gqasr-observed-startup/", _get_default_store())
            self.thread = threading.Thread(target=self._observe, args=(store,), daemon=True)
            self.thread.start()
        except Exception as exc:
            print(f"[GQASR] Aggregate rank status unavailable: {type(exc).__name__}", flush=True)

    def _observe(self, store):
        last = None
        ticks = 0
        try:
            while not self.stop.is_set():
                record = self.record
                if record is not last:
                    store.set(str(self.rank), json.dumps(record))
                    last = record
                if self.rank == 0 and ticks % 15 == 0:
                    records = []
                    for rank in range(self.world):
                        key = str(rank)
                        if store.check([key]):
                            records.append(json.loads(store.get(key)))
                    print(json.dumps(summarize(records, self.world)), flush=True)
                    if len(records) == self.world and all(
                        "first_update_complete" in r["reached"] for r in records
                    ):
                        return
                ticks += 1
                self.stop.wait(1)
        except Exception as exc:
            print(f"[GQASR] Aggregate rank status stopped: {type(exc).__name__}", flush=True)

    def close(self):
        self.stop.set()


def summarize(records, world):
    seen = {r["rank"] for r in records}
    completed = sum("first_update_complete" in r["reached"] for r in records)
    return {
        "event": "rank_startup_summary", "world_size": world,
        "reported": len(seen), "not_reported": sorted(set(range(world)) - seen),
        "joined": sum("joined" in r["reached"] for r in records),
        "first_batch_ready": sum("first_forward" in r["reached"] for r in records),
        "first_update_complete": completed,
        "awaiting_first_update": world - completed,
        "phases": dict(Counter(r["phase"] for r in records)),
        "pending": [{"rank": r["rank"], "host": r["host"], "phase": r["phase"]}
                    for r in records if "first_update_complete" not in r["reached"]],
    }
