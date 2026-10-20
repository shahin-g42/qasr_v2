"""Host-wall timing of batch delivery; no extra CUDA synchronization."""

import time


class BatchTimer:
    def __init__(self, iterator):
        self.iterator = iterator
        self.wait_seconds = 0.0
        self.batches = 0

    def __iter__(self):
        return self

    def __next__(self):
        start = time.monotonic()
        batch = next(self.iterator)
        self.wait_seconds += time.monotonic() - start
        self.batches += 1
        return batch

    def flush(self, updates):
        result = {
            "rank0_data_wait_seconds_per_update": self.wait_seconds / updates,
            "rank0_data_wait_seconds_per_microbatch": self.wait_seconds / self.batches,
        }
        self.wait_seconds = 0.0
        self.batches = 0
        return result
