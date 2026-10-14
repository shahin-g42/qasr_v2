"""Single-node NCCL smoke test for preflight: one process per local GPU.

Launched by preflight.sh with JEV_PREFLIGHT_SMOKE=1 via torchrun. Verifies the
local fabric can actually move the payload class the trainer uses (FP32
gradient-shard all-reduces) and reports achieved bus bandwidth plus free
device memory; it is a sanity measurement, not a substitute for the 8/16/32/
64-GPU qualification jobs the plan requires.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    elements = 64 * 1024 * 1024  # 256 MiB of FP32, gradient-shard sized
    buffer = torch.randn(elements, device="cuda", dtype=torch.float32)

    for _ in range(5):  # warm up the communicator and allocation reuse
        dist.all_reduce(buffer)
    torch.cuda.synchronize()
    iterations = 20
    started = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(buffer)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    if rank == 0:
        # Bus-bandwidth model from the NCCL tests: each rank ends up with the
        # sum, so (world-1)/world of the payload crosses the interconnect per
        # iteration, counted once for send and once for receive.
        bytes_moved = 2 * (world - 1) / world * elements * 4 * iterations
        free, total = torch.cuda.mem_get_info()
        print(
            f"nccl all_reduce: world={world} {iterations}x{elements * 4 // 2**20}MiB "
            f"in {seconds:.3f}s ~{bytes_moved / seconds / 1e9:.1f} GB/s bus bandwidth"
        )
        print(f"device {local_rank} memory: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
