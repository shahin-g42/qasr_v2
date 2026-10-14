import os
from contextlib import nullcontext

import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, precision="fp32"):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        if (
            precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("GPU lacks BF16 support; use precision=fp32")
        self.precision = precision
        if self.world > 1:
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")

    @property
    def main(self):
        return self.rank == 0

    def autocast(self):
        return (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def gather(self, value):
        if self.world == 1:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def all_finite(self, value):
        finite = torch.isfinite(value).all().to(device=self.device, dtype=torch.int32)
        if self.world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite)

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
