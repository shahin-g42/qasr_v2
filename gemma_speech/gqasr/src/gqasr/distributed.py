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
            if self.device.type == "cuda":
                # Bind NCCL to LOCAL_RANK explicitly, including its first barrier.
                # Global-rank-based device inference is unsafe for differing GPU visibility.
                dist.init_process_group("nccl", device_id=self.device)
            else:
                dist.init_process_group("gloo")

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

    def sum(self, value):
        result = value.clone()
        if self.world > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class LossWindow:
    """Mean unscaled microbatch loss across ranks since the last log/resume.

    Each microbatch's loss is already a mean over its supervised tokens. Equal
    microbatch weighting matches the current DDP/accumulation objective; this
    is not a token-weighted corpus loss. All values are detached diagnostics.
    """

    def __init__(self, runtime, start_step=0):
        self.runtime = runtime
        self.start_step = start_step
        self.totals = torch.zeros(2, device=runtime.device, dtype=torch.float32)

    def add(self, loss):
        self.totals[0].add_(loss.detach().float())
        self.totals[1].add_(1)

    def flush(self, step):
        # All ranks call this at the same logging boundaries, even though only
        # rank zero writes the record. One two-scalar reduction per log window.
        total, count = self.runtime.sum(self.totals).tolist()
        if count <= 0 or step <= self.start_step:
            raise ValueError("Loss window requires completed training updates")
        record = {
            "loss": total / count,
            "loss_window_start_step": self.start_step + 1,
            "loss_window_updates": step - self.start_step,
            "loss_microbatches": int(count),
        }
        self.totals.zero_()
        self.start_step = step
        return record
