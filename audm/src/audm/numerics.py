"""Persist tensor-only failing batches for numerical debugging; never model weights."""

import json
from pathlib import Path

import torch


def save_failure(directory, inputs, *, rank, step, error):
    directory = Path(directory) / "numerics"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"rank-{rank}-step-{step}"
    tensors = {
        name: value.detach().cpu().clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    torch.save(tensors, prefix.with_suffix(".pt"))
    report = {
        "rank": rank,
        "global_step": step,
        "error": str(error),
        "batch": str(prefix.with_suffix(".pt")),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all()),
            }
            for name, value in tensors.items()
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def require_all_ranks(condition, description, details=None):
    """One scalar collective, followed by a coordinated failure on every rank."""
    import logging

    flag = condition.detach().all().to(dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if not bool(flag):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logging.getLogger("audm").error(
            "Numerical guard: rank=%s %s details=%s; optimizer update blocked",
            rank, description, details,
        )
        raise FloatingPointError(f"AUD {description}; optimizer update blocked on all ranks")


def check_loss(loss):
    require_all_ranks(torch.isfinite(loss.detach()), "nonfinite training loss")


@torch.no_grad()
def check_gradients(model):
    """DDP/plain Trainer: inspect the accumulated norm before clipping/step."""
    parameters = list(model.parameters())
    grads = [p.grad for p in parameters if p.grad is not None]
    device = parameters[0].device
    if grads:
        norm = torch.nn.utils.get_total_norm(grads)
        valid = torch.isfinite(norm)
    else:
        norm = None
        valid = torch.tensor(False, device=device)
    require_all_ranks(valid, "nonfinite or missing gradients", norm)


def install_zero2_guard(optimizer):
    """Validate native ZeRO-2 group norms before clipping or ANY group update.

    ZeRO may replace a NaN/Inf group norm with -1, then hide it in the L2 norm
    of the group norms. Capture those original results and reject the sentinel.
    Accelerate calls engine.step() inside backward(), so Trainer's usual
    on_pre_optimizer_step callback is too late. No extra full-gradient scan.
    """
    if getattr(optimizer, "_aud_numerics_guard", False):
        return
    norm_name = (
        "complete_grad_norm_calculation_for_cpu_offload"
        if getattr(optimizer, "cpu_offload", False)
        else "get_grad_norm_direct"
    )
    if (
        not getattr(optimizer, "partition_gradients", False)
        or not callable(getattr(optimizer, norm_name, None))
        or not callable(getattr(optimizer, "scaled_global_norm", None))
        or not getattr(optimizer, "bit16_groups", None)
    ):
        raise RuntimeError("Unsupported DeepSpeed optimizer: cannot install AUD ZeRO-2 numerical guard")
    check_overflow = getattr(optimizer, "check_overflow", None)
    if getattr(optimizer, "check_grad_overflow", False):
        if not callable(check_overflow):
            raise RuntimeError("Unsupported ZeRO-2 overflow API")
        def guarded_overflow(*args, **kwargs):
            result = check_overflow(*args, **kwargs)
            require_all_ranks(
                torch.tensor(not optimizer.overflow, device=optimizer.device),
                "DeepSpeed gradient overflow",
            )
            return result
        optimizer.check_overflow = guarded_overflow
    group_norm = getattr(optimizer, norm_name)
    global_norm = optimizer.scaled_global_norm
    observed = []

    def capture(*args, **kwargs):
        norm = group_norm(*args, **kwargs)
        observed.append(norm)
        return norm

    def guarded_global_norm(*args, **kwargs):
        observed.clear()
        total = global_norm(*args, **kwargs)
        if not isinstance(total, torch.Tensor):
            raise RuntimeError("Unexpected ZeRO-2 norm API: expected tensor")
        values = (
            torch.stack([torch.as_tensor(n, device=total.device).reshape(()) for n in observed])
            if observed else total.new_empty(0)
        )
        valid = torch.isfinite(values).all() & values.ge(0).all() & torch.isfinite(total).all()
        valid = valid & (len(observed) == len(optimizer.bit16_groups))
        require_all_ranks(valid, "invalid ZeRO-2 gradient norm", values.detach())
        return total

    setattr(optimizer, norm_name, capture)
    optimizer.scaled_global_norm = guarded_global_norm
    optimizer._aud_numerics_guard = True
