"""Standard PyTorch gradient clipping with failure diagnostics."""

import json

import torch


@torch.no_grad()
def clip_gradients(model, max_norm, runtime, step):
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("GQASR backward produced no gradients")
    # These are the same two operations used by PyTorch clip_grad_norm_.
    # Check the ordinary norm before modifying gradients. Detailed parameter
    # scans are only needed on failure, not hundreds of extra kernels per update.
    norm = torch.nn.utils.get_total_norm([p.grad for _, p in parameters])
    if not runtime.all_finite(norm):
        flags = torch.stack([torch.isfinite(p.grad).all() for _, p in parameters])
        bad = [name for (name, _), ok in zip(parameters, flags.tolist(), strict=True) if not ok]
        print(json.dumps({
            "event": "nonfinite_gradients", "rank": runtime.rank,
            "optimizer_step_attempt": step + 1, "parameters": bad,
            "local_gradients_finite_before_clipping": bool(flags.all()),
            "local_norm_finite": bool(torch.isfinite(norm)),
        }), flush=True)
        raise FloatingPointError(
            f"Nonfinite GQASR gradients at update {step + 1}; "
            "see nonfinite_gradients events. Optimizer was not updated."
        )
    torch.nn.utils.clip_grads_with_norm_([p for _, p in parameters], max_norm, norm)
    return norm
