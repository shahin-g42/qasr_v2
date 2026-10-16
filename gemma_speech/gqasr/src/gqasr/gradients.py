"""Gradient clipping with overflow-resistant norms and actionable diagnostics."""

import torch


@torch.no_grad()
def clip_gradients(model, max_norm, runtime, step):
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("GQASR backward produced no gradients")
    # Accumulate the norms in FP64; finite FP32 gradients can overflow an FP32 L2 norm.
    norms = torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float64) for _, p in parameters])
    finite = torch.isfinite(norms).all()
    if not runtime.all_finite(torch.where(finite, norms.new_zeros(()), norms.new_full((), float("nan")))):
        bad = [name for (name, _), ok in zip(parameters, torch.isfinite(norms).tolist(), strict=True) if not ok]
        import json

        print(json.dumps({
            "event": "nonfinite_gradients", "rank": runtime.rank,
            "optimizer_step_attempt": step + 1, "parameters": bad,
            "local_gradients_finite": bool(finite),
        }), flush=True)
        raise FloatingPointError(
            f"Nonfinite GQASR gradients at update {step + 1}; "
            "see nonfinite_gradients events for affected parameters. Optimizer was not updated."
        )
    total = torch.linalg.vector_norm(norms)
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for _, p in parameters:
        p.grad.mul_(scale.to(device=p.grad.device, dtype=p.grad.dtype))
    return total
